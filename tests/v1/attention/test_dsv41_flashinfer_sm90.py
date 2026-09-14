# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4.1 SM90 cache addressing, exact planning lengths and attention parity.

CPU references exercise the two-call merge; SM90 cases use real FlashInfer.
"""

from types import SimpleNamespace

import pytest
import torch

# isort: off
import vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse as fi_dsv41_mod
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM90Attention,
)
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse_sm90 import (
    DeepseekSparseSWAFlashInferSM90MetadataBuilder,
    DeepseekV4FlashInferSM90MetadataBuilder,
    DeepseekV4FlashInferSM90SparseBackend,
)

# isort: on
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.merge_attn_states import _merge_attn_states_torch
from vllm.v1.kv_cache_interface import (
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
    create_kv_cache_views,
)

BLOCK = 512
WINDOW = 64
TOPK = 2048
NUM_HEADS = 4


class FakeWrapper:
    def __init__(self):
        self.run_calls = []

    def run(self, q_nope, q_pe, ckv, kpe, **kwargs):
        self.run_calls.append((q_nope, q_pe, ckv, kpe, kwargs))
        num_tokens, num_heads, head_dim = q_nope.shape
        out = torch.ones(num_tokens, num_heads, ckv.shape[-1], dtype=torch.bfloat16)
        # The real wrapper returns [num_tokens, num_heads] fp32 LSE.
        lse = torch.full((num_tokens, num_heads), 0.5, dtype=torch.float32)
        return out, lse


class FakeState:
    """Stands in for _SM90State (which imports flashinfer at __init__)."""

    def __init__(self, width):
        self.topk_width = width
        self.kv_indices = torch.zeros(1024 * width, dtype=torch.int32)
        self.wrapper = FakeWrapper()


def make_attn(kv_dtype=torch.bfloat16):
    attn = object.__new__(DeepseekV4FlashInferSM90Attention)
    attn.compress_ratio = 1
    attn.n_local_heads = NUM_HEADS
    attn.head_dim = BLOCK
    attn.kv_cache_torch_dtype = kv_dtype
    attn._sm90_ckv_scale = 1.0
    attn.topk_indices_buffer = torch.full((64, TOPK), -1, dtype=torch.int32)
    attn.kv_cache = torch.zeros(8, 32, BLOCK, dtype=kv_dtype)
    attn.attn_sink = torch.zeros(NUM_HEADS, dtype=torch.float32)
    attn.is_kv_source = True
    attn.compressed_cache_prefix = "l0.attn"
    attn._static_forward_context = {"l0.attn": attn}
    swa_layer = SimpleNamespace(
        prefix="l0.attn.swa_cache",
        kv_cache=torch.zeros(16, 32, BLOCK, dtype=kv_dtype),
    )
    attn.swa_cache_layer = swa_layer
    return attn, swa_layer


def make_swa_metadata(ndt, npt, width=WINDOW):
    return SimpleNamespace(
        num_decode_tokens=ndt,
        num_prefill_tokens=npt,
        decode_swa_indices=torch.arange(ndt * width, dtype=torch.int32).reshape(
            ndt, 1, width
        ),
        prefill_swa_indices=(
            torch.arange(npt * width, dtype=torch.int32).reshape(npt, 1, width) + 1000
            if npt
            else None
        ),
        token_to_req_indices=torch.zeros(ndt + npt, dtype=torch.int32),
        is_valid_token=torch.ones(ndt + npt, dtype=torch.bool),
        flashinfer_sm90_swa_state=None,
    )


def make_fake_topk(*_args, **_kwargs):
    topk = torch.full((64, 16), -1, dtype=torch.int32)
    topk[:, :16] = torch.arange(16, dtype=torch.int32)
    lens = torch.full((64,), 16, dtype=torch.int32)
    return topk, lens


def test_two_call_mixed_batch(monkeypatch):
    """Mixed decode+prefill on a compressed layer: call A carries SWA rows for
    both splits, call B carries the converted top-k rows, and the partials are
    merged with LSE rescaling before the sink correction."""
    attn, swa_layer = make_attn()
    ndt, npt = 2, 4
    num_tokens = ndt + npt
    swa_metadata = make_swa_metadata(ndt, npt)
    swa_state = FakeState(WINDOW)
    swa_metadata.flashinfer_sm90_swa_state = swa_state
    topk_state = FakeState(TOPK)
    flashmla_metadata = SimpleNamespace(
        num_reqs=1,
        block_size=64,
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        flashinfer_sm90_topk_state=topk_state,
    )
    ctx = SimpleNamespace(
        attn_metadata={
            swa_layer.prefix: swa_metadata,
            attn.compressed_cache_prefix: flashmla_metadata,
        }
    )
    monkeypatch.setattr(fi_dsv41_mod, "get_forward_context", lambda: ctx)

    fake_topk = torch.full((num_tokens, TOPK), -1, dtype=torch.int32)
    fake_topk[:, :16] = torch.arange(16, dtype=torch.int32)
    monkeypatch.setattr(
        fi_dsv41_mod,
        "compute_global_topk_indices_and_lens",
        lambda *a, **k: (fake_topk, torch.full((num_tokens,), 16, dtype=torch.int32)),
    )
    merged_calls = []

    def fake_merge(output, p_out, p_lse, s_out, s_lse, output_lse=None):
        merged_calls.append((p_out.shape, s_out.shape, p_lse.shape, s_lse.shape))
        output.copy_(p_out + s_out)

    monkeypatch.setattr(fi_dsv41_mod, "merge_attn_states", fake_merge)
    monkeypatch.setattr(
        type(attn), "_apply_sink_correction", lambda self, out, lse: None
    )

    q = torch.randn(num_tokens, NUM_HEADS, BLOCK, dtype=torch.bfloat16)
    output = torch.zeros_like(q)
    attn.forward_mqa(q, None, None, output)

    # Call A: SWA rows = [decode rows; prefill rows], clamped.
    rows_a = swa_state.kv_indices.view(-1, swa_state.topk_width)[:num_tokens]
    assert torch.equal(rows_a[:ndt], swa_metadata.decode_swa_indices.reshape(ndt, -1))
    assert torch.equal(rows_a[ndt:], swa_metadata.prefill_swa_indices.reshape(npt, -1))
    # Call B: the converted global top-k rows (masked tails clamped to a
    # valid slot by the forward, as planned rows never read past their
    # per-row length).
    rows_b = topk_state.kv_indices.view(-1, TOPK)[:num_tokens]
    assert torch.equal(rows_b, fake_topk.clamp_(min=0))

    swa_run = swa_state.wrapper.run_calls[0]
    topk_run = topk_state.wrapper.run_calls[0]
    # NoPE kernel mode: full 512 head as q_nope, zero-width rope/cache kpe.
    assert swa_run[0].shape == (num_tokens, NUM_HEADS, BLOCK)
    assert swa_run[1].shape == (num_tokens, NUM_HEADS, 0)
    assert swa_run[2].shape == (16 * 32, 1, BLOCK) and swa_run[3].shape[-1] == 0
    assert topk_run[2].shape == (8 * 32, 1, BLOCK) and topk_run[3].shape[-1] == 0
    # Merge consumes both partials and the merged LSE feeds the sink pass.
    assert len(merged_calls) == 1
    assert output.shape == (num_tokens, NUM_HEADS, BLOCK)


def test_swa_only_single_call(monkeypatch):
    """SWA-only layers (compress_ratio == 0) skip the top-k call and merge."""
    attn, swa_layer = make_attn()
    attn.compress_ratio = 0
    ndt = 3
    swa_metadata = make_swa_metadata(ndt, 0)
    swa_state = FakeState(WINDOW)
    swa_metadata.flashinfer_sm90_swa_state = swa_state
    ctx = SimpleNamespace(attn_metadata={swa_layer.prefix: swa_metadata})
    monkeypatch.setattr(fi_dsv41_mod, "get_forward_context", lambda: ctx)
    merged_calls = []
    monkeypatch.setattr(
        fi_dsv41_mod, "merge_attn_states", lambda *a, **k: merged_calls.append(a)
    )

    q = torch.randn(ndt, NUM_HEADS, BLOCK, dtype=torch.bfloat16)
    output = torch.zeros_like(q)
    attn.forward_mqa(q, None, None, output)

    assert len(swa_state.wrapper.run_calls) == 1
    assert merged_calls == []
    rows_a = swa_state.kv_indices.view(-1, swa_state.topk_width)[:ndt]
    assert (rows_a[:, :WINDOW] == torch.arange(ndt * WINDOW).reshape(ndt, WINDOW)).all()
    # Sink correction math: wrapper out=1, lse=0.5, sink=0 →
    # out * sigmoid(lse - sink) = sigmoid(0.5).
    expected = torch.sigmoid(torch.tensor(0.5))
    assert torch.allclose(output, output.new_full(output.shape, expected))


def test_run_wrapper_args():
    """FP8 KV passes per-tensor scales; bf16 passes none. The compressed cache
    is addressed as flat rows (page_size=1) split at the full row width."""
    attn, _ = make_attn(kv_dtype=torch.float8_e4m3fn)
    attn._sm90_ckv_scale = 0.5
    state = FakeState(TOPK)
    q = torch.randn(5, NUM_HEADS, BLOCK, dtype=torch.bfloat16)

    attn._run_wrapper(state, q, attn.kv_cache.reshape(-1, 1, BLOCK))

    q_nope, q_pe, ckv, kpe, kwargs = state.wrapper.run_calls[0]
    assert q_nope.shape == (5, NUM_HEADS, BLOCK) and q_pe.shape == (5, NUM_HEADS, 0)
    assert ckv.shape == (8 * 32, 1, BLOCK) and kpe.shape[-1] == 0
    assert kwargs == {
        "return_lse": True,
        "return_lse_base_on_e": True,
        "ckv_scale": 0.5,
        "kpe_scale": 1.0,
    }


def _make_builder(cls, **attrs):
    builder = object.__new__(cls)
    for key, value in attrs.items():
        setattr(builder, key, value)
    return builder


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("with_positions", [False, True])
def test_topk_lengths_use_device_boundaries_and_mask_padding(
    compress_ratio, with_positions
):
    builder = _make_builder(
        DeepseekV4FlashInferSM90MetadataBuilder,
        _index_topk=8,
        compress_ratio=compress_ratio,
        req_id_per_token_buffer=torch.empty(6, dtype=torch.int32),
    )
    # CPU boundaries and optimistic lengths deliberately differ from the device.
    positions = torch.tensor([0, 1, 16, 17, 18, 19])
    cam = SimpleNamespace(
        num_actual_tokens=6,
        num_reqs=2,
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 6], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([5, 23], dtype=torch.int32),
        seq_lens=torch.tensor([2, 20], dtype=torch.int32),
        positions=positions if with_positions else None,
        slot_mapping=torch.tensor([0, 1, 2, 3, 4, -1]),
        token_to_req_indices=lambda buffer: torch.tensor([0, 0, 1, 1, 1, 1]),
    )
    num_rows, lens = builder._topk_lens_host(cam)
    assert num_rows == 6
    assert lens.tolist() == (
        [1, 2, 8, 8, 8, 0] if compress_ratio == 1 else [0, 1, 8, 8, 8, 0]
    )


@pytest.mark.parametrize("prefill_lens", [None, [4, 80, 0]])
def test_swa_plan_uses_index_kernel_lengths(monkeypatch, prefill_lens):
    metadata = SimpleNamespace(
        num_decode_tokens=2,
        num_prefill_tokens=0 if prefill_lens is None else len(prefill_lens),
        decode_swa_lens=torch.tensor([69, 0], dtype=torch.int32),
        prefill_swa_lens=(
            None
            if prefill_lens is None
            else torch.tensor(prefill_lens, dtype=torch.int32)
        ),
    )
    builder_cls = DeepseekSparseSWAFlashInferSM90MetadataBuilder
    monkeypatch.setattr(builder_cls.__bases__[0], "build", lambda *a: metadata)
    calls = []
    state = SimpleNamespace(plan=lambda n, lens: calls.append((n, lens)))
    builder = _make_builder(builder_cls, state=state)
    result = builder.build(0, None)
    assert result.flashinfer_sm90_swa_state is state
    expected = [69, 0] + (prefill_lens or [])
    assert calls[0][0] == len(expected)
    assert calls[0][1].tolist() == expected
    assert calls[0][1].device.type == "cpu"


def test_backend_gates(monkeypatch):
    backend = DeepseekV4FlashInferSM90SparseBackend
    assert backend.supports_compute_capability(SimpleNamespace(major=9))
    assert not backend.supports_compute_capability(SimpleNamespace(major=10))
    assert backend.supports_sink()

    call = lambda kv="fp8", major=9: backend.supports_combination(
        head_size=BLOCK,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv,
        block_size=64,
        use_mla=True,
        has_sink=True,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=SimpleNamespace(major=major),
    )
    import vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse_sm90 as backend_mod

    monkeypatch.setattr(backend_mod, "has_flashinfer_sm90_nope_mla", lambda: True)
    monkeypatch.setattr(backend_mod, "has_flashinfer_sm90_mla_lse", lambda: True)
    # No V3 kv_lora_rank field or active VllmConfig is needed for full-row NoPE.
    assert call() is None
    assert "SM90" in (call(major=10) or "")
    assert "fp8_ds_mla" in (call(kv="fp8_ds_mla") or "")


@pytest.mark.parametrize(
    "enum_member, path_suffix",
    [
        (
            AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41_SM90,
            "flashinfer_sparse_sm90.DeepseekV4FlashInferSM90SparseBackend",
        ),
    ],
)
def test_registry_enum_path(enum_member, path_suffix):
    assert enum_member.get_path().endswith(path_suffix)


def test_dtype_canonicalization():
    assert (
        DeepseekV4FlashInferSM90Attention._canonicalize_kv_cache_dtype("auto", None)
        == "fp8"
    )
    assert (
        DeepseekV4FlashInferSM90Attention._canonicalize_kv_cache_dtype("fp8", None)
        == "fp8"
    )
    assert (
        DeepseekV4FlashInferSM90Attention._canonicalize_kv_cache_dtype("bfloat16", None)
        == "bfloat16"
    )
    # Base class leaves values untouched.
    assert DeepseekV4Attention._canonicalize_kv_cache_dtype("auto", None) == "auto"
    with pytest.raises(ValueError, match="plain BF16/FP8"):
        DeepseekV4FlashInferSM90Attention._canonicalize_kv_cache_dtype(
            "fp8_ds_mla", None
        )


def make_packed_cache(dtype, block_size, device="cpu"):
    spec = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=BLOCK, dtype=dtype
    )
    page_bytes = block_size * BLOCK * dtype.itemsize
    stride = page_bytes + 512
    raw = torch.zeros(3 * stride, dtype=torch.int8, device=device)
    tensor = KVCacheTensor(
        size=raw.numel(),
        layers=["kv"],
        layer_stride=page_bytes,
        block_stride=stride,
        offset=512,
    )
    cache = create_kv_cache_views(raw, spec, 3, KVCacheLayout.BLHNC, tensor)[0]
    cache = cache.squeeze(1)
    cache.copy_(torch.randn(cache.shape, device=device).to(dtype))
    return cache


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_interleaved_cache_page_indices_alias_original_rows(dtype):
    attn, _ = make_attn(dtype)
    cache = make_packed_cache(dtype, 32)
    assert not cache.is_contiguous()
    ckv, block_stride, token_stride = attn._flat_ckv(cache)
    assert ckv.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    slots = torch.tensor([[0, 31, 32, 63, 95, -1]], dtype=torch.int32)
    indices = torch.empty_like(slots)
    attn._copy_page_indices(indices, slots, 32, block_stride, token_stride)
    selected = ckv.float()[indices.long(), 0]
    logical = slots.clamp(min=0).long()
    expected = cache.float()[logical // 32, logical % 32]
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)


class ReferenceWrapper(FakeWrapper):
    """Compute partial attention from the pages passed by forward_mqa."""

    def __init__(self, state, lengths):
        super().__init__()
        self.state = state
        self.lengths = lengths

    def run(self, q_nope, q_pe, ckv, kpe, **kwargs):
        self.run_calls.append((q_nope, q_pe, ckv, kpe, kwargs))
        out = torch.zeros_like(q_nope)
        lse = torch.full(q_nope.shape[:2], -torch.inf, device=q_nope.device)
        indices = self.state.kv_indices.view(-1, self.state.topk_width)
        for row, length in enumerate(self.lengths):
            if length == 0:
                continue
            kv = ckv.float()[indices[row, :length].long(), 0]
            logits = q_nope[row].float() @ kv.T * BLOCK**-0.5
            out[row] = (logits.softmax(-1) @ kv).to(out.dtype)
            lse[row] = logits.logsumexp(-1)
        return out, lse


@pytest.mark.parametrize("compress_ratio", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_flashinfer", [False, True], ids=["cpu", "sm90"])
def test_interleaved_two_call_attention_matches_joint_softmax(
    monkeypatch, compress_ratio, dtype, use_flashinfer
):
    if use_flashinfer:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("requires SM90")
        pytest.importorskip("flashinfer")
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import _SM90State

    device = "cuda" if use_flashinfer else "cpu"
    torch.manual_seed(0)
    attn, swa_layer = make_attn(dtype)
    attn.compress_ratio = compress_ratio
    swa_layer.kv_cache = make_packed_cache(dtype, 32, device)
    block_size = 64 // max(compress_ratio, 1)
    attn.kv_cache = make_packed_cache(dtype, block_size, device)
    attn.attn_sink = torch.tensor([-torch.inf, -2.0, 0.5, 3.0], device=device)
    # Equal token/head counts detect ambiguous LSE transposes; the final row is padding.
    q = torch.randn(4, NUM_HEADS, BLOCK, dtype=torch.bfloat16, device=device)
    slots_a = torch.tensor(
        [[0, 31, 32], [1, 33, 95], [0, 32, 64], [-1, -1, -1]],
        dtype=torch.int32,
        device=device,
    )
    slots_b = torch.tensor(
        [[-1, -1], [0, block_size], [block_size - 1, 2 * block_size], [-1, -1]],
        dtype=torch.int32,
        device=device,
    )

    def state_for(width, lengths):
        if use_flashinfer:
            state = _SM90State(
                torch.device(device), NUM_HEADS, dtype, 4, width, BLOCK, 0, BLOCK**-0.5
            )
            state.plan(4, torch.tensor(lengths, dtype=torch.int32))
        else:
            state = FakeState(width)
            state.wrapper = ReferenceWrapper(state, lengths)
        return state

    swa_metadata = make_swa_metadata(2, 2, width=3)
    swa_metadata.decode_swa_indices = slots_a[:2].unsqueeze(1)
    swa_metadata.prefill_swa_indices = slots_a[2:].unsqueeze(1)
    swa_metadata.flashinfer_sm90_swa_state = state_for(3, [3, 3, 3, 0])
    topk_state = state_for(2, [0, 2, 2, 0]) if compress_ratio else None
    metadata = {
        swa_layer.prefix: swa_metadata,
        attn.compressed_cache_prefix: SimpleNamespace(
            num_reqs=2,
            block_size=64,
            block_table=torch.zeros(2, 3),
            flashinfer_sm90_topk_state=topk_state,
        ),
    }
    monkeypatch.setattr(
        fi_dsv41_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    monkeypatch.setattr(
        fi_dsv41_mod, "compute_global_topk_indices_and_lens", lambda *a: (slots_b, None)
    )
    if not use_flashinfer:
        monkeypatch.setattr(fi_dsv41_mod, "merge_attn_states", _merge_attn_states_torch)
    output = torch.empty_like(q)
    attn.forward_mqa(q, None, None, output)

    expected = torch.zeros_like(q)
    for row in range(3):
        swa_slots = slots_a[row].long()
        kv = swa_layer.kv_cache.float()[swa_slots // 32, swa_slots % 32]
        if compress_ratio:
            topk = slots_b[row][slots_b[row] >= 0].long()
            kv = torch.cat(
                [kv, attn.kv_cache.float()[topk // block_size, topk % block_size]]
            )
        logits = q[row].float() @ kv.T * BLOCK**-0.5
        weights = torch.cat([logits, attn.attn_sink[:, None]], dim=-1).softmax(-1)
        expected[row] = (weights[:, :-1] @ kv).to(q.dtype)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=1e-2)


def test_selection_logic(monkeypatch):
    """The DSV41 SM90 enum selects the SM90 attention class on SM90 and
    rejects other architectures; generic FlashInfer enums error out."""
    from vllm.models.deepseek_v4_1.nvidia.model import _select_dsv4_attn_cls
    from vllm.platforms.interface import DeviceCapability

    backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41_SM90
    config = SimpleNamespace(attention_config=SimpleNamespace(backend=backend))
    import vllm.models.deepseek_v4_1.nvidia.model as model_mod

    monkeypatch.setattr(
        model_mod,
        "current_platform",
        SimpleNamespace(get_device_capability=lambda: DeviceCapability(9, 0)),
    )
    assert _select_dsv4_attn_cls(config) is DeepseekV4FlashInferSM90Attention

    config.attention_config.backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90
    with pytest.raises(ValueError, match="not a DeepSeek V4.1 attention backend"):
        _select_dsv4_attn_cls(config)
