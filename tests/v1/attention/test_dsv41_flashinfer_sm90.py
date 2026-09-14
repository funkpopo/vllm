# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the DeepSeek V4.1 FlashInfer SM90 backend wiring (no GPU).

The FlashInfer wrappers, top-k conversion, and LSE merge are replaced by CPU
recorders; the tests pin the two-call contract: SWA rows (decode + prefill)
feed call A, converted global top-k rows feed call B, the partials merge via
LSE rescaling, and the sink is applied as a post-correction. Also pins the
host-side length formulas the builders plan with and the backend gates.
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
        out = torch.zeros(num_tokens, num_heads, ckv.shape[-1], dtype=torch.bfloat16)
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
        type(attn), "_apply_sink_correction", lambda self, o, l: None
    )

    q = torch.randn(num_tokens, NUM_HEADS, BLOCK)
    output = torch.zeros_like(q)
    attn.forward_mqa(q, None, None, output)

    # Call A: SWA rows = [decode rows; prefill rows], clamped.
    rows_a = swa_state.kv_indices.view(-1, swa_state.topk_width)[:num_tokens]
    assert torch.equal(rows_a[:ndt], swa_metadata.decode_swa_indices.reshape(ndt, -1))
    assert torch.equal(rows_a[ndt:], swa_metadata.prefill_swa_indices.reshape(npt, -1))
    # Call B: the converted global top-k rows.
    rows_b = topk_state.kv_indices.view(-1, TOPK)[:num_tokens]
    assert torch.equal(rows_b, fake_topk)

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

    q = torch.randn(ndt, NUM_HEADS, BLOCK)
    output = torch.zeros_like(q)
    attn.forward_mqa(q, None, None, output)

    assert len(swa_state.wrapper.run_calls) == 1
    assert merged_calls == []
    rows_a = swa_state.kv_indices.view(-1, swa_state.topk_width)[:ndt]
    assert (rows_a[:, :WINDOW] == torch.arange(ndt * WINDOW).reshape(ndt, WINDOW)).all()


def test_run_wrapper_args():
    """FP8 KV passes per-tensor scales; bf16 passes none. The compressed cache
    is addressed as flat rows (page_size=1) split at the full row width."""
    attn, _ = make_attn(kv_dtype=torch.float8_e4m3fn)
    attn._sm90_ckv_scale = 0.5
    state = FakeState(TOPK)
    q = torch.randn(5, NUM_HEADS, BLOCK)

    attn._run_wrapper(state, q, attn.kv_cache.reshape(-1, 1, BLOCK))

    q_nope, q_pe, ckv, kpe, kwargs = state.wrapper.run_calls[0]
    assert q_nope.shape == (5, NUM_HEADS, BLOCK) and q_pe.shape == (5, NUM_HEADS, 0)
    assert ckv.shape == (8 * 32, 1, BLOCK) and kpe.shape[-1] == 0
    assert kwargs == {"ckv_scale": 0.5, "kpe_scale": 1.0}


def _make_builder(cls, **attrs):
    builder = object.__new__(cls)
    builder._async_scheduling = False
    for key, value in attrs.items():
        setattr(builder, key, value)
    return builder


def test_swa_lens_host_causal():
    builder = _make_builder(
        DeepseekSparseSWAFlashInferSM90MetadataBuilder,
        _is_dspark=False,
        window_size=WINDOW,
        decode_threshold=1,
    )
    cam = SimpleNamespace(
        num_reqs=3,
        query_start_loc_cpu=torch.tensor([0, 5, 7, 10], dtype=torch.int32),
        seq_lens=torch.tensor([100, 9, 3000], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([100, 9, 3000], dtype=torch.int32),
        positions=None,
        causal=True,
    )
    num_rows, lens = builder._swa_lens_host(cam)
    assert num_rows == 10
    # ctx 96..100, 8..9, 2998..3000 all clamped to the window.
    assert lens.tolist() == [WINDOW] * 5 + [8, 9] + [WINDOW] * 3


def test_swa_lens_host_noncausal_dspark():
    builder = _make_builder(
        DeepseekSparseSWAFlashInferSM90MetadataBuilder,
        _is_dspark=True,
        window_size=WINDOW,
        decode_threshold=6,
    )
    cam = SimpleNamespace(
        num_reqs=2,
        max_query_len=6,
        query_start_loc_cpu=torch.tensor([0, 6, 12], dtype=torch.int32),
        seq_lens=torch.tensor([100, 9], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([100, 9], dtype=torch.int32),
        positions=None,
        causal=False,
    )
    num_rows, lens = builder._swa_lens_host(cam)
    assert num_rows == 12
    # Non-causal decode rows (block-anchored): min(seq_len, window + q_len)
    # = min(100, 70) for req0, min(9, 70) for req1.
    assert lens[:6].tolist() == [70] * 6
    assert lens[6:].tolist() == [9] * 6


def test_topk_lens_host_ratio1():
    builder = _make_builder(
        DeepseekV4FlashInferSM90MetadataBuilder,
        _index_topk=2048,
        _async_scheduling=False,
        compress_ratio=1,
    )
    cam = SimpleNamespace(
        num_reqs=2,
        query_start_loc_cpu=torch.tensor([0, 5, 10], dtype=torch.int32),
        seq_lens=torch.tensor([100, 3000], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([100, 3000], dtype=torch.int32),
        positions=None,
    )
    _num_rows, lens = builder._topk_lens_host(cam)
    # ctx 96..100, 2996..3000; cr=1: min(topk, ctx // 1)
    assert lens.tolist() == [96, 97, 98, 99, 100, 2048, 2048, 2048, 2048, 2048]


def test_topk_lens_host_ratio2():
    builder = _make_builder(
        DeepseekV4FlashInferSM90MetadataBuilder,
        _index_topk=8,
        _async_scheduling=False,
        compress_ratio=2,
    )
    cam = SimpleNamespace(
        num_reqs=1,
        query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([40], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([40], dtype=torch.int32),
        positions=None,
    )
    # cr=2: candidates == ctx // 2 -> 19, 20, 20, all clamped to topk=8.
    _num_rows, lens = builder._topk_lens_host(cam)
    assert lens.tolist() == [8, 8, 8]


def test_backend_gates():
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
