# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 FlashInfer SM90 (Hopper) sparse MLA backend.

Opt-in backend (``FLASHINFER_MLA_SPARSE_DSV41_SM90``) serving DSV4.1 on SM90
through FlashInfer's ``BatchMLAPagedAttentionWrapper`` (FA2/FA3 paths, FP8 KV
in-kernel dequantization), aligned with the SM100/SM120 FlashInfer paths.

DSV4.1 differs from the generic SM90 NoPE backend
(``flashinfer_mla_sparse_sm90.py``) in two ways, resolved by a two-call design:

1. Double cache: every layer reads the SWA cache (sliding window) plus, for
   compress_ratio > 0 layers, the compressed cache (indexer top-k). The SM90
   wrapper supports a single KV cache per call, so each layer runs two
   ``page_size=1`` varlen calls — call A over the SWA rows (``page_size=1``
   per-token page table, causality already encoded by the SWA index kernel)
   and call B over the compressed rows mapped by
   ``compute_global_topk_indices_and_lens`` — and merges the partial outputs
   with ``merge_attn_states`` (LSE rescaling), mathematically equivalent to
   FlashMLA's single fused call.
2. The layer is driven by ``DeepseekV4Attention.forward_mqa``, not a generic
   ``MLAAttentionImpl``, so the builders read their parameters (head count,
   scale, widths) from the ``DeepseekV4FlashInferSM90Attention`` instance in
   the static forward context instead of an impl object.

The wrapper bakes per-row KV lengths into its host-side schedule at plan()
time, so both builders replan every step outside CUDA graph capture with
exact host-side lengths (see ``_SM90State.plan`` for why inexact lengths
illegal-address). Lengths are derived from the same formulas as the device
index kernels: SWA rows are ``min(ctx, window)`` (causal) or
``min(seq_len, window + q_len)`` (DSpark non-causal draft rows); compressed
rows are ``min(index_topk, ctx // compress_ratio)``.

Sink handling: attention sinks are applied after the merge as an exact
post-correction (``scale = sigmoid(lse - sink)`` per head), whether the
wrapper natively supports sinks or not; the planned calls always run without
sinks so the returned LSE excludes the sink term.

CUDA-graph: both builders own ``_SM90State`` wrappers with reserved
capture-stable buffers and plan in ``build()`` (always eager); captured
forward runs refresh only the index contents.
"""

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.models.deepseek_v4_1.sparse_mla import (
    DeepseekV41SparseSWAMetadataBuilder,
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import has_flashinfer_sm90_nope_mla
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import _SM90State
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import KVCacheLayout

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def has_flashinfer_sm90_mla_lse() -> bool:
    """Whether the SM90 MLA wrapper exposes per-row LSE from ``run``.

    The two-call merge requires ``return_lse``; without it the backend cannot
    combine the SWA and compressed partials.
    """
    if not has_flashinfer_sm90_nope_mla():
        return False
    try:
        import inspect

        from flashinfer.mla import BatchMLAPagedAttentionWrapper
    except ImportError:
        return False
    try:
        params = inspect.signature(BatchMLAPagedAttentionWrapper.run).parameters
    except (TypeError, ValueError):
        return False
    return "return_lse" in params


def _decode_threshold(vllm_config: VllmConfig) -> int:
    """Decode/query-length split threshold, mirroring the SWA builder."""
    spec_config = vllm_config.speculative_config
    num_spec = spec_config.num_speculative_tokens if spec_config else 0
    spec_mult = 2 if (spec_config is not None and spec_config.parallel_drafting) else 1
    return 1 + spec_mult * num_spec


def _host_rows(
    cam: CommonAttentionMetadata,
    async_scheduling: bool,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Per-row host tensors for the scheduled batch: ``(num_rows, positions,
    seq_lens, q_lens, req_of_row)`` (int64 CPU).

    ``positions`` is exact: derived from the maintained host seq-lens upper
    bound when scheduling is synchronous, otherwise from the device positions
    at the cost of one D2H sync per metadata build.
    """
    num_reqs = cam.num_reqs
    qsl = cam.query_start_loc_cpu[: num_reqs + 1]
    num_rows = int(qsl[-1])
    if num_rows == 0:
        return None
    q_lens = (qsl[1:] - qsl[:-1]).to(torch.int64)
    req_of_row = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int64), q_lens
    )
    # Token offset within the request's tokens (rows are decode-first).
    row_offset = torch.arange(num_rows, dtype=torch.int64) - qsl.to(torch.int64)[
        req_of_row
    ]
    if not async_scheduling and cam.seq_lens_cpu_upper_bound is not None:
        seq_lens = cam.seq_lens_cpu_upper_bound[:num_reqs].to(torch.int64)
        positions = (seq_lens - q_lens)[req_of_row] + row_offset
    elif cam.positions is not None and num_rows <= cam.positions.shape[0]:
        positions = cam.positions[:num_rows].cpu().to(torch.int64)
        seq_lens = cam.seq_lens[:num_reqs].cpu().to(torch.int64)
    else:
        seq_lens = cam.seq_lens[:num_reqs].cpu().to(torch.int64)
        positions = (seq_lens - q_lens)[req_of_row] + row_offset
    return num_rows, positions, seq_lens, q_lens, req_of_row


class DeepseekV4FlashInferSM90SparseBackend(DeepseekV4SparseMLABackend):
    """FlashInfer SM90 backend using the DSv4.1 sparse metadata/cache layout.

    The attention itself runs through ``DeepseekV4FlashInferSM90Attention``;
    this class only declares capability, cache layout, and the metadata
    builder that replans the compressed-cache wrapper every step.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_DSV41_SM90"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # NoPE kernel mode: the whole 512-wide row (448 latent + 64 rope) is
        # the ckv dim, kpe = 0.
        return [512]

    @classmethod
    def supports_sink(cls) -> bool:
        # Applied as an exact LSE post-correction after the merge.
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if device_capability.major != 9:
            return "FLASHINFER_MLA_SPARSE_DSV41_SM90 requires SM90"
        if not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_DSV41_SM90 requires FlashInfer with "
                "SM90 MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )
        if not has_flashinfer_sm90_mla_lse():
            return (
                "FLASHINFER_MLA_SPARSE_DSV41_SM90 requires "
                "BatchMLAPagedAttentionWrapper.run(return_lse=True) for the "
                "two-call LSE merge; upgrade FlashInfer"
            )
        if not use_sparse:
            return "FLASHINFER_MLA_SPARSE_DSV41_SM90 requires sparse MLA"
        if kv_cache_dtype not in (None, "auto", "bfloat16", "fp8", "fp8_e4m3"):
            # fp8_ds_mla is the FlashMLA-only UE8M0 packed layout.
            return (
                "FLASHINFER_MLA_SPARSE_DSV41_SM90 uses plain per-tensor "
                "FP8/bf16 KV, not fp8_ds_mla"
            )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf = vllm_config.model_config.hf_text_config
            if hf.kv_lora_rank != 512:
                return "FLASHINFER_MLA_SPARSE_DSV41_SM90 requires kv_lora_rank=512"
            if hf.qk_rope_head_dim not in (0, 64):
                return (
                    "FLASHINFER_MLA_SPARSE_DSV41_SM90 requires qk_rope_head_dim "
                    "in (0, 64)"
                )
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The DSV4.1 pool packs the indexer pages beside the MLA latent pages
        # inside each block, so the layer dim must sit inside the block dim
        # (same requirement as DeepseekV4IndexerBackend); the layout is
        # resolved worker-globally as the intersection of all backends.
        return (KVCacheLayout.BLHNC, KVCacheLayout.BLNHC)

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4FlashInferSM90MetadataBuilder"]:
        return DeepseekV4FlashInferSM90MetadataBuilder


class DeepseekV4FlashInferSM90MetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    """Metadata builder for the compressed-cache (top-k) wrapper call.

    Owns the compressed-cache ``_SM90State`` and replans it every step with
    exact host-side per-row lengths: a row for the j-th query token of request
    i has ``ctx = position + 1`` and the indexer's valid top-k entry count is
    ``min(index_topk, ctx // compress_ratio)`` (compressed-row granularity; no
    pool expansion, matching ``_fill_short_context_topk_indices``).
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: Any,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
            DeepseekV4FlashInferSM90Attention,
        )

        attention_layer = vllm_config.compilation_config.static_forward_context[
            layer_names[0]
        ]
        if not isinstance(attention_layer, DeepseekV4FlashInferSM90Attention):
            raise TypeError(
                "DeepseekV4FlashInferSM90MetadataBuilder requires a "
                "DeepseekV4FlashInferSM90Attention layer, got "
                f"{type(attention_layer).__name__}."
            )
        hf_config = vllm_config.model_config.hf_text_config
        assert hf_config.index_topk is not None
        self._index_topk = int(hf_config.index_topk)
        self._async_scheduling = bool(vllm_config.scheduler_config.async_scheduling)
        self._decode_threshold = _decode_threshold(vllm_config)
        self.state = _SM90State(
            device,
            attention_layer.n_local_heads,
            kv_cache_spec.dtype,
            vllm_config.scheduler_config.max_num_batched_tokens,
            self._index_topk,
            kv_lora_rank=int(hf_config.kv_lora_rank),
            qk_rope_head_dim=0,  # NoPE kernel mode: rope folded into ckv
            sm_scale=attention_layer.scale,
        )

    def _topk_lens_host(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> tuple[int, torch.Tensor]:
        rows = _host_rows(common_attn_metadata, self._async_scheduling)
        if rows is None:
            return 0, torch.zeros(0, dtype=torch.int32)
        num_rows, positions, _seq_lens, _q_lens, _req_of_row = rows
        # Valid compressed-row entries per token; padded rows have ctx 0.
        ctx = positions + 1
        lens = (torch.clamp(ctx, min=0) // self.compress_ratio).clamp_(
            max=self._index_topk
        )
        return num_rows, lens.to(torch.int32)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4FlashMLAMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        # Replan every step outside any CUDA graph capture with this step's
        # exact per-row lengths; captured runs read the refreshed buffers.
        num_rows, topk_lens = self._topk_lens_host(common_attn_metadata)
        self.state.plan(num_rows, topk_lens)
        metadata.flashinfer_sm90_topk_state = self.state
        return metadata


class DeepseekSparseSWAFlashInferSM90Backend(DeepseekSparseSWABackend):
    """SWA backend for the FlashInfer SM90 DSV4.1 attention class.

    Same kernels as the default SWA path; the builder additionally owns the
    SM90 wrapper state for the sliding-window call, replanned every step.
    """

    @staticmethod
    def get_builder_cls() -> type["DeepseekSparseSWAFlashInferSM90MetadataBuilder"]:
        return DeepseekSparseSWAFlashInferSM90MetadataBuilder


class DeepseekSparseSWAFlashInferSM90MetadataBuilder(
    DeepseekV41SparseSWAMetadataBuilder
):
    """SWA metadata builder that also drives the SM90 sliding-window wrapper.

    SWA-only layers receive no compressed-cache metadata, so the SWA call's
    wrapper state must be carried on the SWA metadata itself.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._async_scheduling = bool(
            self.vllm_config.scheduler_config.async_scheduling
        )
        hf_config = self.vllm_config.model_config.hf_text_config
        self._max_image_tokens = (
            getattr(hf_config, "vision_max_n_token", 0)
            if getattr(hf_config, "vision_n_layers", 0) > 0
            else 0
        )
        if self._max_image_tokens > 0:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_DSV41_SM90 does not support the vision "
                "variant (in-image bidirectional SWA visibility)."
            )
        self._is_dspark = self.is_dspark
        # Wrapper rows use one fixed stride: the widest index width (the
        # non-causal DSpark decode width when present, else the window).
        width = self.window_size
        if self._is_dspark:
            width = max(width, self.noncausal_index_width)
        num_heads = self.vllm_config.model_config.get_num_attention_heads(
            self.vllm_config.parallel_config
        )
        head_size = int(self.kv_cache_spec.head_size)
        self.state = _SM90State(
            self.device,
            num_heads,
            self.kv_cache_spec.dtype,
            self.max_num_batched_tokens,
            width,
            kv_lora_rank=head_size,
            qk_rope_head_dim=0,  # NoPE kernel mode
            sm_scale=head_size**-0.5,
        )

    def _swa_lens_host(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> tuple[int, torch.Tensor]:
        """Exact per-row SWA lengths, mirroring the device index kernels:
        causal rows see ``min(ctx, window)``; DSpark non-causal decode rows
        (block-anchored) see ``min(seq_len, window + q_len)``."""
        rows = _host_rows(common_attn_metadata, self._async_scheduling)
        if rows is None:
            return 0, torch.zeros(0, dtype=torch.int32)
        num_rows, positions, seq_lens, q_lens, req_of_row = rows
        non_causal = not common_attn_metadata.causal
        if not non_causal:
            lens = torch.clamp(positions + 1, min=0).clamp(max=self.window_size)
            return num_rows, lens.to(torch.int32)
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )
        lens = torch.clamp(positions + 1, min=0).clamp(max=self.window_size)
        if num_decode_tokens > 0:
            per_req = torch.minimum(seq_lens, self.window_size + q_lens)
            lens[:num_decode_tokens] = per_req[req_of_row[:num_decode_tokens]]
        return num_rows, lens.to(torch.int32)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> "DeepseekSparseSWAMetadata":
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_rows, swa_lens = self._swa_lens_host(common_attn_metadata)
        self.state.plan(num_rows, swa_lens)
        metadata.flashinfer_sm90_swa_state = self.state
        return metadata
