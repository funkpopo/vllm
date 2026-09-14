# TODO: 将 FlashInferMLASparseSM90Backend 接入 DeepSeek-V4.1 后端选择逻辑

> 目标：让 DSV4.1 在 SM90 (Hopper) 上可选 FlashInfer 稀疏 MLA 后端（FA2/FA3 路径，
> FP8 KV in-kernel 反量化），与 SM100/SM120 的 FlashInfer TRTLLM 路径对齐。
> 默认后端保持 FlashMLA 不变，新后端通过显式 `--attention-backend` 选择（opt-in）。

## 6. 实施进度（本机无 GPU，已完成的部分已暂存，待 GPU 服务器验证）

- [x] 2.2 backend + 双 builder（MLA builder 持 topk `_SM90State`；SWA builder 持 swa `_SM90State`，
      复用 `flashinfer_mla_sparse_sm90.py::_SM90State`，`build()` 时 capture 外 replan，
      host 侧行长度公式：SWA = `min(ctx, window)`（因果）/ `min(seq_len, window+q_len)`（DSpark 非因果），
      topk = `min(index_topk, ctx // compress_ratio)`）— 新文件
      `vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse_sm90.py`
- [x] 2.3 枚举注册（`FLASHINFER_MLA_SPARSE_DSV41_SM90`）+ `_select_dsv4_attn_cls` 接线
      （通用 SM90 枚举对 V4.1 报错；默认值未改，SM90 默认仍 FlashMLA）
- [x] 2.1 attention 类 `DeepseekV4FlashInferSM90Attention`（decode + prefill 统一走双调用 +
      `merge_attn_states` LSE 合并；sink 用 `sigmoid(lse - sink)` 后处理精确修正；
      NoPE kernel 模式：整 512 行作 ckv、kpe=0）— 见 `flashinfer_sparse.py`
- [x] KV dtype 规范化：attention 基类新增 `_canonicalize_kv_cache_dtype` 钩子，
      SM90 类把 `auto` → `fp8`（plain per-tensor E4M3），`fp8` 保持不变（不升 fp8_ds_mla）
- [x] CPU 单元测试 `tests/v1/attention/test_dsv41_flashinfer_sm90.py`（双调用接线/单调用路径/
      NoPE 形状/长度公式/后端门控/枚举路径/选择逻辑）— 需在装好 vllm 的环境执行
- [ ] 第 1 节其余前置检查与第 3/4 节所有 GPU 项（见下）

## 0. 背景与现状

- 现有选择逻辑：`vllm/models/deepseek_v4_1/nvidia/model.py::_select_dsv4_attn_cls`
  - SM12x 默认 → `DeepseekV4FlashInferSM120Attention`（FlashInfer SM120）
  - 其余 CUDA 架构（SM90/SM100）默认 → `DeepseekV4FlashMLAAttention`
  - `FLASHINFER_MLA_SPARSE_DSV41` 仅支持 `major in [10, 12]`
- 可复用但未接线的通用 SM90 后端：
  `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py`
  （为 GLM-5.3-Flash NoPE MLA 编写，wrapper = FlashInfer `BatchMLAPagedAttentionWrapper`）
- DSV4.1 与通用后端的两个关键差异（设计必须解决）：
  1. **双 cache**：每层同时读 SWA cache（滑窗）+ 压缩 cache（indexer topk）；
     FlashMLA 用 `extra_k_cache` 一次调用解决；通用 SM90 后端只支持单 cache。
  2. **impl 驱动方式不同**：DSV4.1 的 attention 由 `DeepseekV4Attention.forward_mqa`
     驱动，不走通用 backend 的 `MLAAttentionImpl`；
     `FlashInferMLASparseSM90Builder` 现在硬性要求 `impl` 是
     `FlashInferMLASparseSM90Impl`。

## 1. 可行性前置检查（不通过则调整方案/终止）

- [x] 确认 FlashInfer `BatchMLAPagedAttentionWrapper.run` 支持 `return_lse`
      （GPU 环境已验证：FlashInfer 0.6.18.post1，return_lse=True）。
- [x] 确认 wrapper 是否支持 attention sink（GPU 环境已验证：**不支持**）。已采用
      合并后的 LSE 做 sink 修正（`scale = sigmoid(lse - sink)`，纯后处理、数学精确），
      `supports_sink()` 已如实声明 True；实现见 `DeepseekV4FlashInferSM90Attention.
      _apply_sink_correction`（数值待 parity 测试验证）。
- [x] 确认 `has_flashinfer_sm90_nope_mla()`（FlashInfer >= 0.6.18，ckv_scale_arr）
      在目标环境为 True（GPU 环境已验证：True）。
- [ ] 确认 FP8 KV 走 per-tensor E4M3（非 fp8_ds_mla）时，v4.1 的压缩 cache 写入
      （compressor / `fused_compress_quant_cache`）与 SWA cache 插入均支持 plain
      E4M3 布局 —— SM100 FlashInfer 类（`use_fp8_ds_mla_layout=False`）已是先例，
      `DeepseekV4SWACache` 注释也声明支持 `float8_e4m3fn` 连续布局，逐路径验证即可。
- [ ] 确认 `has_flashinfer_sm90_nope_mla()`（FlashInfer >= 0.6.18，ckv_scale_arr）
      在目标环境为 True。
- [x] 确认 FlashInfer >= 0.6.18 可 import（GPU 环境已验证 0.6.18.post1；
      CUDA 13.0 工具链下 plan 待实际启动验证）。
- [ ] KV dtype 规范化：生产 CLI 传 `--kv-cache-dtype fp8`。FlashMLA 路径将其
      canonicalize 为 `fp8_ds_mla`；新后端必须把 `fp8`/`auto` canonicalize 为
      plain per-tensor E4M3，保证用户 CLI 无需改动（参照
      `test_sparse_mla_backends.py::_canonicalize_sparse_mla_kv_cache_dtype`）。
- [ ] SimpleCPUOffloadConnector 兼容性：该连接器按 block 做字节级拷贝
      （`v1/simple_kv_offload/copy_backend.py:121`），布局无关，但新后端
      plain E4M3 布局的 per-block 字节数与 fp8_ds_mla (uint8 packed) 不同，
      `cpu_bytes_to_use_per_rank` 的容量核算语义需重新验证；加一条
      kv_both + offload 的集成测试。
- [ ] 常规重复性检查（AGENTS.md）：
      `gh pr list --repo vllm-project/vllm --state open --search "SM90 sparse MLA"`

## 2. 设计方案

### 2.1 新增 attention 类（复制 SM100 类骨架，替换执行器）

`vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py` 新增：

```
class DeepseekV4FlashInferSM90Attention(DeepseekV4Attention):
    backend_cls = DeepseekV4FlashInferSM90SparseBackend   # 新 backend，见 2.2
    swa_backend_cls = DeepseekSparseSWAFlashInferSM90Backend  # 复用/别名
    use_fp8_ds_mla_layout = False   # per-tensor FP8 / bf16，同 SM100 类
```

`forward_mqa` 采用**双调用 + LSE 合并**：

- **调用 A（SWA 部分）**：wrapper 以 page_size=1 遍历 `swa_k_cache`，
  每行 kv_indices = 该 token 的 SWA 可见行（decode 行已是全局行号；
  prefill 用 `prefill_swa_indices`，同 FlashMLA 路径语义）。
- **调用 B（压缩 topk 部分）**：`swa_only` 层（compress_ratio==0）跳过；
  其余层用现成的 `compute_global_topk_indices_and_lens`
  （`deepseek_v4_1/common/ops/cache_utils.py`，FlashMLA decode 路径已在用）
  把 local topk 映射为压缩 cache 全局行 + 有效长度；prefill 行同法处理。
- **合并**：`vllm/v1/attention/ops/merge_attn_states.merge_attn_states`
  （LSE 重标定，数学上与 FlashMLA 单核内合并等价）。
  注意 LSE 布局为 `[NUM_HEADS, NUM_TOKENS]`，需按 wrapper 返回形状适配。
- 若 wrapper 无 sink 支持：在合并后用 LSE 做 sink 后处理修正（见 1 中的决策）。

### 2.2 新增 backend + metadata builder

新文件（建议）`vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse_sm90.py`：

- `DeepseekV4FlashInferSM90SparseBackend(DeepseekV4SparseMLABackend)`：
  - `get_name() == "FLASHINFER_MLA_SPARSE_DSV41_SM90"`
  - `supports_compute_capability`: `major == 9`
  - `supported_kv_cache_dtypes`: `["auto", "bfloat16", "fp8", "fp8_e4m3"]`
    （不支持 `fp8_ds_mla`，与通用 SM90 后端一致）
  - `get_supported_kernel_block_sizes`: `[MultipleOf(64)]`（page_size=1 语义，
    只约束底层 tensor block 对齐）
  - `get_supported_head_sizes()`: `[512]`
  - `supports_combination`: 复用 `has_flashinfer_sm90_nope_mla()` + kv_lora_rank=512
    + qk_rope_head_dim in (0, 64) 检查（参照通用类实现）
- `DeepseekV4FlashInferSM90MetadataBuilder(DeepseekV4SparseMLAMetadataBuilder)`：
  - 复用 v4.1 builder 链（`DeepseekV41SparseSWAMetadataBuilder` 的
    layer-type 分类、`DeepseekV4FlashMLAMetadata`、compress_ratio∈{1,2} 校验）
  - 持有两个 `_SM90State`（SWA / 压缩各一，宽度分别为 swa_width 与 index_topk），
    `build()` 时在 capture 外用**精确 host 侧行长度** replan
    （复用 `flashinfer_mla_sparse_sm90.py::_SM90State.plan` 的约束：
    长度不精确会越界读 -1 尾部 → illegal address）
  - 不要求 `FlashInferMLASparseSM90Impl`（v4.1 无 impl 类），改为从
    `static_forward_context` 中的 `DeepseekV4FlashInferSM90Attention` 读取
    num_heads / scale / topk 宽度等参数
  - `_cudagraph_support = ALWAYS`（plan 在 capture 外、每步 replan，与通用类相同）

### 2.3 接入选择逻辑

- `vllm/v1/attention/backends/registry.py`：新增枚举项
  `FLASHINFER_MLA_SPARSE_DSV41_SM90 = "vllm.models.deepseek_v4_1.nvidia.
  flashinfer_sparse_sm90.DeepseekV4FlashInferSM90SparseBackend"`
  （命名沿用 DSV4/DSV41 的“模型驱动后端单独命名”惯例，见现有注释）。
- `_select_dsv4_attn_cls`（`nvidia/model.py`）：
  - 把新枚举加入 FlashInfer 分支 → 返回 `DeepseekV4FlashInferSM90Attention`；
  - 对旧 `FLASHINFER_MLA_SPARSE`/`FLASHINFER_MLA_SPARSE_SM120`/`SM90` 通用枚举，
    若模型为 V4.1 则报错提示改用 DSV41 专用枚举（与现有处理一致）；
  - **不改默认值**：SM90 默认仍为 FlashMLA。
- 在 `attention_config` 校验处确认：backend 能力校验（capability major==9、
  sinks、kv dtype）在模型实例化前给出清晰报错。

## 3. 实施步骤（按依赖顺序）

- [ ] 2.2 backend/builder 骨架 + 枚举注册（可 import、可被选中）
- [ ] 2.1 attention 类：仅 decode（swa_only 层 + cr∈{1,2} 层）走通
- [ ] prefill 路径（prefill_swa_indices / chunk 语义对齐，注意 v4.1 prefill
      不需要 FlashMLA 的 4-token chunk，确认可整批处理）
- [ ] FP8 per-tensor cache 写路径验证（compressor + SWA cache insert）
- [ ] CUDA graph：确认 plan 不进 capture、buffer 地址稳定
      （复用 `_SM90State` 的 reserved-buffer 机制）
- [ ] **DSpark + FULL breakable cudagraph（生产配置，验收必过）**：
      生产启动参数为 DSpark(num_spec=5, probabilistic/block) +
      `cudagraph_mode FULL` + `VLLM_USE_BREAKABLE_CUDAGRAPH=1` +
      capture sizes [1,2,4,8]，因此：
  - DSpark draft 非因果宽索引（`decode_swa_width` = noncausal_index_width）
    在双调用 LSE 合并下语义必须正确（升级为验收标准，非普通回归项）；
  - builder `_cudagraph_support = ALWAYS` 的每步 capture 外 replan 需在
    breakable FULL capture 下验证（eager break 点与 plan 的交互）；
  - capture sizes 与 `--max-num-seqs` 对齐（生产 [1,2,4,8]）。
- [ ] DSpark（MTP draft）路径回归：draft 注意力同样经 `_select_dsv4_attn_cls`，
      确认非因果（non-causal）宽索引在双调用合并下语义正确
- [ ] DCP/PCP：确认与两调用方案兼容（通用 SM90 后端已支持 DCP 索引转换，
      但 v4.1 candidate_blocks 断言 `dcp_world_size == 1`，行为需对齐 FlashMLA 路径）
- [ ] **长上下文 prefill（生产 394K max-model-len）**：prefill 密集场景下
      双调用的 SWA 行（≤滑窗宽）+ topk 行（≤index_topk）都是常数代价，
      理论上优于 FlashMLA chunk 方案；用 128K/384K prompt 实测确认。

## 4. 测试计划

- [ ] 单元（新增 `tests/v1/attention/test_dsv41_flashinfer_sm90.py` 或并入
      `test_dspark_noncausal_sparse_mla.py` 风格）：
  - FlashInfer SM90 vs FlashMLA SM90 输出 allclose（bf16 / fp8 两种 KV dtype，
    compress_ratio ∈ {0,1,2} 各一组）
  - 双调用 LSE 合并与 FlashMLA 单调用数值一致性（重点：SWA 与 topk 重叠行）
  - sink 有/无两组
- [ ] 复用现有套件回归：`tests/v1/attention/test_sparse_mla_backends.py`、
      `test_dspark_noncausal_sparse_mla.py`（把新 backend 加入参数化矩阵，
      SM90 机器上运行）
- [ ] H100 冒烟：**按生产配置**（TP4+EP、DSpark num_spec=5、
      cudagraph FULL + breakable、enable-prefix-caching、
      SimpleCPUOffloadConnector kv_both、128K/394K 两档 max-model-len）
      各跑 `vllm bench` 一轮，记录吞吐/接受长度/精度，与 FlashMLA 默认后端对比，
      写入 PR 描述。另跑不带 DSpark 与不带 offload 的消融，
      定位收益来源（attention kernel vs offload 交互）。
- [ ] lint/CI：`pre-commit run --all-files`；相关 pytest 如上

## 5. 风险与备选

| 风险 | 缓解 |
|---|---|
| wrapper 不支持 LSE 输出 | 方案退化为 FlashMLA-only（放弃本项）或升级 FlashInfer |
| wrapper 不支持 sink | LSE 后处理精确修正（数学等价），或后端声明不支持 sink 并在校验层拒绝 |
| 双调用合并引入额外 kernel/带宽开销，收益不足 | 先做 microbenchmark（SWA 行数 vs topk 行数比例）再决定默认后端 |
| fp8 per-tensor 精度低于 fp8_ds_mla（UE8M0 块量化） | 用 parity 测试量化误差；必要时仅支持 bf16 KV 首发；注意生产已用 `--kv-cache-dtype fp8`，fp8 是默认诉求而非可选项 |
| SimpleCPUOffloadConnector / engram cpu_offload 与新布局的交互 | 集成测试覆盖 kv_both 模式下的 offload/load 往返 |
| Marlin MXFP4 MoE 是 SM90 上的另一瓶颈，attention 收益可能被掩盖 | 基准测试时分别报告 attention 后端差异与整体吞吐，避免误判 |
