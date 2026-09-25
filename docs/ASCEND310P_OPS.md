# Ascend 310P 算子清单

本文列出 nano-vLLM 在 Ascend 310P（Atlas 300I Duo / 200I Pro）上实际使用的推理算子，
并给出与一份历史算子清单、以及 vLLM-Ascend 310P 官方实现的逐项对照。

- 本项目路径：`nanovllm/backends/ascend.py` 及调用它的通用层（`nanovllm/layers/*.py`）。
- vLLM-Ascend 路径：`vllm_ascend/_310p/`（`attention/attention_v1.py`、`ops/layernorm.py`、
  `ops/activation.py`、`ops/rotary_embedding.py`、`ops/linear.py`、
  `ops/vocab_parallel_embedding.py`、`device/device_op.py`、`sample/sampler.py`）。

| 算子作用 | 本项目（nano-vLLM 310P） | 历史清单 | vLLM-Ascend 310P 官方 |
|---|---|---|---|
| Token Embedding | `F.embedding` | `F.embedding` | `F.embedding` |
| QKV 投影 | `F.linear`（NZ 权重） | `F.linear` | `F.linear`（NZ 权重，custom op `unquantized_gemm`） |
| Q Norm | `npu_rms_norm` | `npu_rms_norm` | `npu_rms_norm` |
| K Norm | `npu_rms_norm` | `npu_rms_norm` | `npu_rms_norm` |
| RoPE | `npu_apply_rotary_pos_emb`（rotary_mode=half，回退 `_npu_rotary_embedding`） | `npu_apply_rotary_pos_emb` | `npu_apply_rotary_pos_emb`（回退 `_npu_rotary_embedding`） |
| KV Cache 写入 | `_npu_reshape_and_cache` | `_npu_reshape_and_cache` | `_npu_reshape_and_cache` |
| Prefill Attention | `_npu_flash_attention_v3`（回退 `_npu_flash_attention`） | （未列） | `_npu_flash_attention_v3`（回退 `_npu_flash_attention`） |
| Decode Attention | `_npu_paged_attention` | `_npu_paged_attention` | `_npu_paged_attention` |
| Chunked Prefill / Spec 校验 | `_npu_paged_attention`（spec 校验，host 预展开）/ `_npu_paged_attention_splitfuse_v2`（chunked prefill） | （未列） | `_npu_paged_attention_splitfuse_v2` |
| O 投影 | `F.linear`（NZ 权重） | `F.linear` | `F.linear`（NZ 权重） |
| 残差融合归一化 | `npu_add_rms_norm` | `npu_add_rms_norm` | `npu_add_rms_norm` |
| Gate+Up 投影 | `F.linear`（NZ 权重） | `F.linear` | `F.linear`（NZ 权重） |
| SwiGLU | `npu_swiglu`（回退 `F.silu`） | `npu_swiglu` | `npu_swiglu`（回退 `F.silu`） |
| Down 投影 | `F.linear`（NZ 权重） | `F.linear` | `F.linear`（NZ 权重） |
| Final RMSNorm | `npu_add_rms_norm` | `npu_add_rms_norm` | `npu_add_rms_norm` |
| LM Head | `F.linear`（`weight_nz`） | `F.linear` | `F.linear`（`weight_nz`，custom op `unquantized_gemm`） |
| Logits 回传 | `logits.float()` 后 `.tolist()` | `.detach().float().cpu()` | 采样在 NPU 上，token 后 `.tolist()` |
| 采样（greedy） | `argmax` | `argmax` | `argmax` |
| 采样（top-k / top-p） | `argsort` / `scatter_` / `masked_fill_` | `topk` / `sort` | `sort` / `cumsum` / `masked_fill`（310P 无 `npu_top_k_top_p` 能力） |
| 采样（随机） | `exponential_` + `argmax`（Gumbel） | `multinomial` | CPU uniform + `cumsum` / `searchsorted`（逆 CDF） |
