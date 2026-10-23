# 310P W8A8 MoE 长跑退化：定位与修复

## 现象

Qwen3-30B-A3B-w8a8 在 310P 上长时间提供服务后，输出逐步退化：

1. 固定 canary（同一 6.2k prompt，要求只输出 `OK`）从 `OK` 变成 `!OK`；
2. 继续跑，个别请求在正常句子后插入 `!`、`�!`、`1_1_1`，直至整段乱码；
3. 坏结果一旦写进 prefix cache，后续命中同一前缀的请求持续退化；
4. 清 prefix cache 可暂时止损，重启进程恢复。

## 复现

自造负载（不使用任何业务数据）：6.2k 共享 system 前缀 + 长/短请求交替，每 20 条打一次固定 canary，同时记录 cache 统计和 NPU 显存。

- 复现脚本：板卡 `/data2/fe_probe/repro_cache.py`（`stress` / `sweep` / `grow` / `consistency`）；
- 服务级脚本：`server_stress.py` + `run_server_stress.sh`；
- 关键判据不是「某条输出看起来怪」，而是**固定 canary 的逐 token 漂移**：`OK` → `!OK`（首个 token 变成 0）→ 整段乱码。

## 根因

### 1. prefill 与 decode 共用按 token 数索引的 attention 输出缓冲

`_get_decode_out` 原来只用 `(bs, heads, dim, dtype, device)` 做 key，prefill 和 decode 共用同一字典。310P 的 decode 图按 batch 桶捕获，attention 输出缓冲的**地址被烧进 NPUGraph**；prefill 的 `bs` 只要等于某个 decode 桶（1/2/4/8/...），就会复用同一张缓冲。长跑中 token 形状不断变化，叠加按字节上限的 LRU 淘汰/重建，图回放会读到已被覆盖或复用的显存，导致该步 KV/logits 污染。

污染一旦发生在完整 prompt 块上，`hash_blocks` 会把它登记进 prefix cache；后续请求命中同一前缀 → 持续退化，直到清缓存/重启。这解释了「canary 先多一个 `!`、再整段乱码」以及「prefix cache 命中越高的长 prompt 越容易中招」。

### 2. 有界缓存的淘汰与图捕获/异步算子的生命周期冲突

`_nz_mask` / `_nd_mask` / `_dflash_mask` / `_rope_apply_cache` 都按字节 LRU 淘汰。被淘汰的张量可能仍被：

- NPUGraph 捕获（地址已固化），或
- 尚未完成的异步 NPU 算子

引用。直接丢引用会让显存被下一次分配复用。实测把 `_decode_out` 也改成这种 LRU 后，300 条内必现 `!OK`；即使淘汰前 `synchronize()`，图捕获地址失效仍会复现。

## 修复

### `flashengine/backends/ascend.py`

1. **输出缓冲分池**（核心修复）
   - prefill：所有层共用一张 `max_num_batched_tokens` 大小的缓冲，按 `bs` 切片；不淘汰、不按 token 数累积；
   - decode：每个图桶/并发数一张固定缓冲，只增不减；桶数由 `max_num_seqs` 决定，天然有界；
   - 两者物理分离，prefill 不再覆盖 decode 图捕获的 `out` 地址。
2. **缓存淘汰安全化**
   - `_lru_put` 支持 `pinned` 集合：图捕获/回放用到的 key 永久钉住；
   - 淘汰时先把张量保活到 `torch.npu.synchronize()`，再释放；
   - 捕获期不做淘汰/同步；
   - mask/RoPE 在图上下文（`use_cuda_graph`）下访问的 key 自动 pin。
3. **prefix hash 防御**（`engine/block_manager.py`）
   - 命中要求 `block.hash == lookup_hash`，拒绝 stale 映射；
   - 父块 hash 为 -1 时不把子块登记成链首；
   - 重写块前删除旧 hash 映射。
### 量化侧（已在 WIP 中）

- 检查点 `deq_scale` 是 float32；310P 的 `npu_quant_matmul` 直接吃 float32 scale 会报 `aclnnQuantMatmulV5` 161001；
- 必须经 `npu_trans_quant_param` 生成 int64 `aclnn_deq_scale`，且 `out_dtype` 与后端 fp16 一致。

## 验证记录（2026-10-23，板卡 192.168.16.118）

| 用例 | 配置 | 结果 |
|---|---|---|
| 修复前（共用输出缓冲 + LRU） | 300 条混合，6.2k 前缀 | canary #239 `!OK`，7/300 bad |
| 引擎级长跑 | 1000 条混合，6.2k 前缀，seqs=1，chunk=1024 | **0/1000 bad**，canary 全 `OK`，graph replay 15001、fallback 0，NPU 32.56G |
| 长上下文/掩码淘汰 | 2k–18k 共 12 个长度 × 2 遍 × 3 轮 | **0 bad**，canary 全 `OK`，触发 mask 淘汰 |
| HTTP 服务级 | 300 条混合，OpenAI server，seqs=1 | **0/300 bad**，canary 全 `OK` |
| host 单测 | `python -m unittest discover -s tests -v` | 50 tests OK（2 skipped） |
| benchmark | 30B W8A8，3 题，64 token | 21.19 tok/s，TTFT 404ms，TPOT 41.5ms |

## 复现命令

```bash
# 引擎级 1000 条
docker exec -d vllm018 bash /data2/fe_probe/start_stress.sh stress 1000 1024 1
# 长上下文/掩码淘汰
docker exec -d vllm018 bash /data2/fe_probe/start_stress.sh sweep 1 1024 1
# HTTP 服务级 300 条
docker exec -d vllm018 bash /data2/fe_probe/run_server_stress.sh 300
```

## 剩余边界

- 本次为 seqs=1、无投机、W8A8 MoE；高并发、DFlash/DSpark、CoreX 仍需分别验证。
- `_lru_put` 淘汰时的全局 `synchronize()` 是低频路径，但会引入毫秒级停顿；若长跑中淘汰频繁，可进一步改成固定池/异步回收。
- 小时级稳定性、故障注入、驱动/固件版本差异未覆盖。
