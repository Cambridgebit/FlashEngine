# Qwen3.5 普通与 DFlash 前缀缓存

Qwen3.5 dense/MoE 文本模型默认开启 prefix cache，普通推理和 DFlash 使用同一机制。
匹配依据仍是聊天模板编码后、从输入开头连续相同的 token 完整块。
固定提示词放在 system 或 user 都可命中；输入中间相同的片段不能单独复用。
本轮 CUDA 实机为 Qwen3.5-2B，MoE/CoreX/310P 走共享实现，新增缓存尚未实机验收。

## 同时复用三种状态

1. **目标 full_attention KV**：沿用 BlockManager 的物理块、前缀链哈希和 token 校验。
2. **目标 GDN 状态**：每个完整块边界保存所有 GDN 层的 raw conv 历史与 fp32 SSM 状态。
   命中后只恢复末块终态，继续计算剩余输入，不能使用请求完成时的终态代替块边界终态。
3. **DFlash 草稿 KV**：稠密路径恢复原有 CPU 快照；分页路径沿用与目标共索引的草稿 KV。
   分块 prefill 每步只保存已经投影的完整块，不保存尚未计算的草稿上下文。

GDN prefill 在所需完整块边界分段递推，并截取对应的卷积 raw 输入窗口。
无需保存逐 token 的巨大状态轨迹，也无需将整个目标模型按块反复执行。
全部 GDN 层完成后才发布快照；失败前向的半成品不参与后续命中。
CUDA 使用锁页 CPU 快照和异步拷贝，在发布前统一同步一次。
CoreX/310P 保守使用同步 CPU 拷贝，后端差异由能力位控制。

`BlockManager.prefix_state_validator` 只校验命中末块状态。
中间 GDN 快照被淘汰，但末块状态仍在时，仍可复用全部连续 KV 前缀。
末块状态缺失时向前寻找可恢复边界；草稿 KV 缺失则缩短到连续有效前缀。
哈希与物理块号必须同时匹配，物理块重用不会误读旧快照。
最后一个 prompt 块仍须重新计算；只有一整块的输入不能全跳过。
分页草稿继续排除可能被非因果查询覆盖的末块哈希。

## 生命周期与内存

- 主状态槽随请求完成、抢占、取消回收。
- 前缀快照独立保留，以 LRU 淘汰，直到预算不足、同物理块更新或引擎退出。
- 新请求命中时恢复一次；分块 prefill 后续步继续使用存活主槽。
- GPU decode/spec Graph 只使用既有主槽与验证行槽，快照操作位于图外。
- GDN 已发布快照默认预算 256 MiB。冷 prefill 暂存另有不超过同预算的空间，
  因而此参数不是进程总 RAM 或锁页内存的硬上限。CUDA 两部分都可能占锁页内存。
- Qwen3.5-2B bf16 每个 GDN 边界快照约 18.63 MiB，256 MiB 可保留 13 个。
  DFlash 既有草稿 CPU 快照预算 1 GiB，与 GDN 预算独立。
- 状态不会落盘，重启服务后缓存为空。预算不足以放一个 GDN 状态时干净关闭 prefix cache。

配置字段 `recurrent_prefix_cache_bytes`；server 与 benchmark 的 CLI 为
`--recurrent-prefix-cache-mb 256`。传 0 或 `--no-prefix-cache` 全链路关闭复用。
启用缓存会增加冷请求的快照保存开销，收益主要是重复长前缀的 TTFT。

```bash
cd /mnt/d/Materials/AVIC/FlashEngine
source /home/yjq040828/vllm-env/bin/activate
python3 -m scripts.openai_server /home/yjq040828/vllm-lab/models/Qwen3.5-2B \
  --spec-model /home/yjq040828/vllm-lab/models/Qwen3.5-2B-Dflash \
  --backend cuda --host 127.0.0.1 --max-model-len 1024 \
  --max-num-seqs 1 --no-think --recurrent-prefix-cache-mb 256
```

无需额外的开启开关。去掉 `--spec-model` 即普通推理。
CUDA FlashAttention 默认 256-token 块；FlashInfer 可使用更小的页。
CoreX/310P 仍受各自 prefix attention/SplitFuse 能力门约束。

## 观察与复现

`runtime_snapshot()["recurrent"]["prefix_cache"]` 提供
`saved / restored / evicted / bytes / limit_bytes / entries`。
server `/metrics` 和 `/v1/stats` 暴露 `flashengine_recurrent_prefix_*` 指标；
benchmark 打印 `Recurrent prefix snapshots`，原有 KV 与草稿命中统计保留。
`reset_measurement_stats()` 仅重置计数，不清空快照。

```bash
# 小状态主机契约，不加载模型
python3 -m scripts.check_qwen35_prefix --host-only
# eager/Graph × 普通/DFlash；192-token chunk 跨越 256-token 边界
python3 -m scripts.check_qwen35_prefix /home/yjq040828/vllm-lab/models/Qwen3.5-2B \
  --draft /home/yjq040828/vllm-lab/models/Qwen3.5-2B-Dflash --preempt
# 默认草稿 k=15 与整段 prefill，小显存卡使用单并发
python3 -m scripts.check_qwen35_prefix /home/yjq040828/vllm-lab/models/Qwen3.5-2B \
  --draft /home/yjq040828/vllm-lab/models/Qwen3.5-2B-Dflash \
  --cases spec_graph --seqs 1 --spec-num-tokens 15 --chunk-size 1024 \
  --output bench_results/qwen35-check/prefix-k15
# FlashInfer 使用已安装 toolkit/ninja 的环境
CUDA_HOME=/usr/local/cuda python3 -m scripts.check_qwen35_prefix \
  /home/yjq040828/vllm-lab/models/Qwen3.5-2B \
  --draft /home/yjq040828/vllm-lab/models/Qwen3.5-2B-Dflash \
  --cases fi_graph,fi_spec --output bench_results/qwen35-check/prefix-fi
# 临时启动真实 HTTP server，验证非流式/流式之间的缓存复用并自动关闭
python3 -m scripts.check_qwen35_prefix /home/yjq040828/vllm-lab/models/Qwen3.5-2B \
  --draft /home/yjq040828/vllm-lab/models/Qwen3.5-2B-Dflash --http \
  --output bench_results/qwen35-check/prefix-http
```

验收默认关闭缓存与开启缓存分别运行，严格比较 greedy token，不豁免 BF16 分歧。
覆盖 system/user 固定前缀、不同后缀、重复请求、短 prompt、整块 prompt、分块边界、
三请求并发、末块状态独立淘汰、草稿状态缺失、取消与可选真实抢占。
`--reuse-off` 仅用于实现不影响关闭缓存路径时复用已有同配置基线。
工件保存于 `bench_results/qwen35-check/`，该目录已忽略。

## 本轮验收记录

2026-10-15，WSL Ubuntu-22.04 / RTX 4070 Laptop 8GB / torch 2.11 / transformers 5.15，
Qwen3.5-2B + 匹配 DFlash，bf16/greedy/no-think。

- FlashAttention 的普通 eager/Graph、DFlash eager/Graph，以及 FlashInfer 的普通
  Graph/DFlash eager：六条路径各串行 10/10、并发 3/3 与关闭缓存基线严格一致。
- system/user 两种固定前缀均命中 2 个 256-token 块。
  system 请求 661 token 只重算 149 token，user 请求 659 token 只重算 147 token。
- 192-token chunk 覆盖非块对齐的调度边界；一状态预算下的 LRU、草稿 KV 缺失、
  短输入/整块输入和取消均通过。普通/投机真实抢占分别 6/5 次，完成后主槽为 0。
- DFlash k=15、单并发、整段 prefill：串行 10/10、批量入口 1/1 冷热严格一致。
- 真实 HTTP server：先 `stream=false` 再 `stream=true`，文本与 usage 严格一致，
  第二次命中 2 块，`recurrent_prefix_restored_total` 增加 1，结束后 active=0。
  临时服务已关闭。桌面占用导致 util=0.85 无 KV 预算，测试用 util=0.95、prefill 预算 192。
- Python 语法检查与小状态主机契约通过。CoreX/310P 新增缓存尚未实机运行。

代表请求 TTFT（ms，开启缓存进程内的第一次与第三次请求；不是固定频率稳态基准）：

| 路径 | 冷前缀 | 热前缀 |
|---|---:|---:|
| FlashAttention 普通 eager | 399.3 | 76.4 |
| FlashAttention 普通 Graph | 487.9 | 56.7 |
| FlashAttention DFlash eager，k=2 | 313.1 | 71.1 |
| FlashAttention DFlash Graph，k=2 | 359.6 | 64.7 |
| FlashInfer 普通 Graph | 1432.4 | 247.0 |
| FlashInfer DFlash eager，k=2 | 1452.0 | 263.4 |
| FlashAttention DFlash Graph，k=15、整段 prefill | 275.6 | 82.3 |

这些数字包含不同路径的冷启动、快照开销及当时硬件状态，不用于跨库性能排名。
首次缓存前缀有额外保存开销；完整关闭缓存的基线和全部 token 已保存在同目录工件。
常规四路径工件在 `prefix/`，FI 在 `prefix-fi/`，默认草稿宽度在 `prefix-k15/`，
HTTP 工件在 `prefix-http/`；其上层为 `bench_results/qwen35-check/`。

短输出严格门不代表所有长输出在不同 attention/batch 形状下逐位一致；
原有 BF16 数值边界仍见 `QWEN35.md` 和 `DFLASH_PERFORMANCE.md`。
