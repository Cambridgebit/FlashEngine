# FlashAttention Graph 采样修复

## 原因

报错来自 `flashengine/layers/sampler.py`，不是 FlashAttention 注意力 kernel。
decode Graph 默认捕获 `模型 → lm_head → sampler`。原 sampler 在 CPU 生成指数噪声，
再 `.to(cuda)`。torch 2.11 拒绝在捕获期间把未锁页的 CPU tensor 复制到 CUDA：

```text
Cannot copy between CPU and CUDA tensors during CUDA graph capture
unless the CPU tensor is pinned.
```

greedy 请求也会执行这段代码：采样器先生成噪声，再屏蔽 greedy 行。
引擎按既有逻辑捕获失败后回退 eager，所以服务可以启动，但 decode 不再回放图。

只给 CPU tensor 加 `pin_memory=True` 不足以修好随机采样。
CPU RNG 不进入设备图，回放会重复读取捕获时的噪声，不能产生新的独立随机样本。
[PyTorch CUDA Graph 规则](https://docs.pytorch.org/docs/stable/notes/cuda#cuda-graphs)
允许 CUDA RNG，并明确 CPU 操作不会在 replay 中重新执行。

## 修复

- CUDA 后端声明 `supports_device_sampling_rng=True`，sampler 在 logits 所在设备生成噪声。
- Ascend 沿用保守的 CPU RNG 能力默认值；CoreX 在 Qwen3.5 扩展中也启用设备 RNG。
- fast 与 legacy 采样路径都使用同一 RNG 入口。CUDA Graph 每次回放会推进随机状态。
- 修复 fast 路径忽略 `top_p`：先过滤 top-k，再归一化计算 nucleus。
- greedy 行直接返回原 logits 的 argmax，不受随机噪声和候选排序影响。
- 修复 `FLASHENGINE_GRAPH_SAMPLE=0`：捕获时把 hidden states 写入固定 `outputs`，
  供回放后的 lm_head 和 sampler 使用。此前输出缓冲一直为零。

随机采样从 CPU RNG 改为 CUDA RNG，相同 seed 的随机文本可能与旧实现不同。
采样分布与 temperature/top-k/top-p 语义保留；不保证旧 CPU/GPU 随机流位等。

## 验证（2026-10-13）

WSL Ubuntu-22.04，RTX 4070 Laptop 8GB，torch 2.11.0，用户本地 Qwen3-0.6B。

| 检查 | 结果 |
|---|---|
| 完整 unittest | 37/37 通过，含 CUDA 实机 sampler Graph 测试 |
| fast / legacy Graph 连续回放 | 随机结果更新，没有冻结噪声 |
| Graph 外修改采样参数 | top-p、top-k、temperature 在下一次 replay 生效 |
| FlashAttention eager / Graph 内采样 / Graph 外采样 | 各 14 个贪心请求逐 token 全等 |
| batch=1/3/4、分块 prefill、前缀冷/热、随机采样 | 跑通；3→4 桶 padding 覆盖 |
| 原 benchmark | 三桶捕获成功，189 次回放，0 fallback |

原 benchmark 条件：coding 3 条，seed=0，max_tokens=64，warmup=3，no-think，
no-prefix-cache，max_model_len=1024，max_num_seqs=4，max_num_batched_tokens=1024，
gpu_mem_util=0.80。输出吞吐 **136.21 token/s**，TTFT 中位数 **37.17 ms**。
这些是小样本冒烟结果，不能作为跨硬件或长期稳态性能结论。

七组模型对照还覆盖 FlashInfer eager/Graph（256/64 页）。FlashInfer 的部分结果
仍有已采集 logits 证据的单 ULP 翻转；报告不把它们标成严格一致。
对照工件保存于 `bench_results/flashinfer-check/`。

```bash
cd /mnt/d/Materials/AVIC/FlashEngine
source /home/yjq040828/vllm-env/bin/activate
python3 -m unittest discover -s tests -v
python3 -m scripts.check_cuda_attention /home/yjq040828/vllm-lab/models/Qwen3-0.6B \
  --allow-bf16-ties
```

该 sampler 修复阶段没有真实草稿实机回归；后续 Qwen3.5+DFlash 的验收见
[Qwen3.5 说明](QWEN35.md)。CoreX/310P 仍无实机验证。
