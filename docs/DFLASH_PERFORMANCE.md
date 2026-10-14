# Qwen3.5 DFlash CUDA 性能修复

## 结果与口径

本地 RTX 4070 Laptop 8GB / WSL2，Qwen3.5-2B + Qwen3.5-2B-Dflash，
bf16、greedy、thinking-off、prefix cache-off。固定 KV=8 块、max_model_len=1024、
max_num_seqs=1、prefill 预算=128。题集为仓库 coding、seed=0、8 条，每条输出 128 token。
每个进程先预热 2 次；主跑每轮 1024 个实际输出 token。指标包含 prefill，不含模型加载/图捕获。

| 配置 | 逐轮 tok/s | 中位 tok/s | E（截断前） |
|---|---|---:|---:|
| 修复前 DFlash，k=15 | 72.74 / 70.65 | 71.70 | 4.143 |
| 最终 DFlash，k=15，独立复验 | 134.59 / 134.47 / 134.98 | 134.59 | 4.083 |
| 同样优化后的普通 Graph | 56.27 / 56.95 / 61.63 | 56.95 | 不适用 |

按这组中位数，DFlash 较修复前提高 **87.7%**，较优化后的普通 Graph 为 **2.36×**。
首轮请求 TTFT 中位数从 214.95ms 降到 49.18ms。
不同阶段重复运行的 DFlash 为约 135–146 tok/s；没有锁频，不能把峰值当稳定承诺。
记录了温度/SM 时钟/功耗；最后复验还记录显存时钟/利用率。
不能仅凭相同 SM 时钟排除所有硬件与桌面干扰。

最终源码再次扫描（每档 8 题 × 2 轮）：

| k | 两轮 tok/s | E |
|---:|---|---:|
| 2 | 85.93 / 85.99 | 2.388 |
| 5 | 114.55 / 114.37 | 3.302 |
| 7 | 130.65 / 131.16 | 3.524 |
| 10 | 135.47 / 135.13 | 3.827 |
| 15 | 140.21 / 140.71 | 4.083 |

当前这组负载的已测最佳为 **k=15**，继续使用检查点 block_size=16 的默认宽度。
优化前两题测试曾是 k=7 更快；融合改变了验证成本，旧调参结论不能直接沿用。
这不是所有模型、上下文、并发或题集上的绝对最优。低 E 负载仍可能不适合投机。

## 已修复的热点

1. **多步 GDN 递推**：原先 18 层 × 每行多个 PyTorch 状态乘法/归约/临时张量。
   CUDA 改为一个 head/value 分片的状态驻留寄存器；单个内核推进多步并写各行终态。
   保留 fp32 SSM 和接受位置回滚能力。q/k 仍按参考式在模型 dtype 内 L2 归一化。
2. **短卷积与历史**：融合因果短卷积、SiLU、raw 历史写回。
   保留“卷积先回投模型 dtype，再激活”的舍入点。decoder padding 不写 dummy 槽。
3. **Gemma / GDN Norm**：融合残差 RMSNorm 与门控 RMSNorm。
   Gemma 残差先按模型 dtype 相加；GDN norm/weight 乘法的中间回投顺序与 HF 保持一致。
4. **prefill**：用多步递推内核替代 CUDA 的逐行三角求解小算子循环，末态只写一次。
   时间长度为运行时参数，不按每个 prompt 长度编译新内核。
   同时复用 host 长度/槽号/初始态元数据，避免每个 GDN 层分别 D2H。
5. **greedy 采样**：旧图即使最终 argmax，仍排序全词表、生成噪声与 nucleus 中间量。
   CUDA 单独捕获 greedy 图，保留原随机采样图；混合/随机请求仍使用完整采样语义。
   eager/prefill 由 host 请求参数选择纯 argmax，不在图内读取设备标量做分支。
6. **状态提交**：图内读取接受数，直接将所选行终态复制回主槽。
   不再在 D2H 后上传 source/target 索引，也不造整块 `index_select` 临时状态。
7. **草稿 attention**：CUDA FlashAttention 模式打包有效前缀 + query KV，用非因果 varlen GQA。
   去掉固定窗口空洞和重复 KV heads，保留 SDPA 参考路径；padding/动态长度已回归。

核心算子位于 `flashengine/backends/cuda_gdn.py` / `cuda_dflash.py`。
没有修改检查点权重，没有降低精度到 int8/FP8，没有改为近似接受规则。
小行数 GEMM 候选在代表形状上未得到稳定整体收益，未纳入生产代码。
后续剖析中，GEMM/大词表 lm_head 仍是主要热点，不能把 E 当加速倍数。

## 验证边界

- `python3 -m unittest discover -s tests -v`：**56/56** 通过，含 CUDA 真实内核与采样图测试。
- 32-token 验收：HF、base eager/Graph、DFlash eager/Graph、FlashInfer Graph/DFlash eager，
  五条提示词严格一致。六条引擎路径独立与同一 HF 工件对比均为 5/5。
- 并发 DFlash k=2：3 请求 → 4 桶，eager/Graph 串行 5/5、并发 3/3 一致。
- prompt=500、窗口=512：恰好 12 个输出，末 11 步单步上下文 fallback，无越界。
- 单 KV 块双请求：18 次真实抢占，完成后状态槽为 0。
- Qwen3-0.6B 普通 Graph：14/14 greedy 输出与修复前 eager 工件一致；随机路径完成冒烟。
- 同一前缀/同一状态双跑目标 verify 诊断：136 轮的相关接受前缀没有 argmax 翻转。
  该诊断单独记录完整 logits 差异，不计入性能结果，也不是所有负载的误差豁免。

长输出性能工件不是完整 HF 质量基准。bf16 下不同累加顺序、attention 内核或批形状
仍可能改变后续 token；128-token 工件与修复前并非全部逐字节一致。
不能将“五条 32-token 严格回归”表述为“所有长输出位等”。

**FlashInfer 的 auto 组合保留参考 prefill**：强制新 prefill 时，五条中一条在第 31 个
token 分歧，未静默豁免；默认组合改回参考 prefill 后六路径与 HF 全过。
FlashInfer 整轮投机 Graph 仍未接入，追求本机最高速请用 FlashAttention。
CoreX/310P 保留参考 GDN/attention 路径；共享 host 元数据优化不依赖 CUDA。
没有这两种硬件的实测，不能宣称其获得同样加速。

## 开关与复现

| 开关 | 默认与作用 |
|---|---|
| `FLASHENGINE_CUDA_GDN=auto|torch|triton` | auto 启用 CUDA 递推/短卷积/Norm/状态提交；torch 回参考式 |
| `FLASHENGINE_CUDA_GDN_PREFILL=0|1` | FlashAttention auto=1，FlashInfer auto=0；强制 FI=1 不推荐 |
| `FLASHENGINE_CUDA_DFLASH_ATTENTION=auto|sdpa|flash_attn` | FlashAttention 模式 auto 用 varlen GQA；FI 保留 SDPA |
| `FLASHENGINE_SPEC_GREEDY_GRAPH=0|1` | 默认 1；0 使用原完整采样图 |
| `FLASHENGINE_DECODE_GREEDY_GRAPH=0|1` | 默认 1；0 使用原完整采样图 |

按本机已测最佳配置启动：

```bash
cd /mnt/d/Materials/AVIC/FlashEngine
source /home/yjq040828/vllm-env/bin/activate
FLASHENGINE_CUDA_ATTENTION=flash_attn python3 -m scripts.openai_server \
  /home/yjq040828/vllm-lab/models/Qwen3.5-2B \
  --spec-model /home/yjq040828/vllm-lab/models/Qwen3.5-2B-Dflash \
  --spec-num-tokens 15 --max-model-len 1024 --max-num-seqs 1 \
  --max-num-batched-tokens 128 --gpu-mem-util 0.85 --no-think
```

精确复现实验探针（固定 8 KV 块；不是 HTTP 吞吐）：

```bash
python3 -m tests.probe_dflash_perf /home/yjq040828/vllm-lab/models/Qwen3.5-2B \
  --draft /home/yjq040828/vllm-lab/models/Qwen3.5-2B-Dflash \
  --sweep 2,5,7,10,15 --num-prompts 8 --max-tokens 128 --repeats 3 \
  --output bench_results/qwen35-check/perf/recheck
```

去掉 `--draft`、使用普通 `--output xxx.json` 得到同环境普通 Graph 基线。
加 `--profile` 的短请求剖析不计入吞吐；`--eager --verify-logits --gdn torch` 为双跑诊断，
该模式的时间不得报为吞吐对照。所有工件在 Git 忽略的 `bench_results/qwen35-check/`。
仍以 `scripts.benchmark` / `scripts.bench_concurrent` 为正式离线 / HTTP 测量入口。
