# CUDA FlashInfer 注意力

CUDA 默认保留 `flash-attn`：prefill 和图内 decode 使用
`flash_attn_varlen_func`，eager decode 使用 `flash_attn_with_kvcache`。
KV 写入由仓库自身的 Triton kernel 完成。

设置 `FLASHENGINE_CUDA_ATTENTION=flashinfer` 可切换至 `flashinfer-python`。
这里直接调用与 vLLM FlashInfer 后端相同的公开 wrapper，不依赖 vLLM 的
调度器、AttentionMetadata 或整个 vLLM 安装。Ada 卡使用 FlashInfer 的 `fa2`
内核后端与 tensor cores；这不等于导入 Dao-AILab 的 `flash_attn` Python 包。

## 路径

| 场景 | FlashInfer 接口 |
|---|---|
| 冷 prefill（包括模型预热） | `BatchPrefillWithRaggedKVCacheWrapper` |
| 前缀命中、分块 prefill、投机 verify | `BatchPrefillWithPagedKVCacheWrapper` |
| eager decode | `BatchDecodeWithPagedKVCacheWrapper` |
| decode Graph | 每桶一个稳定地址的 `BatchDecodeWithPagedKVCacheWrapper` |

KV 布局仍为 BSHD（FlashInfer 称 NHD）。页表先转换成 CSR：页偏移、有效物理页索引、
末页长度。末页正好满时长度为 block size，不能写零。保留原 KV 写入与模型层实现。

FlashInfer 可以使用配置允许的 16、32、64、128、256 等 2 的幂页大小，
不受 `flash-attn` 的 256 倍数限制。默认仍是 256；需要小页时显式传 `--block-size 64`。

FlashInfer 的 `plan` 包含主机操作，必须在 Graph 外运行。普通 decode 在每次 replay 前
从当前请求的页表重新 plan，图内只做模型计算与 wrapper.run。padding 的 KV 写槽为 -1，
只读取有效页，不覆盖真实序列。支持 greedy 图内 argmax；随机采样继续使用原 sampler。
本次 FlashInfer 适配当前要求 TP=1；TP>1 在配置阶段明确拒绝。

**当前边界**：FlashInfer 的整轮投机图尚未接入，`supports_spec_graph=False`，
投机请求走 eager spec。`FLASHENGINE_SPEC_GRAPH=1` 不能绕过这个能力门。
DFlash/DSpark 在 FlashInfer 模式下的稠密草稿非因果 attention 仍使用 PyTorch SDPA；
FlashAttention 模式现可用非因果 varlen GQA，详见 [DFlash 性能修复](DFLASH_PERFORMANCE.md)。
这里替换的是目标模型/分页 Attention 及 EAGLE3 Attention，不包含 MLP、Norm、RoPE 等算子。
Qwen3.5 的 GDN/Norm 融合是独立功能；FI 的 auto 保留参考 prefill，避免强制组合的 BF16 回归。

128 MiB workspace 在模型预热时分配，计入后续 KV 池预算。首次运行可能 JIT 编译，
需要 CUDA toolkit、C++ 编译器和虚拟环境中的 ninja；首次编译时间不应计入稳态吞吐。

## WSL 启动

在 WSL 的仓库根目录执行（本工作区为 `/mnt/d/Materials/AVIC/FlashEngine`）：

```bash
source /home/yjq040828/vllm-env/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHENGINE_CUDA_ATTENTION=flashinfer
python3 -m scripts.openai_server /home/yjq040828/vllm-lab/models/Qwen3-0.6B \
  --backend cuda --host 127.0.0.1 --port 8000 --no-think \
  --max-model-len 1024 --max-num-seqs 4 --gpu-mem-util 0.80 --block-size 256
```

启动摘要和 `/v1/stats` 的 backend_info 会标记实际 attention backend。
切回原实现时设置 `FLASHENGINE_CUDA_ATTENTION=flash_attn` 并重启进程。
同一进程的后端为单例，修改环境变量后需要重启。

## 回归

```bash
python3 -m scripts.check_cuda_attention /home/yjq040828/vllm-lab/models/Qwen3-0.6B
# 审阅分歧 logits 后，显式允许单 ULP 平票翻转：
python3 -m scripts.check_cuda_attention /home/yjq040828/vllm-lab/models/Qwen3-0.6B \
  --compare-only --allow-bf16-ties
python3 -m unittest discover -s tests -p 'test_flashinfer_metadata.py' -v
```

模型检查依次启动七个独立进程：原 FlashAttention eager、Graph 图内/图外采样，
FlashInfer eager/Graph（256 页）、FlashInfer eager/Graph（64 页）。结果保存到
`bench_results/flashinfer-check/`（JSON、日志与 summary）。

检查范围：三个 attention wrapper 对 FP32 参考值、batch=1/3/4 的贪心 token，
前缀冷/热请求、128-token 分块 prefill、Graph 捕获/回放/无 fallback、随机采样。
它使用 Qwen3-0.6B，没有草稿模型，不代表真实投机模型的验收。
bf16 在算子、页大小或 batch 形状变化时可能出现近邻 logits 翻转。
脚本默认要求严格一致。显式传 `--allow-bf16-ties` 时，只允许首次分歧处两侧候选都在
最大 logit 的一个 ULP 范围内，且每个候选在两侧的分数变化也不超过一个 ULP。
topk 的平票排序本身不作为判断标准。JSON 始终保留严格一致性、分歧位置与分数证据。

## 本机验证（2026-10-13）

WSL Ubuntu-22.04 / RTX 4070 Laptop 8GB，torch 2.11.0，FlashInfer 0.6.14，
flash-attn 2.8.3，模型为用户本地 Qwen3-0.6B（bf16）。

采样器修复后的七组对照：FlashAttention eager、Graph 图内/图外采样各 14/14 严格一致。
FlashInfer 256 页 eager/Graph 均为 12/14；64 页 eager 为 12/14、Graph 为 10/14，
其余分歧均满足上面的单 ULP 证据条件。此前一轮 256 页也曾为 14/14。
观测到的近邻翻转分数包括 `24.375/24.25` 对 `24.25/24.25`（ULP=0.125），
所以不能承诺所有 bf16 负载位等。
三种 wrapper 对 FP32 参考 attention 的最大绝对误差不超过 0.0028。
Graph 各桶捕获成功且没有 eager fallback；长前缀冷/热、分块 prefill、并发 padding 与
随机采样路径均跑通。具体计数与逐请求结果以本机生成的 `summary.json` 和 worker JSON 为准。

主机侧测试覆盖架构边界、页表元数据与精度审查；实机模型回归仍需单独执行。

原 `scripts.benchmark` 的独立冒烟参数为：coding 3 条、seed=0、max_tokens=64、
warmup=3、no-think、no-prefix-cache、max_model_len=1024、max_num_seqs=4、
max_num_batched_tokens=1024、gpu_mem_util=0.80。FlashInfer 输出吞吐为 129.89 token/s，
TTFT 中位数 40.70 ms，decode 图 3 桶成功、189 次回放、0 fallback。
这是小样本冒烟读数，不是稳定性能结论。

早期对照的 FlashAttention 默认图捕获失败，回退 eager 后为 21.79 token/s，
这个数不能与上面的图路径计算为算子加速比。采样器问题现已修复：CUDA 在设备上
生成随机数，并修复 `FLASHENGINE_GRAPH_SAMPLE=0` 的输出缓冲写回。
修复后同参数 FlashAttention 为 136.21 token/s、189 次图回放、0 fallback。
原因、采样语义与测试细节见 [CUDA Graph 采样修复](CUDA_GRAPH_SAMPLER.md)。

接口参考：[FlashInfer attention API](https://docs.flashinfer.ai/api/attention.html)、
[vLLM FlashInfer 后端](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py)。
