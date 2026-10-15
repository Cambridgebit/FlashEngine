# Qwen3.5 MoE 文本适配与小模型验收

目标注册名为 `qwen3_5_moe`，覆盖 `Qwen3_5MoeForConditionalGeneration` 的文本部分
和 `Qwen3_5MoeForCausalLM`。读取外层 `text_config` 或纯文本配置，视觉塔与 MTP 不加载。
当前为 fp16/bf16、TP=1；普通生成与 DFlash 接口复用 Qwen3.5 混合注意力引擎。
本次在已有初步实现上补齐路由语义、严格权重装载和公开检查点验收。

## 模型与后端

- GDN、门控全注意力、Gemma RMSNorm、conv/SSM 状态、分块 prefill 和前缀缓存沿用
  Qwen3.5 主干。只有 full_attention 层分配分页目标 KV。
- 每个 MoE 层执行 fp32 softmax → top-k → 归一化 → 路由专家，再加上
  `sigmoid(shared_expert_gate(x)) * shared_expert(x)`。Qwen3.5 始终归一化 top-k，
  不继承 Qwen3 的可选 `norm_topk_prob=False` 语义。
- CUDA/CoreX 复用 `BackendBase.moe_router/fused_moe` 的静态张量计算；CUDA 使用
  FlashAttention 与 CUDA Graph，CoreX 使用 IXFormer/HND 与 CUDA 兼容图接口。
- 310P 复用 fp16/NZ 权重、`npu_moe_init_routing_v2` → grouped matmul → SwiGLU →
  grouped matmul → token unpermute，并通过 NPUGraph 执行。隐层须按 16 对齐。
  本次以 CPU 数学替身检查实际 NPU 调用链，未在板卡上执行 Qwen3.5 MoE。
- DFlash 共享目标 embedding/lm_head，以所选 MoE 层的完整输出作 aux；层号不加一。
  一轮并行生成 k 个提案、目标验证 anchor+k 行，再将接受行 `count-1` 的 GDN 终态提交。
  草稿 KV 与目标 KV、conv/SSM 快照共同约束前缀命中；抢占与取消一起回收。
- 同时装载逐专家 `experts.i.{gate,up,down}_proj.weight` 与融合三维
  `experts.{gate_up,down}_proj`（兼容 `.weight` 后缀）。缺失专家/packed 分片、
  越界专家号、可广播但错误的专家形状均报错。绑词嵌入时允许省略输出头，否则必须装载。

## DFlash 检查点与窗口注意力

真实模型可配对 [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
与 [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash)。
2026-10-15 核对的新版草稿为 6 层，其中 5 层 `sliding_attention`、窗口 4096，
最后一层 `full_attention`。旧版全局非因果草稿仍兼容。

此前草稿实现忽略 `layer_types`，会把新版滑窗层当成全局非因果层。
现按[官方实现](https://github.com/z-lab/dflash/blob/main/dflash/model.py)逐层处理：
滑窗层默认因果、全局层默认非因果；显式 `is_causal` 可覆盖默认值。
窗口包含当前位置，非因果窗口双向限制。掩码使用实际上下文长度，不把缓存 padding 当 token。
缺少窗口、非法层类型、目标/草稿隐层或词表不匹配在启动时拒绝。

| 后端 | 草稿注意力与缓存 |
|---|---|
| CUDA / FlashAttention | 打包实际上下文，原生 causal + window_size；支持整轮 Graph |
| CUDA / FlashInfer | 草稿用通用稠密掩码；投机仍 eager，原有限制不变 |
| CoreX | 通用稠密 KV + 显式掩码；目标 KV 仍 HND |
| 310P | 全局非因果草稿保留 NZ/paged SplitFuse；窗口或因果草稿改走稠密掩码，目标 KV 仍 NZ |

310P 的分页草稿内核当前没有逐层窗口接口，不能给它传一个全零 mask 冒充滑窗。
窗口路径同时切换池分配、草稿写入和 prefix 快照机制，启动摘要会注明 masked 路径。
该兼容路径优先保证语义；大模型/长上下文的性能仍需板卡测量。

```bash
python3 -m scripts.openai_server /path/to/Qwen3.5-35B-A3B \
  --spec-model /path/to/Qwen3.5-35B-A3B-DFlash --backend cuda \
  --spec-num-tokens 7 --max-model-len 2048 --max-num-seqs 1 --gpu-mem-util 0.85
```

CoreX/310P 替换 backend。k=7 只是起始配置，不是本模型实测最优值。
完整 MoE 需装载全部专家权重，35B 目标不能在本机 8GB 卡上测试。

## 4GB 内的公开测试检查点

采用 [optimum-intel-internal-testing/tiny-random-qwen3.5-moe](https://huggingface.co/optimum-intel-internal-testing/tiny-random-qwen3.5-moe)，
固定 revision `3061bafacf5a42fc57e620dc54ab19fae19ad594`，Apache-2.0。
权重 `model.safetensors` 为 64,564,168 字节（约 64.6 MB），低于 4GB。
结构为 hidden=32、4 层（3 GDN + 1 full attention）、4 个专家、top-2、共享专家，
最大上下文 512，完整词表 248320。原始权重 fp32；CUDA 加载时按后端规则转换为 bf16。

这是随机初始化的结构检查点，用于验证权重、路由、状态和执行路径，不能用于评估问答质量，
也不能用其吞吐估算 35B/122B/397B 等完整检查点的性能。

本机下载位置（Git 忽略）：
`bench_results/qwen35-check/models/tiny-random-qwen3.5-moe`。
联网环境可通过以下命令获取同一版本；无需执行仓库中的远程代码：

```bash
hf download optimum-intel-internal-testing/tiny-random-qwen3.5-moe \
  --revision 3061bafacf5a42fc57e620dc54ab19fae19ad594 --local-dir /path/to/tiny-qwen35-moe
```

## 验收命令

从仓库根目录以模块方式运行；WSL 使用现有 `~/vllm-env`。

```bash
# 无 GPU：专家加载/共享专家/三后端支持表/310P 调度接线
python3 -m scripts.check_qwen35_moe --host-only
# CUDA：官方 BF16 参考、eager/Graph、三请求进四桶、重复请求和上下文硬停
python3 -m scripts.check_qwen35 /path/to/tiny-qwen35-moe \
  --cases hf,base_eager,base_graph --seqs 4 --max-tokens 16 --boundary \
  --output bench_results/qwen35-check/moe-short
# 512-token 窗口：冷热缓存、非块对齐 chunk、并发、LRU、取消及真实抢占
python3 -m scripts.check_qwen35_prefix /path/to/tiny-qwen35-moe \
  --cases base_eager,base_graph --max-model-len 512 --prefix-repeats 20 \
  --min-hit-blocks 1 --preempt --output bench_results/qwen35-check/moe-prefix
# 厂商环境：替换 backend；只运行引擎对照，不在 NPU 上启动 CUDA HF 参考
python3 -m scripts.check_qwen35 /path/to/tiny-qwen35-moe \
  --backend ascend310p --cases base_eager,base_graph --max-tokens 16 --seqs 4
```

CoreX 将最后一条的 backend 替换为 `corex`。真实模型必须预留全部专家权重的显存；
激活参数量 A3B 不代表仅需装载 3B 参数。

### 小型 MoE + DFlash 功能验收

没有找到与这个 hidden=32 检查点匹配的已训练草稿。生成器使用固定种子 35 创建
两层草稿（约 89KB），共享目标 embedding/lm_head，aux=[0,1,2]，block_size=16。
默认包含一个窗口为 8 的因果层和一个全局非因果层；`--draft-window 0` 可生成旧式草稿。
它是未训练的测试工件，不是可用的加速模型，不在 Git 中保存权重。

```bash
python3 -m scripts.check_qwen35_moe --make-draft /path/to/tiny-qwen35-moe \
  --output /path/to/tiny-moe-dflash
python3 -m scripts.check_qwen35_moe --cuda-attention-only
python3 -m scripts.check_qwen35 /path/to/tiny-qwen35-moe --draft /path/to/tiny-moe-dflash \
  --cases base_eager,spec_eager,spec_graph --seqs 4 --max-tokens 16 \
  --spec-num-tokens 2 --min-acceptance-length 1 --boundary \
  --output bench_results/qwen35-check/moe-dflash
python3 -m scripts.check_qwen35 /path/to/tiny-qwen35-moe --draft /path/to/tiny-moe-dflash \
  --cases base_eager,spec_eager,spec_graph --seqs 4 --max-tokens 16 \
  --spec-num-tokens 2 --min-acceptance-length 1 --controlled-proposals \
  --output bench_results/qwen35-check/moe-dflash-controlled
python3 -m scripts.check_qwen35_prefix /path/to/tiny-qwen35-moe --draft /path/to/tiny-moe-dflash \
  --cases spec_eager,spec_graph --max-model-len 512 --prefix-repeats 20 \
  --min-hit-blocks 1 --preempt --output bench_results/qwen35-check/moe-dflash-prefix
```

原始随机草稿 E=1，显式 `--min-acceptance-length 1` 只关闭性能门槛，贪心 token 与图回放
断言不变；已训练草稿默认门槛仍为 1.5。`--controlled-proposals` 先真实执行草稿，
再注入已知提案，以覆盖全拒绝、部分接受与全接受。目标前向、接受器和状态提交仍用真实代码。
此模式不是模型质量或性能测试，工件单独记录 `controlled_acceptance_histogram`。

## 2026-10-15 验收记录

环境：WSL Ubuntu-22.04、RTX 4070 Laptop 8GB、torch 2.11.0+cu130、transformers 5.15.0。

- CPU 契约通过：融合/逐专家权重等价、缺失/错误形状拒绝、绑词嵌入、路由与共享专家
  对照、310P 分组 GEMM 权重转置/分派/逆置换，以及 1024 行路由分块边界。
- CUDA FlashAttention：16-token 的五条提示词与同精度 HF 参考严格 5/5；eager 与 Graph
  的 32-token 五条提示词严格 5/5，三请求并发严格 3/3。Graph 桶为 1/2/4，回放无回退。
- 上下文测试在 500-token 输入、512 上限处仅提交 12 个输出；完成后主状态槽为 0。
- 前缀缓存 eager/Graph 各串行 10/10、并发 3/3 与关闭缓存一致。system/user 都命中
  一个 256-token 块；两路径各触发 6 次真实抢占，取消后主状态槽为 0，LRU/统计通过。
- 32-token HF 对照为 4/5，不能宣称所有长输出位等。剩余样例在零基 step=25：HF
  token 175131 的 logit=0.50390625，104563=0.5；引擎两者均为 0.50390625，
  argmax 取较小 ID。完整 logits 已采集，严格脚本仍报失败，不将 BF16 平票静默豁免。
- 本机 FlashInfer fa2 对该 tiny 模型的 head_dim=16 在 ragged prefill 报
  `Invalid configuration`；该检查点用 FlashAttention 验收。此结果不代表完整模型
  head_dim=256 的 FlashInfer 路径不支持，也不将失败用例标成通过。
- CoreX/310P 代码路径及契约已覆盖，本次无对应硬件，不宣称完成板卡实机验收。
- MoE + DFlash：窗口草稿 k=2 的 eager/Graph 各串行 5/5、并发 3/3 与普通解码严格一致。
  Graph 桶 1/2/4 捕获成功；500-token 输入在 512 处仅提交 12 个输出，收尾无主槽残留。
- 受控提案的 eager/Graph 均覆盖 E=1/2/3（分别 22/20/21 轮），输出仍为 5/5 + 3/3。
  不将受控 E 当作已训练草稿的接受率。
- CUDA 草稿注意力直接对照：全局/因果/双向窗口共 8 组，eager 和更新实际长度后的
  Graph replay 均通过，包含零上下文与窗口边界。主机契约额外检查 310P 稠密数学路径，
  三后端全局/窗口草稿的池分配和 prefix 门控也通过。
- DFlash prefix：eager/Graph 各串行 10/10、并发 3/3 与关闭缓存一致，system/user
  命中 1 个块；草稿与 recurrent 快照恢复、缺失状态重算、LRU、取消、各 6 次真实抢占通过。
- 旧式全局草稿 k=15 的 eager/Graph 各 5/5；接近上下文上限的单 token fallback 正常。
  本次没有已训练 Qwen3.5 MoE 草稿的接受率、质量或加速比结论。
- 通用 SDPA 窗口路径在 CUDA 复验 5/5 + 3/3，Graph 正常回放；既有 Qwen3.5-2B
  已训练 DFlash 的 8-token 回归中，普通 eager 与 spec Graph 严格 5/5。

工件在 Git 忽略的 `bench_results/qwen35-check/moe-*`。
