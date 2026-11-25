# 文档地图

> **状态**（2026-11-24）：本文件只负责「哪份文档讲什么」与维护规则；能力结论一律以
> [能力状态](STATUS.md) 为准，声明矩阵以 [支持矩阵](SUPPORT_MATRIX.md) 为准。

`docs/` 下 24 份文档按**用途**分四层，不要跨层混读：结论层给判断，专题层给原因，
实测层给数字与命令，历史层只作追溯。

## 第一层：结论与声明（先读这两份）

| 文档 | 回答什么问题 | 维护方式 |
|---|---|---|
| [STATUS.md](STATUS.md) | 哪些组合**真的**跑过、效果如何、证据有多强、哪些不要宣称 | 人工维护，每次实机验收后更新；**唯一状态源** |
| [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) | 哪些组合**被引擎接受**（启动不报错） | 声明矩阵由 `flashengine/support_matrix.py` 生成，`scripts/check_docs.py` 校验一致性 |

两者不等价：声明矩阵放行 ≠ 实测有效（反例见 [STATUS §3](STATUS.md)）。

## 第二层：架构与机制（解释「为什么」）

| 文档 | 内容 |
|---|---|
| [ENGINE_RUNTIME.md](ENGINE_RUNTIME.md) | 引擎主循环、调度器、KV 管理、投机提交、图执行器接口 |
| [EAGLE3.md](EAGLE3.md) | EAGLE3 链式草稿：shifted draft 流口径、单图整轮、逐位置接受率 |
| [DFlash.md](DFlash.md) | DFlash / DSpark 块草稿：三步一轮、aux 层语义、两种草稿 KV 形态 |
| [STRUCTURED_DECODING.md](STRUCTURED_DECODING.md) | 约束解码、三种工具协议、JSON Schema 子集、流式与 pi 接入 |
| [QWEN35.md](QWEN35.md) | Qwen3.5 文本适配：GDN 混合层、conv/SSM 状态、图与算子约束 |
| [QWEN35_MOE.md](QWEN35_MOE.md) | Qwen3.5 MoE：路由、专家装载、滑窗草稿语义 |
| [QWEN3_ASR_DESIGN.md](QWEN3_ASR_DESIGN.md) | ASR 设计：audio tower、mel 预处理、请求契约 |
| [LORA.md](LORA.md) | 多 LoRA：rank 拼接、融合权重、图内批次与显存 |

## 第三层：后端与环境（解释「在什么硬件上怎么跑」）

| 文档 | 内容 |
|---|---|
| [ASCEND310P_ENV.md](ASCEND310P_ENV.md) | 310P 环境：CANN/容器/驱动版本与坑 |
| [ASCEND310P_OPS.md](ASCEND310P_OPS.md) | 310P 算子与环境变量开关速查 |
| [ASCEND310P_GRAPH_SPEC.md](ASCEND310P_GRAPH_SPEC.md) | 310P 图规格：NZ KV、mask、PA 元数据刷新、分页草稿 |
| [FLASHINFER.md](FLASHINFER.md) | CUDA FlashInfer 适配与精度边界 |
| [CUDA_GRAPH_SAMPLER.md](CUDA_GRAPH_SAMPLER.md) | 图内采样与设备 RNG |
| [PYTHON_MEMORY_TRITON.md](PYTHON_MEMORY_TRITON.md) | 显存预算、卸载、Triton Python 入口 |
| [NATIVE_DEPLOYMENT.md](NATIVE_DEPLOYMENT.md) | 独立 C++ 仓（candidate）的入口与门禁 |

## 第四层：实测与台账（解释「跑出了什么数字」）

| 文档 | 内容 |
|---|---|
| [ASCEND310P_BENCHMARK.md](ASCEND310P_BENCHMARK.md) | 310P 13 组模型/草稿综合基准（TPS/TTFT/TPOT/E） |
| [BENCH_BS1_310P.md](BENCH_BS1_310P.md) | 单并发六配置（9B/9B+DFlash/9B+LoRA/9B+LoRA+DFlash/8B/8B+DFlash）× 中英 × 长上下文/多轮/工具的延迟画像 |
| [BENCH_TPS_310P.md](BENCH_TPS_310P.md) | `bench_data/` 数据集 TPS（六配置 × 中英）+ Qwen3-1.7B-ASR 的 TTFT/RTF/TPS/CER |
| [ASCEND310P_BENCH.md](ASCEND310P_BENCH.md) | 310P 三组合自测配置速查（同批历史记录） |
| [DFLASH_PERFORMANCE.md](DFLASH_PERFORMANCE.md) | DFlash 性能剖分与 kernel 级优化 |
| [QWEN35_OPT.md](QWEN35_OPT.md) | Qwen3.5 优化台账：**含失败候选与否决清单** |
| [QWEN35_PREFIX_CACHE.md](QWEN35_PREFIX_CACHE.md) | Qwen3.5 前缀缓存：完整块状态快照、命中有效性 |
| [QWEN3_ASR.md](QWEN3_ASR.md) | ASR 三后端实测、CER/RTF 口径、310P 构建期陷阱 |
| [ASCEND310P_W8A8_LONGRUN.md](ASCEND310P_W8A8_LONGRUN.md) | W8A8 长跑退化修复与 seqs=1 长跑结果 |
| [BENCHMARK_NATIVE.md](BENCHMARK_NATIVE.md) | 独立 C++ 仓与 Python 基线的同权重对照 |

## 维护规则

1. **结论只写一处**。新的实机结果写进 [STATUS.md](STATUS.md) 对应行；专题文档只写过程、
   命令、失败细节，不再重复给「已验收/待验收」这类总判断。
2. **改代码必须同步声明矩阵**。`flashengine/support_matrix.py` 变了就跑
   `python -m scripts.check_docs --fix`，让 `SUPPORT_MATRIX.md` 的生成块跟上。
3. **每条结论都要有口径**。写数字必须带：硬件、模型与草稿、k、max_model_len、
   max_num_seqs、数据集、thinking 开关、并发。缺任何一项的旧数字标「口径不全」。
4. **失败也要留档**。被否决的方案写进对应专题文档的「否决清单」，注明「实测更慢/不位等」，
   避免下一个人重跑同样的实验。
5. **过时断言不要删，改成历史**。把「当前 / 尚未」改成「2026-xx-xx 时为 …，
   现已 …」，保留追溯能力。
6. **提交前跑健康检查**：

   ```bash
   python -m scripts.check_docs          # 链接、生成块、状态头一致性
   python -m unittest discover -s tests -v
   ```

## 名词口径

| 名词 | 含义 |
|---|---|
| **E / 接受长度** | 每轮 verify 提交的 token 数（含 bonus）；`E = 1 + 接受数/轮数` |
| **TPS** | 聚合输出吞吐，含 prefill；单请求纯解码速度看 **TPOT** |
| **TTFT** | 首 token 时间；服务端日志含排队，客户端观测含 HTTP 往返 |
| **thinking-off** | 对 chat template 显式 `enable_thinking=False`；默认是 thinking-on，两者 E 不可比 |
| **声明支持** | 引擎接受该组合、启动不报错 |
| **实测状态** | 真机上跑过、有可引用的数字与输出质量记录 |
