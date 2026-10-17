# Python / 原生 C++：严格 CUDA 验证与基准

日期：2026-10-17。这里的“C 版”为同级独立 `FlashEngine.cpp` 仓的 **C++20 原生部署程序**，不是把 Python 包进可执行文件。

## 结论与适用范围

真实 Qwen3-0.6B FP16 的默认 Python FlashAttention 对原生 FlashInfer 的 128-token 基准仍有严格 token 分歧，**不能发布为正确性合格或 production**。显式 Python FlashInfer 对照通过同一组 8 条×128 token，但不替换默认基线。当前性能数字仅用于诊断；没有放宽精度门，也没有将合成 MoE/DFlash 当成训练模型性能证据。

本次所有设备运行在 WSL2 Ubuntu 22.04、x86_64、RTX 4070 Laptop 8 GiB。PyTorch 2.11.0+cu130、CUDA Toolkit 13.0.3、FlashInfer 0.6.14。Python 基线使用原有 FlashAttention，原生使用 FlashInfer C++/CUDA；同模型、同权重、同 dtype、同请求和调度预算，不承诺两个注意力库中间值位等。CoreX、AArch64 和 310P 不在本次实测范围。

## 冻结负载和测量边界

共同配置：FP16、TP=1、greedy、thinking-off、prefix cache 关闭；上下文 2048、最大序列 8、prefill 预算 512、partial-prefill 上限 2、KV 32 页 × 256 token。输入来自仓内 `bench_data/question.jsonl` 的 coding 分类，seed=0、8 条；prompt token 长度为 `[51,45,26,40,191,34,39,38]`。

| 口径 | 起止边界 | EOS | 不可混用 |
|---|---|---|---|
| `pretokenized-grouped-engine-v1` | token IDs 入队前至该轮所有分组完成；不计装载、分词和预热 | 默认 ignore_eos，每请求固定 128 token | batch=1 是串行；batch>1 是组完成后才发下一组，不是 HTTP 连续并发 |
| `pretokenized-http-sse-v1` | 客户端各档同步发车前至所有波次完成 | 必须 respect-eos；由 SSE usage 统计实际 token | TTFT 从发请求至首段非空文本，包含网络和排队；不等于引擎首 token 时间 |

每个实现独立进程，预热 3 次；两轮顺序为 Python → C++ → C++ → Python（ABBA）。不同时运行两个 GPU worker。吞吐按每轮 token 总数 / 墙钟；表中为两轮中位数。TTFT 为每轮请求 p50 的两轮中位数，不是合并样本的 p50。HTTP 每档并发 1/4/8，各 8 个请求；Barrier 同步同波发车，尾波不足时重复头部补齐。

### 引擎串行实测

工件：`deployment_output/bench-qwen06-fp16-serial-v1/{summary,audit}.json`。

| 模式 / batch | Python tok/s | C++ tok/s | C++ / Python | Python TTFT p50 ms | C++ TTFT p50 ms |
|---|---:|---:|---:|---:|---:|
| eager / 1 | 31.46 | 167.81 | 5.33× | 36.03 | 7.28 |
| Graph / 1 | 153.12 | 178.48 | 1.17× | 36.90 | 7.64 |

两端各模式都生成 8 × 128 token，Python 与 C++ **7/8 请求严格相同**；失败输入 id=`2`，共同前缀 21 token。各实现自身 eager/Graph 的 8 条最终 token 全部相同。Graph 测量窗口内两端各 replay 1016 次；不把捕获或预热计数当作稳态命中。

每轮和二进制有固定 SHA256。串行基准后增补了原生诊断 CLI 和请求 E2E 字段，所以不能用当前 ELF 哈希替换旧报告的哈希。模型数学代码未因此修改。样本量仅 8 条 × 2 轮，不能据此给出显著性或长稳结论。

### HTTP 实测的有效性

第一批 `bench-qwen06-fp16-http-v1` 发现 Python Completions SSE 错把文本放在 `delta.text`；客户端没有读到文本，旧统计把缺失 TTFT 写成 0。**此批延迟无效，不能用于比较**，旧报告保留。

已修复 Python 服务输出为 `choices[].text`；Chat 仍为 `delta.content`。新客户端强制校验响应格式、非空输出的 TTFT、prompt 长度、SSE usage/DONE/finish，并与服务端输出计数核对。缺测延迟保留 null/报错，绝不填 0。重跑结果独立保存为 `bench-qwen06-fp16-http-v2`；其有效数字见下方增量结果。

| Graph / HTTP 并发 | Python 聚合 tok/s | C++ 聚合 tok/s | C++ / Python | Python TTFT p50 ms | C++ TTFT p50 ms |
|---|---:|---:|---:|---:|---:|
| 1 | 147.58 | 165.59 | 1.12× | 37.46 | 12.75 |
| 4 | 361.05 | 607.69 | 1.68× | 60.91 | 29.13 |
| 8 | 616.24 | 1025.98 | 1.66× | 141.16 | 47.29 |

4 次服务器进程 × 3 档并发共 96 个测量请求；每档每轮输出 1024 token。新审计报告 `bench-qwen06-fp16-http-v2/audit.json` 通过实际配置、结果哈希、首字样本、usage/服务计数、测量期 Graph replay 及结束后 KV=0 检查。HTTP 文本不作为严格 token/逐轮 trace 的替代。

原生两轮并发 8 均采到 running=8；Python 两轮分别采到 8 和 2，且后一轮有两条请求首字约 1.10 秒。0.1 秒 stats 采样受锁和请求接入时序影响，是峰值下界；**不能声称每轮 Python 都形成完整 batch=8**。不丢弃这轮较低吞吐，也不把差异全归因于设备内核。原生两次停机退出 0，Python 由 SIGTERM 退出 -15；这不是 Python 优雅停机验收。

## 严格 CUDA 行为证据

### 真实模型：保留失败，不扩大解释

`deployment_output/qwen06-fp16-teacher-v1.json` 固定 Python 的 128-token 前缀；native `replay --teacher-force --logits` 在相同前缀上取实际 argmax，再强制提交 oracle token。此模式仅定位数值，**不能当作生成通过**。

第 22 个 token（0-based index=21）：Python 期望 1447、C++ 选择 13。Python top-2 为 `[23.6875,23.671875]`、margin=0.015625；原生为 `[23.671875,23.671875]`。该值附近相差一个 FP16 ULP，但 Python margin 非零，**不自动豁免**。原生 eager 和 Graph 均复现，说明并非只在 Graph 下出现。完整逐层 attention/GEMM 定位尚未完成。

已有 BF16 普通路径及真实 1.7B+合成 DFlash 的失败报告继续有效。最终 token 一致但未提交 verify 行不同，也判逐轮 trace 失败。

补充真实模型短输出矩阵 `deployment_output/strict-qwen06-fp16-batch-v1`：相同 8 条 prompt、每条 16 token，eager/Graph × batch 1/2/4/8 × Python/C++ 共 16 个 worker，64 对请求严格相同；各实现 eager/Graph 也全部一致。配置/哈希和测量期 Graph 审计通过。此轮为行为验证，只有一次测量、预热一次，且部分运行与主机编译重叠，**不将这组吞吐作为性能数字**。通过短输出不豁免第 22 个 token 的长输出失败。

附加 attention 对照，不换主基准：`qwen06-fp16-teacher-flashinfer-v2.json` 沿同一 FlashAttention token 前缀改用 Python FlashInfer，128 个实际 argmax 全部与原生 teacher 结果一致，第 22 个 token 也选 13；top-8 logits 只有 1/128 个位置完全一致，中间舍入仍不同。这支持 attention 实现选择影响该分歧的判断，**不等于所有中间层已经定位或位等**。

自由生成对照 `strict-qwen06-fp16-flashinfer-v1` 使用显式 `--attention flashinfer`：相同 8 条×128 token，batch=1，eager/Graph 两模式 Python/C++ 均 8/8 严格相同；各自 eager/Graph 也相同，配置/哈希/测量期 Graph 审计通过。这里只跑一轮、预热一次，是诊断而非新的 ABBA 性能基准；不把较有利的对照结果替换默认 FlashAttention 的失败。此结果不扩展到 BF16、其他批次、DFlash 或其他模型。

复现该对照时在上述 `run_native_benchmark` 命令末尾添加 `--attention flashinfer` 并使用新输出目录。Python FlashInfer 工具需要激活已有虚拟环境及 CUDA 工具 PATH（含 ninja）；初次缺 ninja 路径的尝试未形成有效 oracle。Python 侧依赖可能离线 JIT，这不是原生部署进程的运行路径；原生仍使用预编译 C++/CUDA。

### 合成 MoE/DFlash：状态与边界，不是质量或收益

`deployment_output/strict-moe-matrix-v2`：FP16、4 层、4 专家 top-2、mixed dense/sparse、融合专家权重、DFlash k=2。9 个 raw prompt 长度 `[1,7,15,16,17,31,46,47,8]`，上下文 64、输出 17、prefill 预算 8、KV 4×256、prefix-off。

eager/Graph × batch 1/2/4/8 × Python/C++ 共 16 个 worker，最终 token 和每轮 `draft_ids/verify_ids/accepted_count/committed_ids/context_length` 全部严格一致。覆盖组尾单请求、分块 prefill、请求重排、真实 KV 抢占、上下文尾部单 token 回退及资源释放；所有配置、结果哈希和测量期 Graph 门通过。九种长度只是本次测试点，不代表全范围已穷举。

EOS 夹具分别测试部分接受后的 bonus=EOS，及 EOS 出现在已接受提案中。高 margin 夹具与算子穷举测试互补，不能替代训练 MoE/DFlash 模型验收。合成结果中的 TPS 不作公开性能比较。

| 夹具 / 工件目录 | 最终 token / 完成原因 | 原始投机轮行为 | 已测范围 |
|---|---|---|---|
| `strict-moe-partial-eos-v1` | `[42,43,151645]` / stop | accepted=1，commit=`[43,EOS]` | 9 请求 × eager/Graph × Python/C++ |
| `strict-moe-accepted-eos-v1` | `[43,151645]` / stop | accepted=2，原始 commit 含 3 个 EOS，scheduler 在首个 EOS 停止 | 同上，prompt 均为 1 token |
| `strict-moe-output-cap-v1` | `[42,43]` / length | accepted=1，原始 commit=`[43,EOS]`，max_tokens=2 提前截断 | 同上，九种 prompt 长度 |

三种夹具均 batch=8；最终 token/完成原因和每轮 trace 严格一致。另按上表预期直接断言输出，不仅比较两个实现是否碰巧相同。Graph 测量窗口中原生 draft 和 verify 必须分别非零，Python 必须为 spec_graph，不能只凭已捕获或预热计数放行。

### CUDA 内存与主机回归

Compute Sanitizer memcheck 实跑三条原生命令，均退出 0、`ERROR SUMMARY: 0 errors`：

- 接受/Graph 算子：4050 项检查、540 次 Graph replay，batch 1/2/4/8，k=1..15。
- 融合专家 MoE+DFlash：九种 prompt 长度、batch=8、Graph、预热 1 次，包含实际抢占。
- 真实 Qwen3-0.6B FP16：8 请求×16 token、batch=8、Graph、预热 1 次。

报告为 `cuda-acceptance-memcheck-v1.json`、`cuda-moe-dflash-memcheck-v1.json`、`cuda-qwen06-graph-memcheck-v1.json`。它们只证明这些具体命令没有检出的设备内存错误，不代表全部形状、长稳或 racecheck 通过。带内存检查的吞吐不得用于 benchmark。

最终 Python 纯主机 unittest 21/21，C++ host CTest 2/2、ASan/UBSan CTest 2/2、CUDA CTest 3/3 通过。ASan 只覆盖主机内存，不代替上述设备 memcheck。构建依赖证据为 `native-build-cuda-benchmark-v1.json`，未链接 libpython/libtorch_python/librust；完整供应链与正式镜像仍待验收。

## 复现：串行、分组和 HTTP

从 Python 仓根目录以模块方式执行。以下变量替换为本地绝对路径；先按 C++ 仓部署手册完成 CUDA 编译和 bundle 验证。所有新工件禁止覆盖，重跑使用新目录。

```bash
PY=/path/to/env/bin/python3
MODEL=/models/Qwen3-0.6B
BUNDLE=/bundles/qwen3-native
NATIVE=/path/to/FlashEngine.cpp/build/cuda/flashengine-native

$PY -m scripts.make_native_workload "$MODEL" "$BUNDLE" /reports/workload.json \
  --questions bench_data/question.jsonl --category coding --seed 0 \
  --num-prompts 8 --max-tokens 128 --dtype float16 \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 512 \
  --max-num-partial-prefills 2 --kv-block-size 256 --kv-blocks 32

$PY -m scripts.run_native_benchmark "$NATIVE" "$MODEL" "$BUNDLE" \
  /reports/workload.json /reports/engine-abba \
  --modes eager,graph --batch-sizes 1 --warmup 3 --rounds 2
$PY -m scripts.audit_native_benchmark /reports/engine-abba /reports/engine-audit.json
```

即使性能达到阈值，严格 token 对比失败仍返回非零。`--batch-sizes 1,2,4,8` 扩大组调度矩阵；`--draft /models/DFlash --trace-rounds` 用于逐轮严格验证，开启 trace 后不采信其性能数字。Python 历史 sentinel 影响显式 k=3/4，当前工具明确拒绝而非悄悄换 k。

可单独调用两端 worker：

```bash
$PY -m scripts.bench_python_native "$MODEL" "$BUNDLE" \
  /reports/workload.json /reports/python.json --batch-size 1 --graph --warmup 3
"$NATIVE" benchmark "$BUNDLE" --workload /reports/workload.json \
  --batch-size 1 --graph --warmup 3 --repeats 1 > /reports/cpp-stdout.json
```

HTTP 使用同配置但 respect-eos 的独立 workload；不能把 ignore-eos 的冻结语义悄悄改掉：

```bash
$PY -m scripts.make_native_workload "$MODEL" "$BUNDLE" /reports/http-workload.json \
  --num-prompts 8 --max-tokens 128 --dtype float16 --respect-eos \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 512 \
  --max-num-partial-prefills 2 --kv-block-size 256 --kv-blocks 32
$PY -m scripts.run_native_http_benchmark "$NATIVE" "$MODEL" "$BUNDLE" \
  /reports/http-workload.json /reports/http-abba \
  --concurrency 1,4,8 --warmup 3 --rounds 2
$PY -m scripts.audit_native_http_benchmark /reports/http-abba /reports/http-audit.json
```

外部服务可直接运行 `python3 -m scripts.bench_frozen_http BASE_URL WORKLOAD OUTPUT --concurrency 1,4,8`。该工具校验 `/v1/stats` 的实际配置；需要 stats 可访问、无 API-key 的本机基准环境。不要对不受信任网络开放这些测试服务。

## 归档与发布门

`summary.json` 保存聚合与严格比较；每个 worker 有结果、命令/stdout/stderr/GPU 采样的 `.process.json`、SHA256；HTTP 另有 `.server.log` 和停机退出码。审计报告读取测量期前后快照，检查配置、KV 释放和 Graph replay，不以预热冒充命中。teacher 工件保存 logits、来源哈希和真实 argmax。

GPU 采样只读取温度、SM 时钟、显存和功耗，不更改设备状态；本机是有桌面占用的 Laptop，未锁频，也未排除所有热态波动。正式验收仍需更多训练模型/草稿、数值收敛、长稳、故障、CUDA 内存检查、热态统计和依赖/镜像交付。吞吐 >= Python 95%、TTFT p50 <= Python 110% 只是一部分门，不能覆盖正确性失败。

HTTP v2 测量窗口的 75 个 GPU 样本为 68–77°C、SM 2550 MHz；采样不是连续设备 trace，也不证明没有所有短暂降频。报告目录在父仓被忽略，交付时须另行归档。

固定报告 SHA256（每份报告内另有当次二进制哈希）：

| 工件，均位于 `deployment_output/` | SHA256 |
|---|---|
| `bench-qwen06-fp16-serial-v1/summary.json` | `b3bb1b6d154e894f13ff0f506aa551dc98eacc279d9cd0a25bd255a19e149ec5` |
| `bench-qwen06-fp16-http-v2/summary.json` | `19ed7a4f1ee10d7db842bacd1367183123ee141f130a0375363645e0d769183b` |
| `bench-qwen06-fp16-http-v2/audit.json` | `e02bc0011979f35feba7322906888be3dc0432ba81f2097515d7f9f81585566d` |
| `strict-moe-matrix-v2/summary.json` | `a54282549122dcca7d004cb0b81df5d726493266a82f4918985f4295d750c355` |
| `strict-qwen06-fp16-batch-v1/summary.json` | `21739ea803504cccfc34e0fe6a381bdfc791360f1d37ab0411ac223dec407974` |
| `strict-qwen06-fp16-flashinfer-v1/summary.json` | `7eee9b635516b54cd9dcaeaeec9e08938c461058c2ed221e15393e0d244e0f75` |
| `qwen06-fp16-teacher-flashinfer-v2.json` | `38e96ffc13fb7beb0a67c7c6e968206fe91fc685121468122ef51faffd585e5c` |
| `cuda-moe-dflash-memcheck-v1.json` | `c2a27370c4c2f77709eb6e1e9ba714fd77ca189480c44f7f18c098db9358cfb9` |
| `cuda-qwen06-graph-memcheck-v1.json` | `b150af31a36f7734ae2fd324fff64f233b21ad062eb03d9fceb28ed053520399` |
| `native-build-cuda-benchmark-v1.json` | `8284690313fd829a346f592558c13bbe0555ce4b3c0e0f65228fb788f7d6fad7` |
| `qwen06-fp16-teacher-native-eager-v1.json`（失败） | `d11e1980730fd570535763d0633b666ce36ef43b66b631aad9fdcfcfec0cb026` |
| `qwen06-fp16-teacher-native-graph-v1.json`（失败） | `a68f883fde91a0fc86a3538e959d227f13bd0357da63a70a30edc782759e7c96` |
