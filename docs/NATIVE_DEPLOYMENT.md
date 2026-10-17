# Python oracle 与独立 C++ 部署仓

**三后端源码增量**：独立仓 `docs/BACKEND_PARITY_SOURCE.md` 记录新 310P Qwen3.5/EAGLE3/Quark 参考执行和公共 chunked-prefill 配置/观测；CoreX 共用 CUDA 模型层。按用户要求本轮未构建、未测试，旧 build 内二进制未更新。历史报告不覆盖这次源码，不能据此宣布后端可部署或性能达标。

## 当前落地

`contracts/` 是两仓公共语义源，当前版本 0.4.0、bundle 1.3.0；读取兼容 0.1/0.2/0.3。Prefix 默认开启，新增 Qwen3 dense/MoE + DFlash/EAGLE3、Qwen3.5 文本 dense/MoE + DFlash 的组合及附属状态恢复。组合限制和验收范围见独立仓 `docs/FEATURE_COMBINATIONS.md`；契约有 profile 不代表所有后端都可运行。`flashengine/deployment.py` 与 `scripts/` 负责离线打包、契约发布和差分。

独立 C++ 仓位于同级目录 `D:/Materials/AVIC/FlashEngine.cpp`，不再嵌套于 Python 仓；它单独提交并配置远程仓库，构建结果不上传。构建、CLI 和阶段状态见该仓 README 与 `docs/VALIDATION.md`；**x86_64/AArch64、多后端部署手册位于该仓 `docs/DEPLOYMENT.md`**。CUDA 已实跑；CoreX/310P 原生源码和构建入口已准备，但没有 SDK/真机，仅为 source-prepared。全部后端仍是 candidate。

Python/C++ 冻结输入、严格 CUDA 行为和 ABBA 基准见 [基准与数值报告](BENCHMARK_NATIVE.md)。2026-10-17 的真实 0.6B FP16 128-token 测试为 7/8 严格相同；即使吞吐达到阈值也不能放行。新增 Python 显式 `dtype` 保证模型与 planner 一致；不传时保留原默认。Completions SSE 文本格式修为 `choices[].text`，Chat 仍使用 `delta.content`。

## 离线工具

以下命令从 Python 仓根目录以模块方式运行。只允许本地模型路径。输出工件拒绝覆盖，失败报告不得原地改成通过。

```bash
# 仅主机单元测试，不导入设备运行时
python3 -m unittest discover -s tests -v

# 新建 bundle，复制原始 safetensors；可选 DFlash 明确 k，不解释历史 sentinel
python3 -m scripts.export_bundle /path/to/Qwen3 /path/to/new-bundle \
  --verifier /path/to/flashengine-native
python3 -m scripts.export_bundle /path/to/Qwen3 /path/to/new-spec-bundle \
  --draft /path/to/DFlash --spec-num-tokens 2 \
  --verifier /path/to/flashengine-native

# 发布版本后同步；升级需新版本和 --upgrade，不覆盖旧发布物
python3 -m scripts.sync_cpp_contracts /path/to/FlashEngine.cpp \
  deployment_output/flashengine-contracts-NEW-VERSION.tar.gz --upgrade

# 固定 token 输入的 oracle；投机轮包含提案、verify、接受数、提交及上下文长度
python3 -m scripts.export_oracle /path/to/Qwen3 /path/to/bundle \
  deployment_output/oracle.json --eager --max-tokens 16 \
  --max-model-len 512 --gpu-mem-util .65 --logits

python3 -m scripts.check_native_replay /path/to/flashengine-native \
  /path/to/bundle deployment_output/oracle.json deployment_output/replay.json \
  -- --max-model-len 512 --max-num-seqs 1 --kv-blocks 8 --graph --logits

python3 -m scripts.check_native_server /path/to/flashengine-native \
  /path/to/bundle deployment_output/server.json \
  -- --max-model-len 256 --max-num-seqs 4 --kv-blocks 8 --graph
```

`check_native_tokenizer` 提供固定种子的 Unicode 扩展差分；`native_build_evidence` 检查 ELF 动态依赖、记录源码/头文件/二进制/动态库哈希及设备信息。不得以动态依赖检查代替完整依赖来源和许可证审计。

`export_native_ops_oracle` / `check_native_ops` 导出并比对实际 Python 编译的 RMSNorm/SwiGLU 等基础算子，区分真实转译差异和编译优化；原生测试程序不需要 Python。当前 CUDA 四种算子、两种 dtype、两种宽度共 139264 元素位等，但整模型 DFlash 的数值门仍未通过。

`check_native_shutdown` 在输出进行中留下半包连接并重复 SIGTERM，保存每轮退出状态。可用真实 CUDA 二进制，也可配合 C++ 仓 `fe-server-probe` 在 ASan 下检查相同网络路径；后者是独立测试目标，不进入部署程序，不证明模型执行能力。

## 已知边界与剩余工作

0.4 打包器支持 Qwen3 dense/MoE + DFlash/EAGLE3、Qwen3.5 文本 dense/MoE + DFlash，Prefix 默认开启并可与这些草稿组合；Quark 仍仅逐通道 int8 加载期反量化。ICU 支持实际 tokenizer 的 NFC，不引入 Rust tokenizer。运行时只读 bundle，不执行模型目录代码。仍拒绝 speculators 平铺 DFlash、滑窗草稿、TP>1、Qwen3.5+EAGLE3，以及没有原生实现的量化格式；加载期反量化不是 int8 GEMM。详见独立 C++ 仓 `docs/FEATURE_COMBINATIONS.md`。

0.6B BF16 的普通路径已实跑 eager 与 Graph，但英文 fixture 出现 Python bf16 平票和原生一 ULP 翻转；保存了 logits、margin、来源哈希，严格报告保持失败，没有全局宽松开关。新增 FP16 eager/Graph 对 Python 三组各 16 token 严格一致、Graph 实际回放 45 次，仍不能豁免 BF16 分歧。1.7B 合成 DFlash 的最终 token 已一致，逐轮提案仍有差异；不能据此宣布 P4 通过或证明性能收益。

合成 MoE 高 margin 夹具新增了非零/满接受覆盖：普通与 DFlash Graph 均与 Python 三组各 16 token 一致，DFlash 逐轮 trace 一致，draft/verify Graph 各 15 次回放。可移植 ATen 参考算子在 NVIDIA 上也通过同样差分；**这不是 CoreX 真机验收，不是训练模型准确性或性能证据**。复现入口为 `scripts.create_native_moe_fixture`。

剩余验收包括：真实模型逐层数值定位/投机 trace 收敛、更多受控接受长度、完整批次/抢占/边界矩阵、长稳和故障注入、同硬件交替热态性能 A/B、全部依赖离线锁定与正式 OCI 交付。CoreX 需要厂商 LibTorch C++ SDK；310P 已准备 AscendCL/ACLNN/ACL Graph 的 ND 参考源码，但 ATB/NZ 高性能路径尚未实现。两者均需真机按 eager→Graph→spec eager→spec Graph 验收；不能以本机 CUDA 结果替代。

发布目标仍为 TPS >= Python 95%、p50 TTFT <= Python 110%，E 与逐位置接受率无显著退化；当前没有足够性能证据判定达标。
