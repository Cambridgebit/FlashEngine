# Python 显存与 Triton 接口验证记录（2026-10-17）

环境：WSL Ubuntu-22.04，RTX 4070 Laptop 8GB，Python 3.12.14，torch 2.11.0+cu130，transformers 5.15.0。

主机回归：50 项发现，48 项通过、2 项真实设备测试默认跳过；另显式运行这 2 项 CUDA 测试均通过。
CUDA 接口测试使用真实引擎，Triton 传输对象使用替身；没有运行真实 Triton Server。

| 配置 | 执行路径 | 循环次数 | 记录的最高峰值 MiB | 第二次卸载增量 MiB |
|---|---|---:|---:|---:|
| [qwen3-dense-eager](qwen3-dense-eager.json) | eager | 2 | 3825.0 | 0.00 |
| [qwen3-dense-graph](qwen3-dense-graph.json) | decode_graph | 2 | 3613.2 | 0.00 |
| [qwen3-dflash-graph](qwen3-dflash-graph.json) | spec_graph | 2 | 5685.6 | 0.00 |
| [qwen3-dspark-graph](qwen3-dspark-graph.json) | spec_graph | 2 | 3645.0 | 0.00 |
| [qwen3-eagle3-graph](qwen3-eagle3-graph.json) | spec_graph | 2 | 5334.9 | 0.00 |
| [qwen3-moe-dflash-graph](qwen3-moe-dflash-graph.json) | spec_graph | 2 | 3640.5 | 0.00 |
| [qwen3-moe-graph](qwen3-moe-graph.json) | decode_graph | 2 | 3641.8 | 0.00 |
| [qwen35-dense-graph](qwen35-dense-graph.json) | decode_graph | 2 | 4775.5 | 0.00 |
| [qwen35-dflash-graph](qwen35-dflash-graph.json) | spec_graph | 2 | 5482.1 | 0.00 |

每个配置：3 轮、每轮 3 个请求、最大活跃数 2；包含 Prefix 命中、partial prefill 取消、取消后的再次推理及重复卸载/加载。较早记录保留当时的观测字段，完整检查逻辑见 scripts/check_python_memory.py。

首轮普通路径出现约 8.125 MiB、图路径出现约 16.25 MiB 进程运行库缓存；第二轮卸载增量均为 0。模型权重/KV 的弱引用销毁另有检查，不能将空闲 KV 常驻池或 cuBLAS 工作区当作请求泄漏。

修复：RoPE 强缓存、recurrent/prefix 验证器的卸载引用、手动 KV 绕过预算、EAGLE3 图缺少累计 KV 长度、同进程第二实例影响存活引擎、最终设备步骤期间取消/超时返回成功。

接口设备回归：预算拒绝后进程组清理与默认设备/dtype 恢复；驱动线程真实流式、取消、非流式顺序、重复加载及第二实例拒绝。日志：adapter-device-tests.log。

未验收：CoreX/310P SDK/真机、TP 多卡、Qwen3.5 MoE 与量化检查点、本轮负载以外的长期/大并发压力，以及真实 Triton Server 协议集成。tiny MoE/合成草稿只作功能与状态验证，不证明训练质量、正式大模型验收或加速收益。

未修改独立 C++ 仓；未提交 Git。接口、预算和外部配置说明见 ../../docs/PYTHON_MEMORY_TRITON.md。
