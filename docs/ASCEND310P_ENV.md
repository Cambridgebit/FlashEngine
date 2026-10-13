# Ascend 310P 运行环境与依赖

本文记录 FlashEngine 在华为 Ascend 310P（Atlas 300I Duo / 200I Pro）上运行所需的**完整依赖**、
**实测版本**，以及**为什么不需要 vLLM / vLLM-Ascend**。可直接参照
[`requirements-ascend.txt`](../requirements-ascend.txt) 安装。

## 1. 代码依赖分析

引擎运行路径上的第三方 import（不含测试）：

| 位置 | import | 提供包 |
|---|---|---|
| `flashengine/**` | `torch`、`torch.nn`、`torch.nn.functional`、`torch.distributed`、`torch.multiprocessing` | `torch` |
| `flashengine/backends/ascend.py`、`ascend310p/*.py` | `torch_npu` | `torch-npu` |
| `flashengine/config.py`、`engine/llm_engine.py`、`models/qwen3*.py` | `transformers`（`AutoConfig` / `AutoTokenizer` / `Qwen3Config` / `Qwen3MoeConfig`） | `transformers` |
| `flashengine/utils/loader.py`、`models/*.py` | `safetensors` | `safetensors` |
| `flashengine/engine/block_manager.py` | `xxhash`、`numpy` | `xxhash`、`numpy` |
| `flashengine/engine/llm_engine.py`、`scripts/benchmark.py` | `tqdm` | `tqdm` |
| `scripts/openai_server.py` | 仅标准库（`http.server` / `json` / `threading`…） | — |

结论：

- **核心仅 7 个 pip 包**：`torch`、`torch-npu`、`transformers`、`safetensors`、`xxhash`、`numpy`、`tqdm`。
- OpenAI 兼容服务基于**标准库 `http.server`**，不需要 `fastapi` / `uvicorn` / `openai`。
- **`scripts/*` 与 `flashengine/*` 均不 import `vllm`，也不 import `vllm_ascend`（见下节唯一例外）。**

## 2. 为什么可以剥离 vLLM-Ascend

代码中与 vLLM 相关的**全部**位置只有一处：

```python
# flashengine/backends/ascend.py
try:
    import vllm_ascend.ops.linear  # noqa: F401   ← 仅为副作用：尝试注册自定义算子
except ImportError:
    pass
...
self._unquantized_gemm = getattr(getattr(torch.ops, "vllm", None), "unquantized_gemm", None)

def linear(self, x, weight, bias=None):
    if self._unquantized_gemm is not None:
        return self._unquantized_gemm(x, weight, bias)
    return F.linear(x, weight, bias)          # ← 本环境实际走这一条
```

- 该算子并非由 `vllm_ascend.ops.linear` 注册（它只定义 `Ascend*LinearMethod`）；
  `torch.ops.vllm.unquantized_gemm` 的注册在 `vllm_ascend/patch/worker/patch_unquantized_gemm.py`，
  需 vLLM-Ascend 平台插件在 worker 侧打补丁后才存在。
- **实测**：无论是否 import `vllm_ascend.ops.linear`，`torch.ops.vllm.unquantized_gemm` 均为 `None`；
  引擎日志恒为 `GEMM=torch-fallback`。这与 `docs/ASCEND310P_OPS.md` 的算子表一致——
  FlashEngine 的 QKV / O / Gate+Up / Down / LM Head 全部使用 `F.linear`（权重预转 FRACTAL_NZ）。

### 剥离验证（开发板实测）

用 `sitecustomize.py` 在 `sys.meta_path` 前置拦截器屏蔽 `vllm` 与 `vllm_ascend` 后运行基准：

| 指标 | 带 vLLM-Ascend | 屏蔽 vLLM/vLLM-Ascend |
|---|---|---|
| GEMM 路径 | `torch-fallback` | `torch-fallback` |
| `flashengine` 导入 | OK | OK |
| Engine ready | ~40 s | 37.8 s |
| Output token throughput | 10.72 tok/s | 10.77 tok/s |
| Mean TPOT | 92.48 ms | 92.06 ms |

**结论：vLLM-Ascend 对 FlashEngine 在 310P 上的推理完全无贡献，可安全移除。**
保留 `import vllm_ascend.ops.linear`（`try/except`）只是**向后兼容的尽力而为**：
若某环境恰好注册了该算子，则自动启用，否则回退 `F.linear`。

## 3. 实测环境（开发板）

| 组件 | 版本 | 说明 |
|---|---|---|
| OS | openEuler 24.03 LTS | aarch64 |
| Python | 3.11.10 | |
| CANN toolkit | **9.0.T3** | `compiler` / `opp` `version.info` = 9.0.0；`/usr/local/Ascend/cann-9.0.T3` |
| NPU 驱动 | **24.1.rc4**（`7.6.T7.0.B052`，`V100R001C19SPC001B124`） | 主机侧 |
| torch | **2.9.0** | PyPI CPU 构建（`2.9.0+cpu`），NPU 设备由 torch-npu 提供 |
| torch-npu | **2.9.0.post2.dev20260401** | 必须与 torch 主次版本一致 |
| transformers | 4.57.6 | |
| tokenizers / huggingface_hub | 0.22.2 / 0.36.2 | transformers 传递依赖 |
| safetensors / numpy / xxhash / tqdm | 0.7.0 / 1.26.4 / 3.6.0 / 4.67.3 | |
| pip 源 | `https://mirrors.tools.huawei.com/pypi/simple` | 华为镜像 |

> 注：`pyproject.toml` / `requirements.txt` 里旧注释写的“参考环境 CANN 9.1.0 / torch-npu 2.10.0.post4”
> 是另一套历史环境；本开发板为 **CANN 9.0.T3 / torch-npu 2.9.0.post2.dev**，以 `requirements-ascend.txt` 为准。

## 4. 安装步骤

```bash
# 1) 系统组件（非 pip）：安装 Ascend CANN toolkit + NPU 驱动（版本需匹配，本板：CANN 9.0.T3 / driver 24.1.rc4）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2) Python 依赖（torch-npu 需匹配 CANN 与 torch，来自华为 Ascend 源）
python3 -m pip install -r requirements-ascend.txt

# 3) 安装 FlashEngine 本体
python3 -m pip install -e .
```

运行（每进程需先 `source set_env.sh`）：

```bash
python3 -B -m scripts.benchmark /path/to/Qwen3-8B \
    --questions bench_data/question.jsonl --num-prompts 10 \
    --max-model-len 2048 --max-num-seqs 4 --gpu-mem-util 0.3 \
    --backend ascend310p --no-think
```
