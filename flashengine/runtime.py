"""统一运行配置与 Python API：离线生成、HTTP 服务与验收脚本共用同一份 JSON 配置。

设计口径（与 `.merge_tmp/CONTRACT.md` 一致）：

- 本模块 import 期**只依赖标准库**（json/math/os/pathlib/dataclasses/warnings），不导入
  torch、transformers 或任何后端运行时；设备与引擎只在 :func:`create_engine` 里延迟导入。
  这样 `scripts/framework.py`、`scripts/service_control.py` 等 host 工具可以在没有设备
  运行时的机器上解析/展示配置。
- 字段是主项目 ``flashengine.config.Config`` 的**超集子集**：只覆盖 Config 真实存在的
  构造参数（见 :meth:`RuntimeSettings.engine_kwargs`），不确定的字段一律不进 kwargs。
- 取值域只做「字符串/范围」前置校验；**目标模型 × 后端 × 量化 × 投机**的合法组合交给
  ``flashengine.support_matrix.require_supported`` 与 ``Config.__post_init__`` 在构造引擎时
  裁决，本模块不重复硬编码支持面（否则会随主项目能力演进漂移）。
- 默认值保持主项目现状：``backend="auto"``、``precision="auto"``（跟随 hf config 的
  dtype）、``enable_prefix_cache=True``；随包的 ``configs/*.json`` 才是具体档位示例。
"""

from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import warnings

from flashengine.constraints import (DEFAULT_BANNED_WORDS, normalize_banned_words,
                                     resolve_constraint_words)

# 取值域：与主项目 backends / support_matrix / Config 保持一致。
BACKENDS = ("auto", "cuda", "corex", "ascend310p")
# auto = 不显式指定 dtype（跟随 hf config，即主项目默认行为）。
PRECISIONS = ("auto", "float16", "bfloat16", "float32")
QUANTIZATIONS = ("none", "w8a8", "quark_w8a8")
LINEAR_BACKENDS = ("functional", "selective")
GREEDY_BACKENDS = ("npu", "cpu")
PA_GRAPH_UPDATES = ("auto", "0", "1")

# precision → Config.dtype；auto 表示不传 dtype，让 Config/后端决定。
PRECISION_DTYPES = {
    "auto": None,
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """一份经过校验的统一运行设置（引擎字段 + 服务字段）。

    frozen/slots 是刻意的：配置一旦被服务或压测使用就不该在运行期被改写，
    命令行覆盖一律走 :meth:`load` 的 overrides。
    """

    # ---- 引擎字段（映射 flashengine.config.Config）----
    model: str = ""
    backend: str = "auto"
    precision: str = "auto"
    quantization: str = "none"
    fp16_only: bool = False
    max_model_len: int = 4096
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 16384
    max_num_partial_prefills: int = 1
    enable_chunked_prefill: bool = True
    enable_prefix_cache: bool = True
    enable_constraints: bool = False
    banned_words: tuple[str, ...] = DEFAULT_BANNED_WORDS
    constraint_ignore_spaces: bool = False
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enforce_eager: bool = False
    spec_model: str = ""
    spec_num_tokens: int = 4
    tensor_parallel_size: int = 1
    device_ids: tuple[int, ...] = ()
    max_pending_requests: int = 1024
    recurrent_prefix_cache_bytes: int = 256 * 1024 * 1024
    hccl_port: int = 0
    shm_name: str = ""
    memory_budget_bytes: int = 0
    device_memory_reserve_bytes: int = 0
    memory_workspace_bytes: int = 256 * 1024 * 1024
    graph_memory_reserve_bytes: int = 256 * 1024 * 1024
    # 310P 专属算子选择；非 ascend310p 后端由后端自身忽略。
    decode_linear_backend: str = "functional"
    greedy_backend: str = "npu"
    cpu_threads: int = 4
    pa_graph_update: str = "auto"
    # ---- 服务字段（只属于 HTTP/离线入口，不进引擎构造器）----
    host: str = "0.0.0.0"
    port: int = 8000
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 128
    enable_thinking: bool = False
    reserve_output: int = 512
    stats_interval: float = 10.0
    web_ui: bool = True
    api_key_env: str = "FLASHENGINE_API_KEY"

    def __post_init__(self):
        for name in ("model", "backend", "precision", "quantization", "decode_linear_backend",
                     "greedy_backend", "pa_graph_update", "spec_model", "host", "api_key_env",
                     "shm_name"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"配置 {name} 必须为字符串")
        for name in ("max_model_len", "max_num_seqs", "max_num_batched_tokens",
                     "max_num_partial_prefills", "kvcache_block_size", "cpu_threads",
                     "spec_num_tokens", "port", "max_tokens", "reserve_output",
                     "tensor_parallel_size", "max_pending_requests"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"配置 {name} 必须为正整数")
        # device_ids 之外的计数/预算类字段允许 0。
        for name in ("memory_budget_bytes", "device_memory_reserve_bytes",
                     "memory_workspace_bytes", "graph_memory_reserve_bytes",
                     "recurrent_prefix_cache_bytes", "hccl_port"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"配置 {name} 必须为非负整数")
        if type(self.num_kvcache_blocks) is not int or self.num_kvcache_blocks < -1:
            raise ValueError("配置 num_kvcache_blocks 必须为 -1（自动）或非负整数")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k 必须为非负整数")
        for name in ("fp16_only", "enable_chunked_prefill", "enable_prefix_cache",
                     "enforce_eager", "enable_thinking", "web_ui", "enable_constraints",
                     "constraint_ignore_spaces"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"配置 {name} 必须是 JSON 布尔值")
        # frozen dataclass：归一化后的词库经 object.__setattr__ 回写。
        object.__setattr__(self, "banned_words", normalize_banned_words(self.banned_words))
        resolve_constraint_words(self.enable_constraints, self.banned_words,
                                 tensor_parallel_size=self.tensor_parallel_size,
                                 spec_model=self.spec_model)
        for name in ("temperature", "top_p", "gpu_memory_utilization", "stats_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"配置 {name} 必须为有限数字")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend 必须是 {list(BACKENDS)} 之一")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision 必须是 {list(PRECISIONS)} 之一")
        if self.quantization not in QUANTIZATIONS:
            raise ValueError(f"quantization 必须是 {list(QUANTIZATIONS)} 之一")
        if self.decode_linear_backend not in LINEAR_BACKENDS:
            raise ValueError(f"decode_linear_backend 必须是 {list(LINEAR_BACKENDS)} 之一")
        if self.greedy_backend not in GREEDY_BACKENDS:
            raise ValueError(f"greedy_backend 必须是 {list(GREEDY_BACKENDS)} 之一")
        if self.pa_graph_update not in PA_GRAPH_UPDATES:
            raise ValueError(f"pa_graph_update 必须是 {list(PA_GRAPH_UPDATES)} 之一")
        if not self.host.strip():
            raise ValueError("监听地址不能为空")
        if not 1 <= self.port <= 65535:
            raise ValueError("端口必须在 1..65535 内")
        if not 0 <= self.hccl_port <= 65535:
            raise ValueError("hccl_port 必须在 0..65535 内")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size 必须在 1..8 内")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization 必须在 (0,1) 内")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.stats_interval < 0:
            raise ValueError("采样参数或统计周期越界")
        if self.max_tokens >= self.max_model_len or self.reserve_output >= self.max_model_len:
            raise ValueError("输出上限/预留输出必须小于上下文窗口")
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("关闭分块 Prefill 时，max_num_batched_tokens 必须至少覆盖 max_model_len")
        # fp16_only 是主项目新增的可选开关（nano 是「永远 FP16」）：只在显式开启时收紧。
        if self.fp16_only and self.precision not in ("auto", "float16"):
            raise ValueError("fp16_only=True 时 precision 只能为 float16 或 auto")
        if self.fp16_only and self.quantization != "none":
            raise ValueError("fp16_only=True 时 quantization 必须为 none（量化由检查点元数据判定）")
        self._validate_block_size()
        # 310P 专属开关在其他后端不生效，只告警不失败（auto 无法在此判定，不告警）。
        if self.backend in ("cuda", "corex") and (
                self.decode_linear_backend != "functional" or self.greedy_backend != "npu"):
            warnings.warn(
                "decode_linear_backend/greedy_backend 只影响 ascend310p 后端，"
                f"当前 backend={self.backend} 会忽略这些取值",
                RuntimeWarning, stacklevel=3,
            )

    def _validate_block_size(self) -> None:
        """KV 块大小按后端裁决；硬约束交给 Config/后端，这里只做前置提示。

        主项目 Config 要求 2 的幂且 >=16：这是硬约束，直接失败。
        - ascend310p：仅 64/128 可用，但后端 ``normalize_config`` 会自动调整并打印
          一行提示（与 ``scripts/benchmark.py`` 的默认路径一致），所以这里不再告警。
        - cuda：默认 flash-attn 分页内核要求 256 的倍数；``FLASHENGINE_CUDA_ATTENTION=
          flashinfer`` 允许更小页，故只告警。
        """
        size = self.kvcache_block_size
        if size < 16 or size & (size - 1):
            raise ValueError("kvcache_block_size 必须是 >=16 的 2 的幂（Config 硬约束）")
        if self.backend == "cuda" and size % 256:
            warnings.warn(
                f"cuda 默认 flash-attn 分页内核要求 kvcache_block_size 为 256 的倍数，"
                f"当前 {size}；只有 FLASHENGINE_CUDA_ATTENTION=flashinfer 支持更小页",
                RuntimeWarning, stacklevel=3,
            )

    @classmethod
    def from_mapping(cls, data):
        """从 JSON 对象构造设置；未知键、错误类型一律 ``ValueError``。"""
        if not isinstance(data, dict):
            raise ValueError("配置文件顶层必须是 JSON 对象")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"未知配置项：{sorted(unknown)}")
        values = dict(data)
        if "device_ids" in values:
            raw = values["device_ids"]
            if not isinstance(raw, (list, tuple)) or any(type(i) is not int for i in raw):
                raise ValueError("device_ids 必须是整数数组")
            values["device_ids"] = tuple(raw)
        if isinstance(values.get("pa_graph_update"), bool):
            values["pa_graph_update"] = "1" if values["pa_graph_update"] else "0"
        return cls(**values)

    @classmethod
    def load(cls, path=None, **overrides):
        """读取 UTF-8 JSON（可带 BOM），再叠加显式非 None 覆盖。"""
        data = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
        if not isinstance(data, dict):
            raise ValueError("配置文件顶层必须是 JSON 对象")
        data.update({key: value for key, value in overrides.items() if value is not None})
        return cls.from_mapping(data)

    def as_dict(self):
        data = asdict(self)
        data["device_ids"] = list(self.device_ids)
        return data

    def apply_environment(self):
        """在设备后端导入/初始化之前统一设置环境变量。

        主项目 Config 也接受同名显式字段（作为覆盖），环境变量仍是 310P 算子选择与
        图更新开关的权威入口，因此这里必须在 ``LLM(...)`` 之前调用。
        """
        os.environ["FLASHENGINE_BACKEND"] = self.backend
        os.environ["LITEQ_DECODE_LINEAR_BACKEND"] = self.decode_linear_backend
        os.environ["LITEQ_GREEDY_BACKEND"] = self.greedy_backend
        os.environ["LITEQ_CPU_THREADS"] = str(self.cpu_threads)
        os.environ["FLASHENGINE_ACLGRAPH_PA_UPDATE"] = self.pa_graph_update

    def engine_kwargs(self):
        """只返回 ``flashengine.config.Config`` 真实存在的构造参数。

        服务字段（host/port/采样/密钥环境变量）绝不进入引擎；``precision`` 映射成
        ``dtype``（auto 表示不指定）；量化方式不传参——它以检查点元数据为准，
        :func:`create_engine` 只做一致性告警。
        """
        names = (
            "model", "backend", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
            "max_num_partial_prefills", "enable_chunked_prefill", "enable_prefix_cache",
            "gpu_memory_utilization", "kvcache_block_size", "enforce_eager", "spec_model",
            "spec_num_tokens", "tensor_parallel_size", "max_pending_requests",
            "recurrent_prefix_cache_bytes", "hccl_port", "shm_name", "memory_budget_bytes",
            "device_memory_reserve_bytes", "memory_workspace_bytes",
            "graph_memory_reserve_bytes", "fp16_only", "decode_linear_backend",
            "greedy_backend", "enable_constraints", "banned_words",
            "constraint_ignore_spaces",
        )
        kwargs = {name: getattr(self, name) for name in names}
        kwargs["device_ids"] = tuple(self.device_ids)
        dtype = PRECISION_DTYPES[self.precision]
        if dtype is not None:
            kwargs["dtype"] = dtype
        if self.num_kvcache_blocks >= 0:
            kwargs["num_kvcache_blocks"] = self.num_kvcache_blocks
        return kwargs


def create_engine(settings=None):
    """按统一配置创建单个活动引擎；调用方负责 ``close()`` 或 ``with`` 生命周期。"""
    settings = settings or RuntimeSettings()
    if not settings.model:
        raise ValueError("未指定本地模型目录：请通过 --config 的 model 字段或 --model 提供")
    settings.apply_environment()
    from flashengine import LLM

    engine = LLM(**settings.engine_kwargs())
    _warn_declared_quantization(settings, engine)
    return engine


def _warn_declared_quantization(settings, engine):
    """声明的 quantization 只用于一致性检查：实际量化方式以检查点元数据为准。"""
    config = getattr(engine, "config", None)
    detected = getattr(config, "quant_method", "") or "none"
    if settings.quantization != detected:
        print(
            f"[FlashEngine] 配置声明 quantization={settings.quantization}，"
            f"检查点实际为 {detected}；以检查点为准。",
            flush=True,
        )
