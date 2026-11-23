__all__ = [
    "LLM",
    "LLMEngine",
    "SamplingParams",
    "RuntimeSettings",
    "create_engine",
    "__version__",
]

# 必须发生在任何 transformers 导入之前（见 flashengine/npu_bootstrap.py）：
# Ascend 上 torch_npu 的首次 _npu_init 若晚于 transformers 导入，会因
# ACL_ALLOW_HF32 compileopt 返回 500001 而失败。非 Ascend 机器上是 no-op。
from flashengine.npu_bootstrap import ensure_npu_device_before_transformers

ensure_npu_device_before_transformers()


def __getattr__(name):
    # 全部惰性：导入 flashengine 不应拉入 torch / transformers / 服务端模块。
    if name in ("RuntimeSettings", "create_engine"):
        from flashengine.runtime import RuntimeSettings, create_engine
        return RuntimeSettings if name == "RuntimeSettings" else create_engine
    if name == "__version__":
        # 发布身份来自 flashengine.version（统一入口模块负责维护）；该文件尚未
        # 落地时退回占位版本，避免 import flashengine 或取版本号被阻塞。
        try:
            from flashengine.version import VERSION
        except ImportError:
            VERSION = "0.0.0"
        globals()["__version__"] = VERSION
        return VERSION
    if name == "SamplingParams":
        from flashengine.sampling_params import SamplingParams
        return SamplingParams
    if name in ("LLM", "LLMEngine"):
        from flashengine.llm import LLM
        from flashengine.engine.llm_engine import LLMEngine
        return LLM if name == "LLM" else LLMEngine
    raise AttributeError(name)
