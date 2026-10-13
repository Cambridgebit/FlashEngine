__all__ = ["LLM", "LLMEngine", "SamplingParams"]


def __getattr__(name):
    if name == "SamplingParams":
        from flashengine.sampling_params import SamplingParams
        return SamplingParams
    if name in ("LLM", "LLMEngine"):
        from flashengine.llm import LLM
        from flashengine.engine.llm_engine import LLMEngine
        return LLM if name == "LLM" else LLMEngine
    raise AttributeError(name)
