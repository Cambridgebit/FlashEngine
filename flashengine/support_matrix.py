"""FlashEngine 支持矩阵：目标模型 × 后端 × 量化 × 投机方法的声明式支持表。

这是「哪些组合被声明支持」的单一事实来源：
- 运行时：`Config` 在解析完 target_adapter / backend / 量化 / spec_adapter 后调用
  `require_supported(...)`，不支持组合在此干净拒绝（NotImplementedError）。
- 文档：`format_matrix()` 枚举全部组合，`docs/SUPPORT_MATRIX.md` 与本模块保持一致。
- 测试：`tests/test_support_matrix.py` 交叉校验本表与已注册 adapter 的能力位。

注意：本表是「声明支持」；运行时能力（310P 缺 SplitFuse、图捕获失败、草稿缺 aux 层等）
可能进一步收紧并优雅降级，见各 backend 的 normalize_config/validate_config 与捕获回退。

矩阵（当前）：
                    普通   EAGLE3   DFlash   DSpark   W8A8量化
  Qwen3  dense      ✅      ✅       ✅       ✅       仅 310P
  Qwen3  MoE     cuda/corex/310P ❌ cuda/corex/310P ❌  仅 310P
  后端：cuda / corex / ascend310p（MoE = 三后端；W8A8 仅 ascend310p）
  另加 Quark W8A8 int8 检查点（cuda/corex：加载期权重反量化，见 quant=quark_w8a8）
"""

from __future__ import annotations

from dataclasses import dataclass

TARGETS = ("qwen3", "qwen3_moe")
BACKENDS = ("cuda", "corex", "ascend310p")
QUANTIZATIONS = ("none", "w8a8", "quark_w8a8")
SPEC_METHODS = ("none", "eagle3", "dflash", "dspark")

# 静态视图（供文档/测试；运行时以 backend 能力位覆盖）。MoE = cuda/corex（纯 torch
# 专家前向）与 ascend310p（npu grouped matmul）；W8A8（msmodelslim int8 算子）仅
# ascend310p；quark_w8a8（AMD Quark int8 检查点，加载期反量化）仅 cuda/corex。
_MOE_BACKENDS = frozenset({"cuda", "corex", "ascend310p"})
_W8A8_BACKENDS = frozenset({"ascend310p"})
_QUARK_W8A8_BACKENDS = frozenset({"cuda", "corex"})
# 各目标的投机方法集（none = 普通推理）。本表是单一事实来源；models/registry.py 的
# adapter 直接由此派生 supported_spec_methods，杜绝两处口径漂移（本模块不 import
# 引擎运行时，故测试可脱离 torch 运行）。
_TARGET_SPEC_METHODS = {
    "qwen3": frozenset({"none", "eagle3", "dflash", "dspark"}),
    "qwen3_moe": frozenset({"none", "dflash"}),
}


@dataclass(frozen=True)
class SupportDecision:
    supported: bool
    reason: str = ""


def evaluate(
    target: str,
    backend: str,
    quantization: str = "none",
    spec_method: str = "none",
    *,
    backend_supports_moe: bool | None = None,
    backend_supports_w8a8: bool | None = None,
    backend_supports_quark_w8a8: bool | None = None,
) -> SupportDecision:
    """判定 (目标, 后端, 量化, 投机方法) 组合是否被声明支持。

    backend_supports_* 为运行时后端能力位（None = 用静态视图，供文档/测试；引擎运行时
    由 Config 传入真实 backend 能力位）。
    """
    if target not in _TARGET_SPEC_METHODS:
        return SupportDecision(False, f"未注册的目标模型 {target!r}")
    if backend not in BACKENDS:
        return SupportDecision(False, f"未知后端 {backend!r}")
    if quantization not in QUANTIZATIONS:
        return SupportDecision(False, f"未知量化方式 {quantization!r}")
    if spec_method not in SPEC_METHODS:
        return SupportDecision(False, f"未知投机方法 {spec_method!r}")
    supports_moe = (
        backend in _MOE_BACKENDS if backend_supports_moe is None else backend_supports_moe
    )
    supports_w8a8 = (
        backend in _W8A8_BACKENDS if backend_supports_w8a8 is None else backend_supports_w8a8
    )
    supports_quark = (
        backend in _QUARK_W8A8_BACKENDS
        if backend_supports_quark_w8a8 is None else backend_supports_quark_w8a8
    )
    if target == "qwen3_moe" and not supports_moe:
        return SupportDecision(
            False, f"qwen3_moe 需要支持 MoE 的后端（当前 {backend}）"
        )
    if quantization == "w8a8" and not supports_w8a8:
        return SupportDecision(
            False,
            f"W8A8 量化仅支持后端 {sorted(_W8A8_BACKENDS)}（当前 {backend}）",
        )
    if quantization == "quark_w8a8" and not supports_quark:
        return SupportDecision(
            False,
            f"Quark W8A8 量化仅支持后端 {sorted(_QUARK_W8A8_BACKENDS)}（当前 {backend}）",
        )
    allowed_spec = _TARGET_SPEC_METHODS[target]
    if spec_method not in allowed_spec:
        supported = sorted(allowed_spec - {"none"}) or ["none"]
        return SupportDecision(
            False, f"{target} 不支持投机方法 {spec_method!r}（支持：{supported}）"
        )
    return SupportDecision(True)


def require_supported(
    target: str,
    backend: str,
    quantization: str = "none",
    spec_method: str = "none",
    *,
    backend_impl=None,
) -> None:
    decision = evaluate(
        target,
        backend,
        quantization,
        spec_method,
        backend_supports_moe=getattr(backend_impl, "supports_moe", None),
        backend_supports_w8a8=getattr(backend_impl, "supports_w8a8", None),
        backend_supports_quark_w8a8=getattr(
            backend_impl, "supports_int8_dequant", None),
    )
    if not decision.supported:
        raise NotImplementedError(
            "Unsupported combination: "
            f"target={target}, backend={backend}, quantization={quantization}, "
            f"spec_method={spec_method}. {decision.reason}"
        )


def format_matrix() -> str:
    """枚举全部组合生成 Markdown 表格（文档/测试用）。"""
    lines = [
        "| target | backend | quantization | spec_method | supported |",
        "|---|---|---|---|---|",
    ]
    for target in TARGETS:
        for backend in BACKENDS:
            for quantization in QUANTIZATIONS:
                for spec_method in SPEC_METHODS:
                    decision = evaluate(target, backend, quantization, spec_method)
                    lines.append(
                        f"| {target} | {backend} | {quantization} | {spec_method} "
                        f"| {'yes' if decision.supported else 'no'} |"
                    )
    return "\n".join(lines)


def supported_spec_methods(target: str) -> frozenset[str]:
    return _TARGET_SPEC_METHODS.get(target, frozenset())
