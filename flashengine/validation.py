"""FP16 数值/参数精度检查（验收与图初始验证用，依赖 torch）。

三处消费者：
- `backends/ascend.py` 的 `validate_model_decode`：functional vs selective 的
  单行 decode logits 对比；
- `backends/ascend310p/graph.py`：捕获后回放自检与每桶前两次真实输入对比；
- `engine/model_runner.py`：`fp16_only` 启动时的参数 dtype 审计。

默认（`Config.fp16_only` 为假）不做任何审计，主项目行为不变。
"""

import torch


def compare_logits(reference, candidate, atol=0.005, rtol=0.003) -> dict:
    """比较两份 logits 的形状/有限性/误差/top-1；不满足门禁抛 RuntimeError。

    容限口径 `|ref - got| <= atol + rtol * |ref|`；`bad` 为越界元素数。
    形状不一致抛 ValueError（配置/接线错误，不是数值问题）。
    """
    ref = reference.detach().to(device="cpu", dtype=torch.float32)
    got = candidate.detach().to(device="cpu", dtype=torch.float32)
    if ref.shape != got.shape:
        raise ValueError(
            f"参考与候选 logits 形状不一致：{tuple(ref.shape)} vs {tuple(got.shape)}"
        )
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(got).all())
    difference = (ref - got).abs()
    # 非有限值无法比较误差，直接把全部元素记为 bad，让调用方走回退路径。
    bad = int((difference > atol + rtol * ref.abs()).sum()) if finite else ref.numel()
    top1_equal = finite and torch.equal(ref.argmax(dim=-1), got.argmax(dim=-1))
    result = {
        "finite": finite,
        "bad": bad,
        "top1_equal": bool(top1_equal),
        "max_abs": float(difference.max()) if finite else None,
        "atol": atol,
        "rtol": rtol,
        "elements": ref.numel(),
    }
    if not finite or bad or not top1_equal:
        raise RuntimeError(f"FP16 logits 数值检查失败：{result}")
    return result


def _iter_parameters(models):
    for model in models:
        yield from model.named_parameters()


def audit_fp16_parameters(*models) -> dict:
    """要求参与计算的全部参数为 FP16；不合法参数抛 RuntimeError。

    只审浮点参数：整数/布尔参数（token 映射等非计算张量）跳过并单独计数，
    不因为它们不是 FP16 就判失败。
    """
    checked = 0
    skipped = 0
    for name, param in _iter_parameters(models):
        if not param.is_floating_point():
            skipped += 1
            continue
        if param.dtype != torch.float16:
            raise RuntimeError(f"计算参数必须是 FP16：{name}={param.dtype}")
        checked += 1
    return {
        "checked": True,
        "runtime_dtype": "float16",
        "quantization": "none",
        "parameters_checked": checked,
        "parameters_skipped": skipped,
    }


def audit_parameter_dtypes(*models) -> dict:
    """只统计参数 dtype 分布并给出告警，不抛异常（非 fp16_only 路径用）。

    用于启动摘要/运行快照：量化模型（int8 权重 + fp16 尺度）也会出现多种
    dtype，因此这里只报"与目标计算类型不一致"的浮点参数，交由调用方决定是
    否打印告警。
    """
    counts: dict[str, int] = {}
    warnings: list[str] = []
    for name, param in _iter_parameters(models):
        key = str(param.dtype)
        counts[key] = counts.get(key, 0) + 1
        if param.is_floating_point() and param.dtype != torch.float16:
            warnings.append(f"{name}={param.dtype}")
    return {
        "dtypes": counts,
        "warnings": warnings,
        "warning_count": len(warnings),
        "checked": True,
    }
