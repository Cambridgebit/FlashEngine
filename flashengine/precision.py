"""统一 FP16 发布入口的检查点精度守卫。

只做元数据检查（纯 `json`/`pathlib`），**import 期不依赖 torch**，因此脚本与
host 侧测试可以脱离设备运行时直接调用。

是否拒绝量化检查点由 `Config.fp16_only` 决定（默认关闭）：主项目保留 W8A8
（msmodelslim）与 Quark int8 加载期反量化，只有显式开启 fp16_only 时才在此
干净拒绝量化/反量化来源的检查点，避免"半量化权重 + FP16 计算参数"混跑。
"""

import json
from pathlib import Path

# msmodelslim 导出会在模型目录留下逐权重的量化方案表；存在即视为量化检查点。
QUANT_METADATA_FILE = "quant_model_description.json"

# config.json（或 HF config 对象）里可能出现的量化/压缩配置键。Quark 的
# quantization_config.quant_method == "quark" 与其它导出的同类字段都在此覆盖。
QUANT_CONFIG_KEYS = (
    "quantization_config",
    "quant_config",
    "compression_config",
    "quant_scheme",
)


def _config_metadata(root: Path, config) -> dict:
    """把调用方传入的 config 归一化成 dict；None 表示读目录下的 config.json。

    接受 HF config 对象（`vars` 取字段）、dict 或 None。目录里没有 config.json
    时返回空表——只有 config.json 的量化键才算证据，不因缺文件而误判。
    """
    if config is not None:
        if isinstance(config, dict):
            return config
        data = getattr(config, "to_dict", None)
        if callable(data):
            try:
                mapping = data()
            except (AttributeError, TypeError, ValueError):
                mapping = None
            if isinstance(mapping, dict):
                return mapping
        try:
            mapping = vars(config)
        except TypeError:
            mapping = None
        return mapping if isinstance(mapping, dict) else {}
    config_path = root / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def quantized_metadata_reason(path, config=None) -> str | None:
    """返回量化痕迹的说明串；未发现痕迹返回 None。

    拆成独立函数便于调用方在报错/日志里给出具体原因（哪个键、哪个文件），
    而 `is_quantized_checkpoint` / `require_unquantized_checkpoint` 只做布尔
    判定与抛异常。
    """
    root = Path(path)
    if (root / QUANT_METADATA_FILE).exists():
        return f"存在量化方案表 {QUANT_METADATA_FILE}"
    data = _config_metadata(root, config)
    for key in QUANT_CONFIG_KEYS:
        value = data.get(key)
        if value:
            return f"config 含 {key}"
    return None


def is_quantized_checkpoint(path, config=None) -> bool:
    """检查点是否带量化/反量化痕迹（不抛异常，供预检与日志使用）。"""
    return quantized_metadata_reason(path, config) is not None


def require_unquantized_checkpoint(path, config=None) -> Path:
    """允许原始 FP16/BF16/FP32 权重，拒绝量化及反量化检查点。

    仅由 `Config.fp16_only` 为真时调用；路径不存在直接抛 FileNotFoundError，
    发现量化痕迹抛 ValueError（带具体原因，便于部署侧定位）。
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"检查点目录不存在：{root}")
    reason = quantized_metadata_reason(root, config)
    if reason is not None:
        raise ValueError(f"FP16 统一版禁止加载量化检查点：{root}（{reason}）")
    return root
