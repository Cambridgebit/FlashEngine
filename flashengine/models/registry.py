
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from flashengine.support_matrix import supported_spec_methods
from flashengine.utils.loader import (
    iter_safetensors,
    iter_safetensors_dequant,
    iter_with_progress,
    load_model,
    safetensors_total_bytes,
)


def _identity_model_config(hf_config: Any) -> Any:
    return hf_config


def _qwen3_aux_layers(
    method: str,
    draft_config: Any,
    explicit_layers: tuple[int, ...],
    model_config: Any,
) -> tuple[int, ...]:
    if explicit_layers:
        return tuple(explicit_layers)
    if method in ("dflash", "dspark"):
        dflash_config = getattr(draft_config, "dflash_config", None) or {}
        return tuple(int(i) + 1 for i in dflash_config["target_layer_ids"])
    n = int(model_config.num_hidden_layers)
    return (2, n // 2, n - 3)


@dataclass(frozen=True, slots=True)
class TargetModelAdapter:
    name: str
    model_types: tuple[str, ...]
    architectures: tuple[str, ...]
    module: str
    class_name: str
    supported_spec_methods: frozenset[str]
    model_config_selector: Any
    aux_layer_resolver: Any
    # MoE（Qwen3-30B-A3B 系）标记：引擎据此走 MoE 元数据/门控与专家权重后处理。
    is_moe: bool = False

    def matches(self, hf_config: Any) -> bool:
        model_type = str(getattr(hf_config, "model_type", ""))
        architectures = set(getattr(hf_config, "architectures", None) or ())
        return model_type in self.model_types or bool(
            architectures.intersection(self.architectures)
        )

    def model_config(self, hf_config: Any) -> Any:
        return self.model_config_selector(hf_config)

    def build(self, hf_config: Any):
        cls = getattr(import_module(self.module), self.class_name)
        return cls(hf_config)

    def load_weights(self, model, model_path: str, dequant: bool = False,
                     dtype=None) -> None:
        # dequant=True（Quark int8 检查点）：加载期把 int8 权重连带 scale/zero_point
        # 反量化成计算 dtype，之后前向与普通 bf16 模型完全一致。
        weights = (
            iter_safetensors_dequant(model_path, dtype)
            if dequant else iter_safetensors(model_path)
        )
        # 加载期反量化会把 int8 展开成 2 倍字节，文件头总量对不上，故不传总量。
        total_bytes = None if dequant else safetensors_total_bytes(model_path)
        weights = iter_with_progress(weights, "Target model loading", total_bytes)
        if hasattr(model, "load_weights"):
            model.load_weights(weights)
        else:
            load_model(model, model_path, weights=weights)

    def validate_spec_method(self, method: str) -> None:
        if method not in self.supported_spec_methods:
            supported = ", ".join(sorted(self.supported_spec_methods)) or "none"
            raise NotImplementedError(
                f"Target model {self.name} does not support speculative method "
                f"{method!r}; supported methods: {supported}."
            )

    def resolve_aux_layers(
        self,
        method: str,
        draft_config: Any,
        explicit_layers: tuple[int, ...],
        model_config: Any,
    ) -> tuple[int, ...]:
        return self.aux_layer_resolver(
            method, draft_config, explicit_layers, model_config
        )


_TARGET_ADAPTERS: list[TargetModelAdapter] = []


def register_target_model(adapter: TargetModelAdapter) -> TargetModelAdapter:
    if any(existing.name == adapter.name for existing in _TARGET_ADAPTERS):
        raise ValueError(f"Target model adapter {adapter.name!r} is already registered")
    _TARGET_ADAPTERS.append(adapter)
    return adapter


def resolve_target_model(hf_config: Any) -> TargetModelAdapter:
    for adapter in _TARGET_ADAPTERS:
        if adapter.matches(hf_config):
            return adapter
    model_type = getattr(hf_config, "model_type", None)
    architectures = getattr(hf_config, "architectures", None)
    supported = ", ".join(adapter.name for adapter in _TARGET_ADAPTERS)
    raise NotImplementedError(
        "Unsupported target model architecture: "
        f"model_type={model_type!r}, architectures={architectures!r}. "
        f"Registered targets: {supported}."
    )


def registered_target_models() -> tuple[str, ...]:
    return tuple(adapter.name for adapter in _TARGET_ADAPTERS)


def target_adapters() -> tuple[TargetModelAdapter, ...]:
    return tuple(_TARGET_ADAPTERS)


register_target_model(
    TargetModelAdapter(
        name="qwen3",
        model_types=("qwen3",),
        architectures=("Qwen3ForCausalLM",),
        module="flashengine.models.qwen3",
        class_name="Qwen3ForCausalLM",
        supported_spec_methods=supported_spec_methods("qwen3") - {"none"},
        model_config_selector=_identity_model_config,
        aux_layer_resolver=_qwen3_aux_layers,
    )
)

register_target_model(
    TargetModelAdapter(
        name="qwen3_moe",
        model_types=("qwen3_moe",),
        architectures=("Qwen3MoeForCausalLM",),
        module="flashengine.models.qwen3_moe",
        class_name="Qwen3MoeForCausalLM",
        # MoE 目标当前仅 DFlash 草稿有对应路径（EAGLE3/DSpark 无 MoE 草稿；
        # MoE 与草稿均为 dense 残差流，aux 口径与 dense Qwen3 相同 → +1 换算）。
        supported_spec_methods=supported_spec_methods("qwen3_moe") - {"none"},
        model_config_selector=_identity_model_config,
        aux_layer_resolver=_qwen3_aux_layers,
        is_moe=True,
    )
)
