"""Target-model registration and adaptation contracts.

The engine depends on :class:`TargetModelAdapter`, not concrete model classes.
Adding a target architecture therefore requires registering one adapter instead
of editing ModelRunner, the scheduler, and every speculative decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from nanovllm.utils.loader import iter_safetensors, load_model


def _identity_model_config(hf_config: Any) -> Any:
    return hf_config


def _qwen3_aux_layers(
    method: str,
    draft_config: Any,
    explicit_layers: tuple[int, ...],
    model_config: Any,
) -> tuple[int, ...]:
    """Translate draft aux-layer semantics to Qwen3 layer-input hooks."""
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

    def matches(self, hf_config: Any) -> bool:
        model_type = str(getattr(hf_config, "model_type", ""))
        architectures = set(getattr(hf_config, "architectures", None) or ())
        return model_type in self.model_types or bool(
            architectures.intersection(self.architectures)
        )

    def model_config(self, hf_config: Any) -> Any:
        """Return the config consumed by the text target model.

        The default is suitable for dense decoder-only models. Future adapters
        may override this for nested text configs or multimodal containers.
        """
        return self.model_config_selector(hf_config)

    def build(self, hf_config: Any):
        cls = getattr(import_module(self.module), self.class_name)
        return cls(hf_config)

    def load_weights(self, model, model_path: str) -> None:
        if hasattr(model, "load_weights"):
            model.load_weights(iter_safetensors(model_path))
        else:
            load_model(model, model_path)

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
        """Translate a spec checkpoint's aux-layer contract to model hooks.

        Qwen3 captures layer *inputs*. DFlash/DSpark checkpoints describe layer
        outputs, so output of layer ``l`` maps to input of layer ``l + 1``.
        EAGLE3 checkpoints use the established three layer-input capture points.
        """
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


register_target_model(
    TargetModelAdapter(
        name="qwen3",
        model_types=("qwen3",),
        architectures=("Qwen3ForCausalLM",),
        module="nanovllm.models.qwen3",
        class_name="Qwen3ForCausalLM",
        supported_spec_methods=frozenset({"eagle3", "dflash", "dspark"}),
        model_config_selector=_identity_model_config,
        aux_layer_resolver=_qwen3_aux_layers,
    )
)
