
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


def _noop_normalizer(draft_config: Any) -> None:
    return None


def _identity_num_tokens(draft_config: Any, requested: int) -> int:
    return requested


def block_draft_attention(config: Any) -> tuple[tuple[bool, int | None], ...]:
    """逐层解析官方 DFlash 因果/窗口语义；窗口包含当前位置。"""
    count = int(config.num_hidden_layers)
    kinds = getattr(config, "layer_types", None) or ["full_attention"] * count
    if len(kinds) != count or set(kinds) - {"full_attention", "sliding_attention"}:
        raise ValueError("DFlash layer_types 必须逐层指定 full_attention/sliding_attention")
    override = getattr(config, "is_causal", None)
    result = []
    for kind in kinds:
        sliding = kind == "sliding_attention"
        window = getattr(config, "sliding_window", None) if sliding else None
        if sliding and (not isinstance(window, int) or isinstance(window, bool) or window < 1):
            raise ValueError("DFlash sliding_attention 需要正整数 sliding_window")
        result.append((sliding if override is None else bool(override), window))
    return tuple(result)


def _normalize_block_draft(draft_config: Any) -> None:
    dflash_config = getattr(draft_config, "dflash_config", None) or {}
    if not dflash_config:
        target_layer_ids = getattr(draft_config, "target_layer_ids", None)
        if target_layer_ids is None:
            aux_layer_ids = getattr(draft_config, "aux_hidden_state_layer_ids", None)
            if aux_layer_ids:
                # speculators 格式：aux_hidden_state_layer_ids 是 vLLM/eagle 口径的
                # 捕获层号；DFlash 主体的 target_layer_ids 比它小一（对齐 vLLM 对
                # speculators 的转换），本仓按 Qwen3 做 +1 换算后正好还原捕获层号。
                target_layer_ids = [int(i) - 1 for i in aux_layer_ids]
        dflash_config = {
            "block_size": getattr(draft_config, "block_size", None),
            "mask_token_id": getattr(draft_config, "mask_token_id", None),
            "target_layer_ids": target_layer_ids,
        }
        draft_config.dflash_config = dflash_config
    if dflash_config.get("target_layer_ids") is None:
        raise ValueError(
            "Block-draft config must provide target_layer_ids or "
            "aux_hidden_state_layer_ids."
        )
    draft_config.block_draft_requires_mask = any(
        causal or window is not None for causal, window in block_draft_attention(draft_config)
    )


def _normalize_dspark(draft_config: Any) -> None:
    _normalize_block_draft(draft_config)
    if getattr(draft_config, "markov_head_type", "vanilla") != "vanilla":
        raise NotImplementedError("Only vanilla DSpark Markov heads are supported.")
    draft_vocab = getattr(draft_config, "draft_vocab_size", None)
    if (draft_vocab or draft_config.vocab_size) != draft_config.vocab_size:
        raise NotImplementedError(
            "DSpark reduced-vocabulary checkpoints are not supported."
        )


def load_draft_config(path: str) -> Any:
    """加载草稿模型配置：标准 HF 自动解析优先，回退 speculators 平铺格式。

    vllm-project/speculators 导出的 DFlash 草稿没有 ``model_type``，
    ``AutoConfig`` 无法识别；这里按官方转换口径（vLLM 对 speculators 的处理）
    从 ``transformer_layer_config`` 重建配置，并透传顶层字段供归一化使用。
    """
    import json
    import os

    from transformers import AutoConfig, PretrainedConfig

    try:
        return AutoConfig.from_pretrained(path)
    except Exception:
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise
        with open(config_path, encoding="utf-8") as handle:
            data = json.load(handle)
        layer_config = data.get("transformer_layer_config")
        if not layer_config or not (
            data.get("speculators_model_type")
            or data.get("aux_hidden_state_layer_ids")
        ):
            raise
        config = PretrainedConfig(**layer_config)
        config.architectures = list(
            data.get("architectures") or ["DFlashDraftModel"])
        config.block_size = data.get("block_size")
        config.mask_token_id = data.get("mask_token_id")
        config.draft_vocab_size = data.get("draft_vocab_size")
        config.aux_hidden_state_layer_ids = data.get("aux_hidden_state_layer_ids")
        config.speculators_model_type = data.get("speculators_model_type")
        return config


def _block_num_tokens(draft_config: Any, requested: int) -> int:
    dflash_config = getattr(draft_config, "dflash_config", None) or {}
    block_size = int(
        dflash_config.get("block_size", getattr(draft_config, "block_size", 16))
    )
    maximum = block_size - 1
    value = maximum if requested in (3, 4) else requested
    if not 1 <= value <= maximum:
        raise ValueError(
            f"Block-draft token count must be in [1, {maximum}] for "
            f"draft block_size={block_size}; got {value}."
        )
    return value


def _dspark_num_tokens(draft_config: Any, requested: int) -> int:
    dflash_config = getattr(draft_config, "dflash_config", None) or {}
    block_size = int(
        dflash_config.get("block_size", getattr(draft_config, "block_size", 16))
    )
    maximum = (
        block_size
        if getattr(draft_config, "sample_from_anchor", True)
        else block_size - 1
    )
    value = maximum if requested in (3, 4) else requested
    if not 1 <= value <= maximum:
        raise ValueError(
            f"DSpark draft token count must be in [1, {maximum}] for "
            f"draft block_size={block_size}; got {value}."
        )
    return value


@dataclass(frozen=True, slots=True)
class SpecMethodAdapter:
    name: str
    architectures: tuple[str, ...]
    block_draft: bool
    module: str
    class_name: str
    loader_name: str
    normalizer: Any
    num_tokens_resolver: Any
    can_sample_from_anchor: bool = False

    def build_runtime(self, runner):
        runtime = import_module("flashengine.speculative.runtime")
        cls = runtime.BlockDraftRuntime if self.block_draft else runtime.SpecRuntime
        return cls(runner)

    def matches(self, draft_config: Any) -> bool:
        architectures = tuple(
            str(a) for a in (getattr(draft_config, "architectures", None) or ())
        )
        return any(
            marker in architecture
            for marker in self.architectures
            for architecture in architectures
        )

    def normalize_config(self, draft_config: Any) -> None:
        self.normalizer(draft_config)

    def configure_num_tokens(self, draft_config: Any, requested: int) -> int:
        return self.num_tokens_resolver(draft_config, requested)

    def build_draft(self, config, target_model):
        module = import_module(self.module)
        cls = getattr(module, self.class_name)
        loader = getattr(module, self.loader_name)
        draft_config = config.draft_config
        if self.block_draft:
            model = cls(
                draft_config,
                target_model.model.embed_tokens,
                target_lm_head_weight=target_model.lm_head.weight,
                max_position=config.max_model_len,
            )
        else:
            model = cls(draft_config, target_model.model.embed_tokens)
        loader(model, config.spec_model)
        return model


_SPEC_METHODS: dict[str, SpecMethodAdapter] = {}


def register_spec_method(adapter: SpecMethodAdapter) -> SpecMethodAdapter:
    if adapter.name in _SPEC_METHODS:
        raise ValueError(f"Spec method {adapter.name!r} is already registered")
    _SPEC_METHODS[adapter.name] = adapter
    return adapter


def resolve_spec_method(draft_config: Any) -> SpecMethodAdapter:
    architectures = tuple(
        str(a) for a in (getattr(draft_config, "architectures", None) or ())
    )
    if any("DFlash2" in architecture for architecture in architectures):
        raise NotImplementedError(
            "DFlash2 support was removed; use an EAGLE3, DFlash, or DSpark "
            "checkpoint."
        )
    for name in ("dspark", "dflash"):
        adapter = _SPEC_METHODS[name]
        if adapter.matches(draft_config):
            adapter.normalize_config(draft_config)
            return adapter
    if getattr(draft_config, "dflash_config", None):
        adapter = _SPEC_METHODS["dflash"]
        adapter.normalize_config(draft_config)
        return adapter
    adapter = _SPEC_METHODS["eagle3"]
    adapter.normalize_config(draft_config)
    return adapter


def get_spec_method(name: str) -> SpecMethodAdapter:
    return _SPEC_METHODS[name]


def block_draft_methods() -> frozenset[str]:
    return frozenset(
        name for name, adapter in _SPEC_METHODS.items() if adapter.block_draft
    )


def registered_spec_methods() -> tuple[str, ...]:
    return tuple(_SPEC_METHODS)


register_spec_method(
    SpecMethodAdapter(
        name="eagle3",
        architectures=("Eagle", "EAGLE"),
        block_draft=False,
        module="flashengine.models.eagle3",
        class_name="Eagle3DraftModel",
        loader_name="load_eagle3_draft",
        normalizer=_noop_normalizer,
        num_tokens_resolver=_identity_num_tokens,
    )
)
register_spec_method(
    SpecMethodAdapter(
        name="dflash",
        architectures=("DFlash",),
        block_draft=True,
        module="flashengine.models.dflash",
        class_name="DFlashDraftModel",
        loader_name="load_dflash_draft",
        normalizer=_normalize_block_draft,
        num_tokens_resolver=_block_num_tokens,
    )
)
register_spec_method(
    SpecMethodAdapter(
        name="dspark",
        architectures=("DSpark",),
        block_draft=True,
        module="flashengine.models.dspark",
        class_name="DSparkDraftModel",
        loader_name="load_dspark_draft",
        normalizer=_normalize_dspark,
        num_tokens_resolver=_dspark_num_tokens,
        can_sample_from_anchor=True,
    )
)
