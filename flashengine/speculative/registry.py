
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


def _noop_normalizer(draft_config: Any) -> None:
    return None


def _identity_num_tokens(draft_config: Any, requested: int) -> int:
    return requested


def _normalize_block_draft(draft_config: Any) -> None:
    dflash_config = getattr(draft_config, "dflash_config", None) or {}
    if not dflash_config:
        dflash_config = {
            "block_size": getattr(draft_config, "block_size"),
            "mask_token_id": getattr(draft_config, "mask_token_id"),
            "target_layer_ids": getattr(draft_config, "target_layer_ids"),
        }
        draft_config.dflash_config = dflash_config


def _normalize_dspark(draft_config: Any) -> None:
    _normalize_block_draft(draft_config)
    if getattr(draft_config, "markov_head_type", "vanilla") != "vanilla":
        raise NotImplementedError("Only vanilla DSpark Markov heads are supported.")
    draft_vocab = getattr(draft_config, "draft_vocab_size", None)
    if (draft_vocab or draft_config.vocab_size) != draft_config.vocab_size:
        raise NotImplementedError(
            "DSpark reduced-vocabulary checkpoints are not supported."
        )


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
