"""Qwen3 DSpark: DFlash block backbone plus a sequential Markov head.

The DeepSeek checkpoint uses a vanilla low-rank Markov transition and may
contain its own embedding/output head. Confidence weights are loaded for
checkpoint compatibility; adaptive verification is not enabled here because
the current verifier uses a fixed, graph-captured block width.
"""

import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open

from nanovllm.models.dflash import DFlashDraftModel
from nanovllm.utils.loader import default_weight_loader


class _MarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def bias(self, previous_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(
            self.markov_w1(previous_ids),
            getattr(self.markov_w2, "weight_nz", self.markov_w2.weight))


class _ConfidenceHead(nn.Module):
    def __init__(self, input_size: int, with_markov: bool = True):
        super().__init__()
        self.with_markov = with_markov
        self.proj = nn.Linear(input_size, 1, bias=True, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor,
                markov_embed: torch.Tensor | None = None) -> torch.Tensor:
        if self.with_markov:
            if markov_embed is None:
                raise ValueError("Markov embedding is required by this confidence head")
            hidden = torch.cat([hidden, markov_embed], dim=-1)
        return torch.sigmoid(self.proj(hidden.float())).squeeze(-1)


class DSparkDraftModel(DFlashDraftModel):
    def __init__(self, config, target_embed_tokens, target_lm_head_weight,
                 max_position=None):
        if getattr(config, "markov_head_type", "vanilla") != "vanilla":
            raise NotImplementedError("Only vanilla DSpark Markov heads are supported")
        super().__init__(config, target_embed_tokens, target_lm_head_weight,
                         max_position=max_position)
        rank = int(config.markov_rank)
        if rank <= 0:
            raise ValueError("DSpark markov_rank must be positive")
        self.markov_head = _MarkovHead(config.vocab_size, rank)
        self.confidence_head = None
        if getattr(config, "enable_confidence_head", False):
            with_markov = getattr(config, "confidence_head_with_markov", False)
            self.confidence_head = _ConfidenceHead(
                config.hidden_size + (rank if with_markov else 0),
                with_markov=with_markov)

    def markov_bias(self, previous_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.bias(previous_ids)


def load_dspark_draft(model: DSparkDraftModel, path: str) -> None:
    """Load full-vocabulary vanilla DSpark weights, including optional heads."""
    bin_path = os.path.join(path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu", weights_only=True)

        def items():
            yield from state.items()
    else:
        files = sorted(glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No DSpark weights found in {path}")

        def items():
            for file in files:
                with safe_open(file, framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        yield name, handle.get_tensor(name)

    loaded = set()
    for source_name, tensor in items():
        name = source_name.removeprefix("model.")
        if name.startswith("t2d") or name.startswith("d2t") or name.startswith("mask_embedding"):
            continue
        if name == "embed_tokens.weight":
            # Never copy a draft embedding into the shared target module.
            if model.embed_tokens.weight.shape != tensor.shape:
                raise ValueError("DSpark embedding size differs from the target vocabulary")
            model.embed_tokens = nn.Embedding(
                tensor.shape[0], tensor.shape[1],
                device=model.fc.weight.device, dtype=model.fc.weight.dtype)
        if name == "lm_head.weight":
            if model.lm_head_weight.shape != tensor.shape:
                raise ValueError("DSpark LM-head size differs from the target vocabulary")
            model.lm_head_weight = nn.Parameter(torch.empty(
                tensor.shape, device=model.fc.weight.device,
                dtype=model.fc.weight.dtype))
            name = "lm_head_weight"
        shard_id = None
        for source, (packed, shard) in model.packed_modules_mapping.items():
            if source in name:
                name = name.replace(source, packed)
                shard_id = shard
                break
        try:
            param = model.get_parameter(name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported DSpark checkpoint weight: {source_name}") from exc
        loader = getattr(param, "weight_loader", default_weight_loader)
        if shard_id is not None:
            loader(param, tensor, shard_id)
        else:
            loader(param, tensor)
        loaded.add(name)
    for name in ("markov_head.markov_w1.weight", "markov_head.markov_w2.weight"):
        if name not in loaded:
            raise ValueError(f"DSpark checkpoint is missing {name}")
    if model.confidence_head is not None and "confidence_head.proj.weight" not in loaded:
        raise ValueError("DSpark checkpoint is missing its confidence head")
