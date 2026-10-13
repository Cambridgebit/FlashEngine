
import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open

from flashengine.models.dflash import DFlashDraftModel
from flashengine.utils.loader import default_weight_loader


class _MarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int, device=None):
        super().__init__()
        # 310P: build on CPU (default init uses AI-CPU random ops, broken on old
        # drivers), then move. Weights are overwritten when the checkpoint loads.
        with torch.device("cpu"):
            self.markov_w1 = nn.Embedding(vocab_size, rank)
            self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)
        if device is not None:
            self.to(device)

    def bias(self, previous_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(
            self.markov_w1(previous_ids),
            getattr(self.markov_w2, "weight_nz", self.markov_w2.weight))


class _ConfidenceHead(nn.Module):
    def __init__(self, input_size: int, with_markov: bool = True, device=None):
        super().__init__()
        self.with_markov = with_markov
        with torch.device("cpu"):
            self.proj = nn.Linear(input_size, 1, bias=True, dtype=torch.float32)
        if device is not None:
            self.to(device)

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
        self.sample_from_anchor = getattr(config, "sample_from_anchor", True)
        rank = int(config.markov_rank)
        if rank <= 0:
            raise ValueError("DSpark markov_rank must be positive")
        self.markov_head = _MarkovHead(config.vocab_size, rank, device=self.fc.weight.device)
        self.confidence_head = None
        if getattr(config, "enable_confidence_head", False):
            with_markov = getattr(config, "confidence_head_with_markov", False)
            self.confidence_head = _ConfidenceHead(
                config.hidden_size + (rank if with_markov else 0),
                with_markov=with_markov, device=self.fc.weight.device)

    def markov_bias(self, previous_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.bias(previous_ids)

    def sample_proposals(self, logits, anchor_ids, num_tokens, temperatures,
                         top_ps, top_ks, sample):
        base = logits.view(anchor_ids.numel(), num_tokens, -1)
        previous = anchor_ids
        proposals = []
        for position in range(num_tokens):
            previous = sample(
                base[:, position] + self.markov_bias(previous),
                temperatures,
                top_ps,
                top_ks,
            )
            proposals.append(previous)
        return torch.stack(proposals, dim=1)


def load_dspark_draft(model: DSparkDraftModel, path: str) -> None:
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
            if model.embed_tokens.weight.shape != tensor.shape:
                raise ValueError("DSpark embedding size differs from the target vocabulary")
            # 310P: build on CPU (nn.Embedding default init uses normal_, an
            # AI-CPU op broken on old drivers), then move to the target device.
            with torch.device("cpu"):
                _emb = nn.Embedding(tensor.shape[0], tensor.shape[1],
                                    dtype=model.fc.weight.dtype)
            model.embed_tokens = _emb.to(model.fc.weight.device)
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
