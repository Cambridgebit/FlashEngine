import os
from dataclasses import dataclass, field
from typing import Any

from transformers import AutoConfig

from nanovllm.backends import detect_backend
from nanovllm.models.registry import resolve_target_model
from nanovllm.speculative.registry import resolve_spec_method


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    enable_chunked_prefill: bool = True
    max_num_partial_prefills: int = 1
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    backend: str = "auto"
    hccl_port: int = 28000
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_prefix_cache: bool = True
    spec_model: str = ""
    spec_num_tokens: int = 4
    spec_aux_layers: tuple[int, ...] = ()
    spec_method: str = ""
    draft_config: AutoConfig | None = None
    target_adapter: Any = field(default=None, init=False, repr=False)
    spec_adapter: Any = field(default=None, init=False, repr=False)
    model_config: AutoConfig | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.backend != "auto":
            os.environ["NANOVLLM_BACKEND"] = self.backend
        backend_name = detect_backend()
        if not os.path.isdir(self.model):
            raise ValueError(f"Model must be a local directory: {self.model}")
        if self.kvcache_block_size < 16 or (
            self.kvcache_block_size & (self.kvcache_block_size - 1)
        ):
            raise ValueError("kvcache_block_size must be a power of two and at least 16")
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_partial_prefills < 1:
            raise ValueError("max_num_partial_prefills must be positive")
        if backend_name == "cuda" and self.kvcache_block_size % 256:
            raise ValueError(
                "CUDA flash-attn paged KV requires block_size to be a power "
                "of two and a multiple of 256 (256, 512, 1024, ...)."
            )
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be in [1, 8]")
        if backend_name == "ascend310p" and self.tensor_parallel_size != 1:
            raise NotImplementedError(
                "Ascend 310P tensor parallelism is not implemented; use "
                "tensor_parallel_size=1."
            )

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.target_adapter = resolve_target_model(self.hf_config)
        self.model_config = self.target_adapter.model_config(self.hf_config)
        self.max_model_len = min(
            self.max_model_len, self.model_config.max_position_embeddings
        )
        if (
            not self.enable_chunked_prefill
            and self.max_num_batched_tokens < self.max_model_len
        ):
            raise ValueError(
                "Chunked prefill is disabled, so max_num_batched_tokens must "
                "be at least max_model_len."
            )

        if self.spec_model:
            if self.tensor_parallel_size != 1:
                raise NotImplementedError(
                    "Speculative decoding currently requires tensor_parallel_size=1."
                )
            if not os.path.isdir(self.spec_model):
                raise ValueError(f"Draft model must be a local directory: {self.spec_model}")
            if self.spec_num_tokens < 1:
                raise ValueError("spec_num_tokens must be positive")
            self.draft_config = AutoConfig.from_pretrained(self.spec_model)
            self.spec_adapter = resolve_spec_method(self.draft_config)
            self.spec_method = self.spec_adapter.name
            self.target_adapter.validate_spec_method(self.spec_method)
            self.spec_num_tokens = self.spec_adapter.configure_num_tokens(
                self.draft_config, self.spec_num_tokens
            )
            self.spec_aux_layers = self.target_adapter.resolve_aux_layers(
                self.spec_method,
                self.draft_config,
                self.spec_aux_layers,
                self.model_config,
            )
            if self.spec_adapter.block_draft:
                target_layers = tuple(
                    (getattr(self.draft_config, "dflash_config", None) or {})[
                        "target_layer_ids"
                    ]
                )
                if len(self.spec_aux_layers) != len(target_layers):
                    raise ValueError(
                        "Draft projection width does not match auxiliary layers: "
                        f"spec_aux_layers={self.spec_aux_layers}, "
                        f"target_layer_ids={target_layers}."
                    )
            elif len(self.spec_aux_layers) != 3:
                raise ValueError(
                    "EAGLE3 draft projection expects three auxiliary hidden states."
                )
            if max(self.spec_aux_layers) >= self.model_config.num_hidden_layers:
                raise ValueError(
                    f"Auxiliary layer index is out of range: {self.spec_aux_layers}."
                )

        if backend_name == "ascend310p":
            if self.kvcache_block_size not in (64, 128):
                print(
                    f"[nano-vllm] Ascend 310P KV-cache block size adjusted: "
                    f"{self.kvcache_block_size} -> 128; supported values are 64 and 128.",
                    flush=True,
                )
                self.kvcache_block_size = 128
            head_dim = getattr(self.model_config, "head_dim", None)
            if head_dim is None:
                head_dim = (
                    self.model_config.hidden_size
                    // self.model_config.num_attention_heads
                )
            if self.kvcache_block_size * head_dim > 128 * 128:
                raise ValueError(
                    "Ascend 310P paged attention requires block_size * head_dim "
                    f"<= 16384; got {self.kvcache_block_size} * {head_dim}."
                )
