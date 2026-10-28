import json
import os
import torch
from dataclasses import dataclass, field
from typing import Any

from transformers import AutoConfig

from flashengine.backends import get_backend
from flashengine.models.registry import resolve_target_model
from flashengine.precision import require_unquantized_checkpoint
from flashengine.speculative.registry import load_draft_config, resolve_spec_method
from flashengine.support_matrix import require_supported
from flashengine.engine.memory import validate_memory_options


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    enable_chunked_prefill: bool = True
    max_num_partial_prefills: int = 1
    gpu_memory_utilization: float = 0.9
    memory_budget_bytes: int = 0
    device_memory_reserve_bytes: int = 0
    memory_workspace_bytes: int = 256 * 1024 * 1024
    graph_memory_reserve_bytes: int = 256 * 1024 * 1024
    device_ids: tuple[int, ...] = ()
    max_pending_requests: int = 1024
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    backend: str = "auto"
    dtype: str | None = None
    # ---- 统一 FP16 发布入口（默认关闭，保持主项目既有行为）----
    # fp16_only：拒绝量化/反量化检查点 + 计算类型固定 FP16 + 启动参数审计。
    fp16_only: bool = False
    # 310P 单行 decode 的线性算子：functional（默认）/ selective（QKV/O/Down 走 npu_linear）。
    decode_linear_backend: str = "functional"
    # 图外 greedy 归约位置：npu（设备 argmax，默认）/ cpu（FP16 logits 读回）。
    greedy_backend: str = "npu"
    hccl_port: int = 0
    shm_name: str = ""
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_prefix_cache: bool = True
    recurrent_prefix_cache_bytes: int = 256 * 1024 * 1024
    spec_model: str = ""
    spec_num_tokens: int = 4
    spec_aux_layers: tuple[int, ...] = ()
    spec_method: str = ""
    draft_config: AutoConfig | None = None
    target_adapter: Any = field(default=None, init=False, repr=False)
    spec_adapter: Any = field(default=None, init=False, repr=False)
    model_config: AutoConfig | None = field(default=None, init=False, repr=False)
    backend_impl: Any = field(default=None, init=False, repr=False)
    spec_is_block_draft: bool = field(default=False, init=False)
    spec_sample_from_anchor: bool = field(default=False, init=False)
    use_paged_draft: bool = field(default=False, init=False)
    keep_last_block_unhashed: bool = field(default=False, init=False)
    # ---- MoE（Qwen3-30B-A3B 系）派生标记与形状（__post_init__ 填充）----
    is_moe: bool = field(default=False, init=False)
    num_experts: int = field(default=0, init=False)
    num_experts_per_tok: int = field(default=0, init=False)
    moe_intermediate_size: int = field(default=0, init=False)
    norm_topk_prob: bool = field(default=True, init=False)
    decoder_sparse_step: int = field(default=1, init=False)
    moe_mlp_only_layers: tuple[int, ...] = field(default=(), init=False)
    # ---- W8A8 量化（Ascend msmodelslim "ascend" 导出）----
    quant_method: str = field(default="", init=False)
    quant_scheme: dict = field(default_factory=dict, init=False)
    # Quark W8A8 int8 检查点：加载期权重反量化（cuda/corex，无 int8 算子）。
    dequant_int8_weights: bool = field(default=False, init=False)

    def __post_init__(self):
        validate_memory_options(self.gpu_memory_utilization, self.memory_budget_bytes,
                                self.device_memory_reserve_bytes, self.memory_workspace_bytes,
                                self.graph_memory_reserve_bytes, self.num_kvcache_blocks)
        self.device_ids = tuple(self.device_ids) or tuple(range(self.tensor_parallel_size))
        if (len(self.device_ids) != self.tensor_parallel_size
                or len(set(self.device_ids)) != len(self.device_ids)
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.device_ids)):
            raise ValueError("device_ids 必须包含每个 TP rank 对应的独立设备索引")
        if (isinstance(self.max_pending_requests, bool)
                or not isinstance(self.max_pending_requests, int) or self.max_pending_requests < 1):
            raise ValueError("max_pending_requests 必须为正数")
        if not 0 <= self.hccl_port <= 65535:
            raise ValueError("hccl_port 必须在 [0, 65535] 内")
        if self.decode_linear_backend not in ("functional", "selective"):
            raise ValueError("decode_linear_backend 仅支持 functional/selective")
        if self.greedy_backend not in ("npu", "cpu"):
            raise ValueError("greedy_backend 仅支持 npu/cpu")
        if self.backend != "auto":
            os.environ["FLASHENGINE_BACKEND"] = self.backend
        # 显式传入的非默认值覆盖 env（保持默认值时不动 env，避免盖掉外部设置）。
        if self.decode_linear_backend != "functional":
            os.environ["LITEQ_DECODE_LINEAR_BACKEND"] = self.decode_linear_backend
        if self.greedy_backend != "npu":
            os.environ["LITEQ_GREEDY_BACKEND"] = self.greedy_backend
        self.backend_impl = get_backend()
        if (self.backend_impl.name != "ascend310p"
                and (self.decode_linear_backend != "functional" or self.greedy_backend != "npu")):
            # 仅告警不失败：这两个开关只作用于 310P，其它后端保持既有行为。
            print("[FlashEngine] decode_linear_backend/greedy_backend 仅对 ascend310p 生效，"
                  f"当前后端 {self.backend_impl.name} 按默认实现。", flush=True)
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
        if self.recurrent_prefix_cache_bytes < 0:
            raise ValueError("recurrent_prefix_cache_bytes must be non-negative")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be in [1, 8]")

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.target_adapter = resolve_target_model(self.hf_config)
        self.model_config = self.target_adapter.model_config(self.hf_config)
        # 加载期浮点守卫是进程级开关，按本次配置显式设定并复位，避免同进程内
        # 先构造 fp16_only 配置后影响后续默认配置；不开 fp16_only 时等价于不校验。
        from flashengine.utils.loader import set_require_float_weights
        set_require_float_weights(bool(self.fp16_only))
        if self.fp16_only:
            # 统一 FP16 发布入口：先拒绝量化/反量化检查点，再固定计算类型。
            # 默认关闭时不进入本分支，W8A8/Quark 装配逻辑完全保持原样。
            require_unquantized_checkpoint(self.model, self.hf_config)
            if self.dtype not in (None, "float16"):
                raise ValueError("fp16_only 要求 dtype 为 float16 或不指定（跟随统一入口）")
            self.model_config.dtype = torch.float16
        if self.dtype is not None:
            if self.dtype not in ("float16", "bfloat16"):
                raise ValueError("dtype 仅支持 float16/bfloat16")
            requested = getattr(torch, self.dtype)
            forced = self.backend_impl.default_dtype
            if forced is not None and forced != requested:
                raise ValueError("dtype 与后端强制计算类型冲突")
            # planner 和模型必须读取同一个实际 dtype，不能只覆盖权重转换。
            self.model_config.dtype = requested
        if self.target_adapter.recurrent_state:
            if self.tensor_parallel_size != 1:
                raise NotImplementedError("带 recurrent 状态的目标模型当前要求 TP=1")
            if self.enable_prefix_cache and not self.recurrent_prefix_cache_bytes:
                print("[FlashEngine] Recurrent 前缀快照预算为 0，关闭 prefix cache。", flush=True)
                self.enable_prefix_cache = False
            self.max_num_batched_tokens = min(self.max_num_batched_tokens, self.max_model_len)
        self.max_model_len = min(
            self.max_model_len, self.model_config.max_position_embeddings
        )
        # RoPE 表准备上限跟随实际窗口（后端按它展开 cos/sin 全表），避免按
        # 检查点原始 max_position_embeddings（如 40960）白准备内存。
        self.model_config._liteq_max_position_embeddings = self.max_model_len
        # ---- MoE（Qwen3-30B-A3B）元数据（形状自 model_config 取）----
        self.is_moe = bool(getattr(self.target_adapter, "is_moe", False))
        if self.is_moe:
            mc = self.model_config
            self.num_experts = int(mc.num_experts)
            self.num_experts_per_tok = int(mc.num_experts_per_tok)
            self.moe_intermediate_size = int(mc.moe_intermediate_size)
            self.norm_topk_prob = bool(getattr(mc, "norm_topk_prob", True))
            self.decoder_sparse_step = int(getattr(mc, "decoder_sparse_step", 1) or 1)
            self.moe_mlp_only_layers = tuple(getattr(mc, "mlp_only_layers", None) or ())
            if not 1 <= self.num_experts_per_tok <= self.num_experts:
                raise ValueError(
                    "MoE requires 1 <= num_experts_per_tok <= num_experts; "
                    f"got {self.num_experts_per_tok} / {self.num_experts}."
                )
            if self.moe_intermediate_size < 1:
                raise ValueError("MoE moe_intermediate_size must be positive.")
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
            self.draft_config = load_draft_config(self.spec_model)
            if self.fp16_only:
                # 草稿侧同样只接受原始浮点检查点（与目标模型同口径）。
                require_unquantized_checkpoint(self.spec_model, self.draft_config)
            self.spec_adapter = resolve_spec_method(self.draft_config)
            self.spec_method = self.spec_adapter.name
            self.spec_is_block_draft = self.spec_adapter.block_draft
            self.spec_sample_from_anchor = bool(
                self.spec_adapter.can_sample_from_anchor
                and getattr(self.draft_config, "sample_from_anchor", True)
            )
            self.use_paged_draft = bool(
                self.spec_is_block_draft
                and self.backend_impl.supports_dflash_paged
                # 当前分页草稿内核只支持全局非因果注意力；窗口草稿走稠密掩码。
                and not getattr(self.draft_config, "block_draft_requires_mask", False)
            )
            self.keep_last_block_unhashed = self.use_paged_draft
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
                if (self.draft_config.hidden_size != self.model_config.hidden_size
                        or self.draft_config.vocab_size != self.model_config.vocab_size):
                    raise ValueError("块草稿的 hidden_size/vocab_size 必须与目标模型匹配")
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

        self._resolve_quantization()
        require_supported(
            self.target_adapter.name,
            self.backend_impl.name,
            self.quant_method or "none",
            self.spec_method or "none",
            backend_impl=self.backend_impl,
        )
        self.backend_impl.normalize_config(self)
        self.backend_impl.validate_config(self)

    def _resolve_quantization(self) -> None:
        """识别量化检查点格式（Ascend msmodelslim / AMD Quark），设置 quant_method。

        - msmodelslim：读 quant_model_description.json（逐权重名 → 方案），方案表透传到
          hf_config.quant_scheme（模型侧逐层查表构造量化线性层）。
        - Quark（config.json 的 quantization_config.quant_method == "quark"）：int8 权重 +
          per-channel scale/zero_point，无 int8 算子的后端（cuda/corex）在加载期反量化。
        量化本身的支持范围（后端/目标）由 support_matrix 统一裁决。
        """
        if self.fp16_only:
            # fp16_only：__post_init__ 已拒绝量化检查点，这里固定为未量化，
            # 避免模型侧按 quant_scheme 构造量化线性层。
            self.quant_method = ""
            self.quant_scheme = {}
            self.dequant_int8_weights = False
            return
        quant_desc = os.path.join(self.model, "quant_model_description.json")
        if os.path.exists(quant_desc):
            with open(quant_desc, "r", encoding="utf-8") as f:
                desc = json.load(f)
            self.quant_scheme = {k: v for k, v in desc.items() if isinstance(v, str)}
            self.quant_method = "w8a8"
            self.hf_config.quant_scheme = self.quant_scheme
            if self.model_config is not None:
                self.model_config.quant_scheme = self.quant_scheme
            return
        quant_config = getattr(self.hf_config, "quantization_config", None)
        if isinstance(quant_config, dict) and quant_config.get("quant_method") == "quark":
            self.quant_method = "quark_w8a8"
            self.dequant_int8_weights = True
