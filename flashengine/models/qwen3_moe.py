"""Qwen3 MoE（Qwen3-30B-A3B 系）目标模型，cuda / Ascend 310P 后端。

结构（对齐 vllm model_executor/models/qwen3_moe.py）：
- 注意力 = dense Qwen3 注意力（GQA + q/k RMSNorm + rope），直接复用 qwen3.Qwen3Attention；
- 每层 MLP 由 decoder_sparse_step / mlp_only_layers 决定走稀疏 MoE 还是 dense MLP
  （Qwen3-30B-A3B：decoder_sparse_step=1、无 mlp_only_layers → 逐层稀疏）；
- 稀疏 MoE = gate（Linear H→E，无 bias）→ softmax → top-k → norm_topk_prob 归一化
  → 专家前向（后端算子：cuda = 纯 torch bmm，见 backends/cuda.py；
    ascend310p = npu grouped matmul，见 backends/ascend.py）；
- 专家权重整存为 w13=[E, 2I, H]（gate|up）与 w2=[E, H, I]；无 shared expert；
- aux 隐层捕获与 qwen3.py 同口径（层输入残差流），供 DFlash aux 投影（MoE 路由
  不改变残差流，故 aux 口径与 dense Qwen3 一致，config 侧 target_layer_ids+1 换算）。

量化（Ascend msmodelslim W8A8）：注意力 qkv/o 走 "W8A8" 静态；专家走
"W8A8_DYNAMIC"；gate/embed/lm_head 保持 fp16（对齐 vLLM-Ascend 310P 口径）。
方案表经 config.quant_scheme 透传（键为检查点权重名，见 config.py）。

权重装载：本类自带 load_weights（对齐 qwen3_5.py 模式），因为逐专家权重
（...mlp.experts.E.{gate,up,down}_proj.{weight,weight_scale,weight_offset}）需要
堆叠进 w13/w2；其余权重复用通用 packed 映射。
"""

import torch
from torch import nn
from transformers import Qwen3MoeConfig

from flashengine.backends import get_backend
from flashengine.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from flashengine.layers.layernorm import RMSNorm
from flashengine.layers.linear import ReplicatedLinear
from flashengine.models.qwen3 import (
    PACKED_MODULES_MAPPING,
    Qwen3Attention,
    Qwen3MLP,
    resolve_quant_scheme,
)
from flashengine.utils.loader import default_weight_loader

W8A8_DYNAMIC = "W8A8_DYNAMIC"


class Qwen3MoeExperts(nn.Module):
    """专家权重容器：w13 = [E, 2I, H]（gate|up）、w2 = [E, H, I]。

    前向委托后端 fused_moe；310P 后端在 optimize_moe_weights 里把权重转置/转 NZ
    （fp16）或转 NZ + 展平尺度（W8A8_DYNAMIC）。`is_moe_experts` 供后端按模块识别。
    """

    is_moe_experts = True

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        quantized: bool = False,
    ) -> None:
        super().__init__()
        self.backend = get_backend()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.quantized = quantized
        if quantized:
            # W8A8_DYNAMIC：int8 权重 + [E, N, 1] 逐通道尺度/偏移（{w13,w2}）。
            self.w13_weight = nn.Parameter(
                torch.empty(num_experts, 2 * intermediate_size, hidden_size, dtype=torch.int8))
            self.w2_weight = nn.Parameter(
                torch.empty(num_experts, hidden_size, intermediate_size, dtype=torch.int8))
            self.w13_weight_scale = nn.Parameter(
                torch.empty(num_experts, 2 * intermediate_size, 1, dtype=torch.float32))
            self.w13_weight_offset = nn.Parameter(
                torch.empty(num_experts, 2 * intermediate_size, 1))
            self.w2_weight_scale = nn.Parameter(
                torch.empty(num_experts, hidden_size, 1, dtype=torch.float32))
            self.w2_weight_offset = nn.Parameter(
                torch.empty(num_experts, hidden_size, 1))
        else:
            self.w13_weight = nn.Parameter(
                torch.empty(num_experts, 2 * intermediate_size, hidden_size))
            self.w2_weight = nn.Parameter(
                torch.empty(num_experts, hidden_size, intermediate_size))

    def forward(self, x, topk_weights, topk_ids):
        if self.quantized:
            return self.backend.fused_moe(
                x, self.w13_weight, self.w2_weight, topk_weights, topk_ids,
                self.w13_weight_scale, self.w2_weight_scale,
            )
        return self.backend.fused_moe(
            x, self.w13_weight, self.w2_weight, topk_weights, topk_ids)


class Qwen3MoeSparseMoeBlock(nn.Module):
    """路由 + 专家：softmax→topk→（可选）归一化 → 后端专家前向。"""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.renormalize = bool(getattr(config, "norm_topk_prob", True))
        # gate 保持 fp16（对齐 vLLM-Ascend：MoE gate 不量化）。
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False)
        expert_scheme = resolve_quant_scheme(
            config, f"model.layers.{layer_idx}.mlp.experts.0.gate_proj")
        if expert_scheme and expert_scheme != W8A8_DYNAMIC:
            raise NotImplementedError(
                f"Unsupported MoE expert quantization scheme: {expert_scheme} "
                f"(310P supports {W8A8_DYNAMIC})."
            )
        self.experts = Qwen3MoeExperts(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            quantized=bool(expert_scheme),
        )
        self.backend = get_backend()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(hidden_states)
        topk_weights, topk_ids = self.backend.moe_router(
            router_logits, self.top_k, self.renormalize)
        return self.experts(hidden_states, topk_weights, topk_ids)


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        # 注意力与 dense Qwen3 逐参数同构（GQA + q/k RMSNorm + rope），直接复用；
        # 量化方案经 config.quant_scheme 逐层查表（q_proj/o_proj）。
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            qkv_quant_scheme=resolve_quant_scheme(
                config, f"model.layers.{layer_idx}.self_attn.q_proj") or None,
            o_quant_scheme=resolve_quant_scheme(
                config, f"model.layers.{layer_idx}.self_attn.o_proj") or None,
        )
        mlp_only_layers = tuple(getattr(config, "mlp_only_layers", None) or ())
        is_sparse = (
            config.num_experts > 0
            and layer_idx not in mlp_only_layers
            and (layer_idx + 1) % int(getattr(config, "decoder_sparse_step", 1) or 1) == 0
        )
        if is_sparse:
            self.mlp = Qwen3MoeSparseMoeBlock(config, layer_idx)
        else:
            if resolve_quant_scheme(config, f"model.layers.{layer_idx}.mlp.down_proj"):
                raise NotImplementedError(
                    "Quantized dense MLP layers are not supported for MoE checkpoints."
                )
            self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # aux 隐层捕获索引（层输入序号，与 qwen3.py 同口径；DFlash target_layer_ids+1）。
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def forward(self, input_ids, positions):
        capture = bool(self.aux_hidden_state_layers)
        aux_states = [] if capture else None
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for idx, layer in enumerate(self.layers):
            if capture and idx in self.aux_hidden_state_layers:
                aux_states.append(hidden_states if residual is None else hidden_states + residual)
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        if capture:
            return hidden_states, torch.cat(aux_states, dim=-1)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):

    packed_modules_mapping = PACKED_MODULES_MAPPING

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    # ---- 权重装载 ----
    _EXPERT_FIELDS = ("weight", "weight_scale", "weight_offset")

    def _load_expert_weight(self, name: str, tensor: torch.Tensor) -> None:
        """逐专家权重堆叠进 w13=[E,2I,H] / w2=[E,H,I]（及其量化尺度）。

        支持两种布局：① 逐专家（HF / msmodelslim 标准）
        `experts.<i>.{gate,up,down}_proj.{weight,weight_scale,weight_offset}`；
        ② 融合 3D `experts.{gate_up,down}_proj.<field>`。
        """
        parts = name.split(".")
        field = parts[-1]
        if field not in self._EXPERT_FIELDS:
            raise ValueError(f"Unexpected MoE expert tensor: {name}")
        li = int(parts[parts.index("layers") + 1])
        experts = self.model.layers[li].mlp.experts
        I = experts.intermediate_size

        def target(proj: str):
            base = "w2" if proj == "down_proj" else "w13"
            if field == "weight":
                return getattr(experts, base + "_weight")
            if not experts.quantized:
                raise ValueError(f"Quantized expert scale in a non-quantized model: {name}")
            suffix = "weight_scale" if field == "weight_scale" else "weight_offset"
            return getattr(experts, f"{base}_{suffix}")

        proj = parts[-2]
        if proj in ("gate_up_proj", "down_proj") and len(parts) >= 3 and parts[-3] == "experts":
            # 融合 3D 布局：整表拷贝。
            dst = target(proj).data
            if field != "weight":
                tensor = tensor.reshape(dst.shape)
            dst.copy_(tensor)
            return
        if proj == "gate_proj":
            lo, hi = 0, I
        elif proj == "up_proj":
            lo, hi = I, 2 * I
        elif proj == "down_proj":
            lo, hi = None, None
        else:
            raise ValueError(f"Unexpected MoE expert weight name: {name}")
        ei = int(parts[parts.index("experts") + 1])
        dst = target(proj).data
        if field != "weight":
            # 尺度/偏移检查点可能存 [N] 或 [N,1]，统一 reshape 到目标片形状。
            tensor = tensor.reshape(dst[ei].shape if lo is None else dst[ei, lo:hi].shape)
        if lo is None:
            dst[ei].copy_(tensor)
        else:
            dst[ei, lo:hi].copy_(tensor)

    def load_weights(self, weights) -> None:
        packed = self.packed_modules_mapping
        skipped = set()
        for name, tensor in weights:
            # 逐专家 / 融合专家权重先于通用 packed 映射拦截（expert 名含 gate_proj/up_proj）。
            if ".mlp.experts." in name:
                self._load_expert_weight(name, tensor)
                continue
            target_name = name
            shard_id = None
            for k, (v, sid) in packed.items():
                if k in name:
                    target_name = name.replace(k, v)
                    shard_id = sid
                    break
            try:
                param = self.get_parameter(target_name)
            except AttributeError:
                skipped.add(name)
                continue
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is not None:
                loader(param, tensor, shard_id)
            else:
                loader(param, tensor)
        if skipped:
            print(
                f"[FlashEngine] Qwen3Moe load_weights skipped {len(skipped)} key(s) "
                f"absent from the model, e.g. {sorted(skipped)[:3]}",
                flush=True,
            )
