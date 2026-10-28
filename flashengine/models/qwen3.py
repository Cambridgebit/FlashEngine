import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from flashengine.layers.activation import SiluAndMul
from flashengine.layers.attention import Attention
from flashengine.layers.layernorm import RMSNorm
from flashengine.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from flashengine.layers.rotary_embedding import get_rope
from flashengine.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

PACKED_MODULES_MAPPING = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


def resolve_quant_scheme(config, prefix: str) -> str:
    """查 config.quant_scheme 里 `prefix + '.weight'` 的量化方案；空串 = 未量化。

    config 为 HF 配置（quant_scheme 由 Config 从 quant_model_description.json 解析后
    透传到 hf_config）；prefix 为检查点权重前缀（如 model.layers.0.self_attn.q_proj）。
    "FLOAT" 是 msmodelslim 的显式未量化标记（如部分导出只量化 gate/up、down 保持
    fp16）——按未量化处理，不能透传给 quantized_class 否则报「Unsupported scheme」。
    """
    table = getattr(config, "quant_scheme", None) or {}
    value = table.get(prefix + ".weight", "")
    if not isinstance(value, str) or value == "FLOAT":
        return ""
    return value


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        qkv_input_size: int | None = None,
        qk_norm: bool | None = None,
        qkv_quant_scheme: str | None = None,
        o_quant_scheme: str | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        if qk_norm is None:
            qk_norm = not qkv_bias

        qkv_cls = QKVParallelLinear
        o_cls = RowParallelLinear
        if qkv_quant_scheme or o_quant_scheme:
            from flashengine.layers.quant import quantized_class
            if qkv_quant_scheme:
                qkv_cls = quantized_class(QKVParallelLinear, qkv_quant_scheme)
            if o_quant_scheme:
                o_cls = quantized_class(RowParallelLinear, o_quant_scheme)
        self.qkv_proj = qkv_cls(
            qkv_input_size or hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = o_cls(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q = qkv[..., :self.q_size]
        k = qkv[..., self.q_size:self.q_size + self.kv_size]
        v = qkv[..., self.q_size + self.kv_size:]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if hasattr(self, "q_norm"):
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        gate_up_quant_scheme: str | None = None,
        down_quant_scheme: str | None = None,
    ) -> None:
        super().__init__()
        gu_cls = MergedColumnParallelLinear
        dn_cls = RowParallelLinear
        if gate_up_quant_scheme or down_quant_scheme:
            from flashengine.layers.quant import quantized_class
            if gate_up_quant_scheme:
                gu_cls = quantized_class(MergedColumnParallelLinear, gate_up_quant_scheme)
            if down_quant_scheme:
                dn_cls = quantized_class(RowParallelLinear, down_quant_scheme)
        self.gate_up_proj = gu_cls(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = dn_cls(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        prefix = f"model.layers.{layer_idx}"
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            # RoPE 表按实际窗口准备（Config 写入 _liteq_max_position_embeddings），
            # 缺该属性时退回检查点原始 max_position_embeddings。
            max_position=getattr(
                config, "_liteq_max_position_embeddings", config.max_position_embeddings),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            qkv_quant_scheme=resolve_quant_scheme(
                config, f"{prefix}.self_attn.q_proj") or None,
            o_quant_scheme=resolve_quant_scheme(
                config, f"{prefix}.self_attn.o_proj") or None,
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            gate_up_quant_scheme=resolve_quant_scheme(
                config, f"{prefix}.mlp.gate_proj") or None,
            down_quant_scheme=resolve_quant_scheme(
                config, f"{prefix}.mlp.down_proj") or None,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = PACKED_MODULES_MAPPING

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
