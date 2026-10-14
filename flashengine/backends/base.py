from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
import torch.nn.functional as F

# Match the 0924 CUDA/CoreX fast path: keep enough Dynamo variants for
# dynamic batch shapes instead of falling back to eager after the default limit.
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 256
else:
    # 兼容厂商环境中旧版 PyTorch 的配置名。
    torch._dynamo.config.cache_size_limit = 256


def _apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class BackendBase(ABC):

    name = "base"
    dist_name = "nccl"
    supports_spec_graph = False
    supports_spec = True
    supports_prefix_cache_attn = True
    supports_paged_context_attn = False
    supports_dflash_paged = False
    device_name = "cuda"
    default_dtype = None
    graph_sample_default = True
    supports_device_sampling_rng = False
    supports_fused_gdn = False
    supports_fused_gdn_prefill = False
    supports_fused_qwen35_norm = False
    supports_device_state_commit = False
    supports_greedy_graph_variants = False
    supports_dflash_dense_varlen = False
    supports_async_prefix_snapshots = False
    spec_graph_try_default = True

    def normalize_config(self, config) -> None:
        # flash-attn 类算子仅支持 fp16/bf16：fp32 检查点自动降 bf16（加载期转换），
        # 否则 attention 前向直接 RuntimeError。后端可按需覆盖（310P 用 default_dtype
        # 强制 fp16，故覆盖且不调用本方法）。
        if getattr(config.model_config, "dtype", None) == torch.float32:
            print(
                f"[FlashEngine] Backend {self.name} does not support fp32 attention "
                "(flash-attn requires fp16/bf16); using bfloat16.",
                flush=True,
            )
            config.model_config.dtype = torch.bfloat16

    def validate_config(self, config) -> None:
        pass

    def runtime_info(self) -> dict[str, str]:
        return {
            "attention_backend": "flash-attn",
            "graph_backend": "CUDA Graph",
            "kv_cache_layout": "BSHD",
            "fused_ops": "backend default",
        }

    def create_decode_graph_executor(self, plan):
        return None

    def prefix_cache_disable_reason(self) -> str | None:
        return None

    def dense_noncausal_attention(self, q, k, v, mask, scale: float):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

    def gdn_causal_conv1d(self, inputs, weight):
        """GDN 的已补历史窗口深度卷积，输入为 [batch, channels, length]。"""
        return F.conv1d(inputs, weight, groups=weight.shape[0])

    def linear(self, x, weight, bias=None):
        return F.linear(x, weight, bias)

    def optimize_model_weights(self, model) -> int:
        return 0

    def optimize_block_draft_weights(self, draft_model, target_lm_head) -> int:
        return 0

    # ---- MoE（Qwen3-30B-A3B 系）算子代理 ----
    supports_moe = False
    # W8A8 int8 量化算子可用性（310P 独占；见 layers/quant.py）。
    supports_w8a8 = False
    # Quark W8A8 int8 检查点加载期反量化能力（无 int8 算子的 cuda/corex 置真）。
    supports_int8_dequant = False
    # 默认专家前向单次聚合的权重预算（字节）：按它换算每次 index_select 的行数，
    # 限制大 bucket 图捕获时的峰值显存（见 fused_moe）。
    _MOE_GATHER_BYTES = 256 << 20

    def moe_router(self, router_logits, top_k: int, renormalize: bool):
        """返回 (topk_weights, topk_ids)；默认纯 torch（softmax→topk→可选归一化）。

        310P 覆盖为 npu_moe_gating_top_k_softmax（fp16 + ≤1024 token 分块）。
        """
        probs = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(probs, top_k, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights.to(router_logits.dtype), topk_ids.to(torch.int32)

    def fused_moe(self, hidden_states, w13_weight, w2_weight, topk_weights, topk_ids,
                  w13_scale=None, w2_scale=None):
        """专家前向默认纯 torch 向量化实现（fp16/bf16/fp32），供 CUDA/CoreX 使用。

        权重布局与模型一致：w13 = [E, 2I, H]（gate|up）、w2 = [E, H, I]。按
        (token, top-k) 展平后用 index_select 取每行对应专家权重（形状静态、无
        host 同步），bmm 逐行完成 gate/up → silu → down，最后 index_add 回各
        token 并按 top-k 权重求和。分块只压峰值显存、不改数值，故 CUDA Graph 可
        捕获。310P 覆盖为 npu grouped matmul（含 W8A8_DYNAMIC 分支）。
        """
        if w13_scale is not None or w2_scale is not None:
            raise NotImplementedError(
                f"Backend {self.name} does not implement W8A8 MoE experts; "
                "int8 MoE experts are only available on ascend310p."
            )
        x = hidden_states
        hidden_size = x.shape[-1]
        top_k = topk_ids.shape[-1]
        expert_ids = topk_ids.reshape(-1).to(torch.long)          # [N]
        weights = topk_weights.reshape(-1).to(x.dtype)            # [N]
        x_expanded = x.repeat_interleave(top_k, dim=0)            # [N, H]
        token_ids = torch.arange(
            expert_ids.numel(), device=x.device).div(top_k, rounding_mode="floor")
        out = torch.zeros_like(x)
        row_bytes = w13_weight.shape[1] * hidden_size * x.element_size()
        chunk = max(1, self._MOE_GATHER_BYTES // max(1, row_bytes))
        for start in range(0, expert_ids.numel(), chunk):
            end = min(start + chunk, expert_ids.numel())
            ids = expert_ids[start:end]
            w13 = w13_weight[ids]                                 # [c, 2I, H]
            gate_up = torch.bmm(w13, x_expanded[start:end].unsqueeze(-1)).squeeze(-1)
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up                               # [c, I]
            w2 = w2_weight[ids]                                   # [c, H, I]
            token_out = torch.bmm(w2, act.unsqueeze(-1)).squeeze(-1)  # [c, H]
            token_out = token_out * weights[start:end].unsqueeze(-1)
            out.index_add_(0, token_ids[start:end], token_out)
        return out

    def optimize_moe_weights(self, model) -> int:
        """MoE 专家权重的后端后处理（310P = transpose + FRACTAL_NZ）。"""
        return 0

    # ---- W8A8 int8 量化算子代理（310P 独占）----
    def prepare_quant_linear(self, module, scheme: str) -> None:
        """量化权重装载后的设备侧后处理（NZ 布局 / 尺度展开 / 激活尺度预计算）。"""
        raise NotImplementedError(
            f"Backend {self.name} does not implement W8A8 quantization."
        )

    def quant_linear_static(self, x, weight, deq_scale, quant_bias,
                            input_scale, input_scale_reciprocal, input_offset,
                            out_dtype, bias):
        raise NotImplementedError(
            f"Backend {self.name} does not implement W8A8 static matmul."
        )

    def quant_linear_dynamic(self, x, weight, weight_scale, out_dtype, bias):
        raise NotImplementedError(
            f"Backend {self.name} does not implement W8A8 dynamic matmul."
        )

    def dflash_paged_attention(
        self,
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        rows: int,
        scale: float,
        max_seq_len: int,
    ):
        raise NotImplementedError("DFlash paged attention is not supported by this backend")

    @abstractmethod
    def store_kvcache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        raise NotImplementedError

    @abstractmethod
    def prefill_attention(
        self,
        q,
        k,
        v,
        context,
        scale,
    ):
        raise NotImplementedError

    @abstractmethod
    def decode_attention(
        self,
        q,
        k_cache,
        v_cache,
        context,
        scale,
    ):
        raise NotImplementedError

    def set_device(self, rank):
        torch.cuda.set_device(rank)

    def synchronize(self):
        torch.cuda.synchronize()

    def empty_cache(self):
        torch.cuda.empty_cache()

    def shutdown(self):
        self.synchronize()
        self.empty_cache()

    def mem_get_info(self):
        return torch.cuda.mem_get_info()

    def memory_stats(self):
        return torch.cuda.memory_stats()

    def reset_peak_memory_stats(self):
        torch.cuda.reset_peak_memory_stats()

    def new_graph(self):
        return torch.cuda.CUDAGraph()

    @contextmanager
    def graph_ctx(self, graph, pool=None):
        with torch.cuda.graph(graph, pool=pool):
            yield

    def graph_pool(self, graph):
        return graph.pool()

    @staticmethod
    @torch.compile
    def rms_norm(x, weight, eps):
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + eps))
        x = x.to(orig_dtype).mul_(weight)
        return x

    @staticmethod
    @torch.compile
    def add_rms_norm(x, residual, weight, eps):
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + eps))
        x = x.to(orig_dtype).mul_(weight)
        return x, residual

    @staticmethod
    @torch.compile
    def silu_and_mul(x):
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    @staticmethod
    def rope(positions, q, k, cos_sin_cache):
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        return q, k

    def get_kv_cache_shape(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        return (
            2,
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )

    def graph_slot_mapping_dtype(self):
        # CUDA Graph 的 static slot_mapping buffer dtype。
        return torch.int32

    def allocate_kv_layers(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype,
        device,
    ):
        return None

    def graph_pad_block(self, num_blocks: int) -> int:
        return -1

    def graph_pad_context_len(self) -> int:
        return 0

    def graph_capture_block(self, num_blocks: int) -> int:
        # 捕获用有效的物理页，-1 仅表示回放 padding，不能给暖启动窗口读取。
        return max(0, self.graph_pad_block(num_blocks))

    def graph_capture_context_len(self) -> int:
        return max(1, self.graph_pad_context_len())

    def reserve_dummy_kv_block(self) -> bool:
        return False

    def graph_warmup_iters(self) -> int:
        return 1

    def all_reduce(self, tensor):
        import torch.distributed as dist

        dist.all_reduce(tensor)
        return tensor
