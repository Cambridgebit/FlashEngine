
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .base import BackendBase

_npu = None
_vllm_import_thread = None

_COMPRESSED_MASK_SEQ_LEN = 2048
_MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION = 3
_MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION = 5
_PAGED_ATTENTION_COMPRESSED_MASK_VALUE = -10000.0

# ---- 按长度缓存的掩码/缓冲上限（防止长测中随“见过的序列长度”无限累积） ----
# 310P 上每个新长度都会生成一份 O(L²) 因果掩码（L=20k、fp16 单份约 762MiB），
# 旧实现只增不减：长测里出现几十~几百个不同长度后可累积到数十 GB，把统存
# 池推到 90%+（停止/重启才归还）。策略：
#   1) 掩码长度按 _MASK_LEN_BUCKET 向上取整做键——因果掩码的左上角子矩阵性质
#      保证“更大的掩码”对更短序列完全等价（批量内多序列本就共用按最大长度
#      构造的掩码），取整只提高同桶复用、不改变数值；
#   2) 每个缓存池按字节做 LRU 上限，仅当超预算时淘汰最久未用项。
# 稳态命中率不受影响（同一批内 36 层复用同一份掩码），内存封顶为常量级。
_MASK_LEN_BUCKET = 1024
_MASK_CACHE_BYTES = 2 << 30            # _nd_mask / _nz_mask 每池上限 2 GiB
_DFLASH_MASK_CACHE_BYTES = 64 << 20    # 草稿 non-causal 掩码池上限
_ROPE_BUFFER_CACHE_BYTES = 64 << 20    # RoPE cos/sin 复用缓冲池上限


def _bucket_mask_len(max_seq_len: int) -> int:
    step = _MASK_LEN_BUCKET
    return max(step, ((max_seq_len + step - 1) // step) * step)


def _tensor_bytes(value) -> int:
    if isinstance(value, tuple):
        return sum(_tensor_bytes(item) for item in value)
    numel = getattr(value, "numel", None)
    if numel is None:
        return 0
    return value.numel() * value.element_size()


def _lru_get(cache: "OrderedDict", key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _lru_put(cache: "OrderedDict", key, value, max_bytes: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    total = sum(_tensor_bytes(item) for item in cache.values())
    while total > max_bytes and len(cache) > 1:
        _, dropped = cache.popitem(last=False)
        total -= _tensor_bytes(dropped)

_REQUIRED_OPS = (
    "_npu_reshape_and_cache",
    "_npu_paged_attention",
    "_npu_flash_attention",
    "npu_rms_norm",
    "npu_add_rms_norm",
    "npu_swiglu",
)


def _start_vllm_gemm_import():
    """后台线程导入 vllm_ascend，注册 torch.ops.vllm.unquantized_gemm。

    导入 vllm/vllm_ascend 全套在 310P 板上实测约 6s，同步执行会直接阻塞启动；
    放到后台与权重加载（20s 量级）重叠，GEMM 算子在首次使用时惰性解析。
    未安装 / 导入失败时静默回退 torch（与旧行为一致）；环境变量
    FLASHENGINE_VLLM_GEMM=0 可完全跳过该导入。
    """
    global _vllm_import_thread
    if _vllm_import_thread is not None:
        return _vllm_import_thread
    if os.environ.get("FLASHENGINE_VLLM_GEMM", "auto").strip().lower() in (
            "0", "off", "false"):
        return None

    def _load():
        try:
            import vllm_ascend.ops.linear  # noqa: F401
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 -- 仅是可选 GEMM，失败不影响引擎
            print("[FlashEngine] vllm_ascend GEMM import failed "
                  f"(falling back to torch): {exc}", flush=True)

    _vllm_import_thread = threading.Thread(
        target=_load, name="flashengine-vllm-gemm-import", daemon=True)
    _vllm_import_thread.start()
    return _vllm_import_thread


def _torch_npu():
    global _npu
    if _npu is None:
        try:
            import torch_npu
        except ImportError as e:
            raise RuntimeError(
                "missing torch_npu: ascend310p needs torch_npu/CANN "
                "(torch-npu==2.10.0.post4 / CANN 9.1.0)"
            ) from e
        missing = [op for op in _REQUIRED_OPS if not hasattr(torch_npu, op)]
        if missing:
            raise RuntimeError(
                "torch_npu missing required ops: " + ", ".join(missing)
                + " -- upgrade torch_npu/CANN (torch-npu==2.10.0.post4 / CANN 9.1.0)"
            )
        _start_vllm_gemm_import()
        _npu = torch_npu
    return _npu


def _acl_format_nz():
    tn = _torch_npu()
    acl_format = getattr(tn, "ACL_FORMAT_FRACTAL_NZ", 29)
    try:
        from torch_npu.utils import ACL_FORMAT_FRACTAL_NZ as acl_format
    except Exception:
        pass
    return acl_format


def _acl_format_nd():
    tn = _torch_npu()
    acl_format = getattr(tn, "ACL_FORMAT_ND", 2)
    try:
        from torch_npu.utils import ACL_FORMAT_ND as acl_format
    except Exception:
        pass
    return acl_format


class Ascend310PBackend(BackendBase):

    name = "ascend310p"
    dist_name = "hccl"
    supports_spec = True
    supports_spec_graph = True
    supports_prefix_cache_attn = False
    supports_paged_context_attn = True
    supports_splitfuse = False
    supports_moe = True
    supports_w8a8 = True
    device_name = "npu"
    default_dtype = torch.float16
    graph_sample_default = False
    spec_graph_try_default = True

    def gdn_causal_conv1d(self, inputs, weight):
        # 310P 用静态短卷积展开，避免数千组 depthwise Conv1d 的算子约束。
        # fp16→fp32 转换一次做整张（逐抽头 cast 位等，省 kernel-1 次小算子）；
        # 权重同样一次转 fp32 再逐抽头取列。
        kernel = weight.shape[-1]
        length = inputs.shape[-1] - kernel + 1
        x = inputs.float()
        w = weight[:, 0, :].float()  # [C, K]
        out = x[:, :, :length] * w[:, 0][None, :, None]
        for offset in range(1, kernel):
            out = out + x[:, :, offset:offset + length] * w[:, offset][None, :, None]
        return out.to(inputs.dtype)

    def normalize_config(self, config) -> None:
        model = config.model_config
        head_dim = getattr(model, "head_dim", model.hidden_size // model.num_attention_heads)
        preferred = 64 if head_dim > 128 else 128
        if (config.kvcache_block_size not in (64, 128)
                or config.kvcache_block_size * head_dim > 16384):
            print(
                "[FlashEngine] Ascend 310P KV-cache block size adjusted: "
                f"{config.kvcache_block_size} -> {preferred}; head_dim={head_dim}.",
                flush=True,
            )
            config.kvcache_block_size = preferred

    def validate_config(self, config) -> None:
        if config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "Ascend 310P tensor parallelism is not implemented; use "
                "tensor_parallel_size=1."
            )
        model = config.model_config
        head_dim = getattr(model, "head_dim", None)
        if head_dim is None:
            head_dim = model.hidden_size // model.num_attention_heads
        if config.kvcache_block_size * head_dim > 128 * 128:
            raise ValueError(
                "Ascend 310P paged attention requires block_size * head_dim "
                f"<= 16384; got {config.kvcache_block_size} * {head_dim}."
            )
        if config.is_moe and model.hidden_size % 16:
            # npu_moe_init_routing_v2 要求激活末维 16 对齐（对齐 vLLM-Ascend 310P
            # token_dispatcher 的断言），否则运行期报错难定位——启动即拒绝。
            raise ValueError(
                "Ascend 310P MoE requires hidden_size aligned to 16 for "
                f"npu_moe_init_routing_v2; got hidden_size={model.hidden_size}."
            )

    def __init__(self):
        tn = _torch_npu()
        self._setup_runtime()
        self._nz_mask = OrderedDict()
        self._nd_mask = OrderedDict()
        self._dflash_mask = OrderedDict()
        self._splitfuse_mask = {}
        self._decode_out = {}
        self._rope_cache = {}
        self._rope_apply_cache = OrderedDict()
        # vllm_ascend 的 unquantized_gemm 在后台线程导入（见 _start_vllm_gemm_import），
        # 这里只登记未解析状态，首次 linear()/runtime_info() 时再惰性取算子。
        self._unquantized_gemm = None
        self._gemm_op_resolved = False
        self._splitfuse_v2 = getattr(tn, "_npu_paged_attention_splitfuse_v2", None)
        self._splitfuse_legacy = getattr(tn, "_npu_paged_attention_splitfuse", None)
        self.supports_splitfuse = (
            self._splitfuse_v2 is not None or self._splitfuse_legacy is not None
        )
        dflash_paged_flag = os.getenv(
            "FLASHENGINE_310P_DFLASH_PAGED", "auto").strip().lower()
        if dflash_paged_flag not in {"auto", "1", "true", "on", "0", "false", "off"}:
            raise ValueError(
                "FLASHENGINE_310P_DFLASH_PAGED must be auto/1/0, got "
                + dflash_paged_flag
            )
        force_paged = dflash_paged_flag in {"1", "true", "on"}
        disable_paged = dflash_paged_flag in {"0", "false", "off"}
        if force_paged and not self.supports_splitfuse:
            raise RuntimeError(
                "FLASHENGINE_310P_DFLASH_PAGED=1 requires "
                "_npu_paged_attention_splitfuse(_v2)"
            )
        self.supports_dflash_paged = self.supports_splitfuse and not disable_paged
        self._compressed_mask_supported = (
            hasattr(tn, "_npu_flash_attention_v3")
            and hasattr(tn, "_npu_paged_attention_splitfuse_v2")
        )
        requested_flash = os.getenv("FLASHENGINE_310P_FLASH_OP", "auto").lower()
        selected_flash = (
            "v3" if requested_flash == "auto" and self._compressed_mask_supported
            else "legacy" if requested_flash == "auto"
            else requested_flash
        )
        splitfuse = (
            "v2" if self._splitfuse_v2 is not None
            else "legacy" if self._splitfuse_legacy is not None
            else "off"
        )
        rope = (
            "apply_rotary" if hasattr(tn, "npu_apply_rotary_pos_emb")
            else "rotary_embedding" if hasattr(tn, "_npu_rotary_embedding")
            else "torch"
        )
        self._runtime_info = {
            "attention_backend": (
                f"torch-npu (flash={selected_flash}, splitfuse={splitfuse})"
            ),
            "graph_backend": "NPUGraph/ACLGraph",
            "kv_cache_layout": "FRACTAL_NZ",
            "fused_ops": (
                f"RMSNorm, AddRMSNorm, SwiGLU, RoPE={rope}, "
                "Linear/LM-head weights=FRACTAL_NZ, "
                "GEMM={gemm}"  # 首次 runtime_info() 时解析（vllm 在后台导入）
            ),
            "dflash_backend": (
                f"paged-KV/SplitFuse-{splitfuse}" if self.supports_dflash_paged
                else "dense-KV/SDPA-fallback"
            ),
        }

    def runtime_info(self) -> dict[str, str]:
        info = dict(self._runtime_info)
        gemm = ("vllm.unquantized_gemm" if self._gemm_op() is not None
                else "torch-fallback")
        info["fused_ops"] = info["fused_ops"].format(gemm=gemm)
        nd_bytes = sum(_tensor_bytes(v) for v in self._nd_mask.values())
        nz_bytes = sum(_tensor_bytes(v) for v in self._nz_mask.values())
        info["mask_cache"] = (
            f"nd={len(self._nd_mask)}({nd_bytes >> 20}MiB), "
            f"nz={len(self._nz_mask)}({nz_bytes >> 20}MiB)"
        )
        return info

    def _gemm_op(self):
        """取 vllm_ascend 的 unquantized_gemm（后台导入完成后解析一次）。"""
        if not self._gemm_op_resolved:
            thread = _vllm_import_thread
            if thread is not None:
                thread.join(timeout=120)
            vllm_ops = getattr(torch.ops, "vllm", None)
            self._unquantized_gemm = (
                getattr(vllm_ops, "unquantized_gemm", None)
                if vllm_ops is not None else None
            )
            self._gemm_op_resolved = True
        return self._unquantized_gemm

    def linear(self, x, weight, bias=None):
        gemm = self._gemm_op()
        if gemm is not None:
            return gemm(x, weight, bias)
        return F.linear(x, weight, bias)

    def optimize_model_weights(self, model) -> int:
        from flashengine.layers.embed_head import ParallelLMHead
        from flashengine.layers.linear import LinearBase

        tn = _torch_npu()
        acl_format = _acl_format_nz()
        converted = 0
        for module in model.modules():
            if isinstance(module, ParallelLMHead):
                weight = module.weight.detach()
                if weight.ndim >= 2 and min(weight.shape[-2:]) > 1:
                    module.weight_nz = tn.npu_format_cast(weight, acl_format)
                    converted += 1
                continue
            scheme = getattr(module, "quant_scheme_name", None)
            if scheme is not None:
                # W8A8 量化线性：int8 权重转置 + NZ、尺度展开、激活尺度预计算。
                self.prepare_quant_linear(module, scheme)
                converted += 1
                continue
            if isinstance(module, LinearBase):
                weight = module.weight.data
                if (weight.dtype != torch.float32 and weight.ndim >= 2
                        and min(weight.shape[-2:]) > 1):
                    module.weight.data = tn.npu_format_cast(weight, acl_format)
                    converted += 1
        return converted

    def optimize_block_draft_weights(self, draft_model, target_lm_head) -> int:
        tn = _torch_npu()
        acl_format = _acl_format_nz()
        converted = 0
        fused = getattr(draft_model, "_fused_kv_weight", None)
        if fused is not None and fused.ndim == 2:
            draft_model._fused_kv_weight = tn.npu_format_cast(fused, acl_format)
            converted += 1
        markov = getattr(draft_model, "markov_head", None)
        if markov is not None:
            markov.markov_w2.weight_nz = tn.npu_format_cast(
                markov.markov_w2.weight, acl_format)
            converted += 1
        lm_weight = getattr(target_lm_head, "weight_nz", None)
        if draft_model.lm_head_weight is not target_lm_head.weight:
            draft_model.lm_head_weight_nz = tn.npu_format_cast(
                draft_model.lm_head_weight, acl_format)
            converted += 1
        elif lm_weight is not None:
            draft_model.lm_head_weight_nz = lm_weight
        return converted

    # ---- MoE（Qwen3-30B-A3B）算子 ----
    # 对齐 vLLM-Ascend _310p/fused_moe：gating_top_k_softmax（fp16 ND，单次 ≤1024
    # token 分块）→ mor_init_routing_v2 → grouped_matmul + swiglu → token_unpermute。
    _MOE_GATING_MAX_TOKENS = 1024

    def _require_moe_ops(self):
        if getattr(self, "_moe_ops_ok", False):
            return
        tn = _torch_npu()
        required = (
            "npu_moe_gating_top_k_softmax",
            "npu_moe_init_routing_v2",
            "npu_grouped_matmul",
            "npu_moe_token_unpermute",
        )
        missing = [op for op in required if not hasattr(tn, op)]
        if missing:
            raise RuntimeError(
                "ascend310p MoE requires torch_npu ops: " + ", ".join(missing)
                + " (upgrade torch_npu/CANN)."
            )
        self._moe_ops_ok = True

    def moe_router(self, router_logits, top_k: int, renormalize: bool):
        self._require_moe_ops()
        tn = _torch_npu()
        # npu_moe_gating_top_k_softmax 仅支持 fp16 + ND（gate 输出可能是 NZ）。
        logits = router_logits.to(torch.float16).contiguous()
        if logits.shape[0] > self._MOE_GATING_MAX_TOKENS:
            chunks = logits.split(self._MOE_GATING_MAX_TOKENS, dim=0)
            results = [tn.npu_moe_gating_top_k_softmax(c, k=top_k) for c in chunks]
            topk_weights = torch.cat([r[0] for r in results], dim=0)
            topk_ids = torch.cat([r[1] for r in results], dim=0)
        else:
            topk_weights, topk_ids, _ = tn.npu_moe_gating_top_k_softmax(logits, k=top_k)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights.to(router_logits.dtype), topk_ids.to(torch.int32)

    def fused_moe(self, hidden_states, w13_weight, w2_weight, topk_weights, topk_ids,
                  w13_scale=None, w2_scale=None):
        self._require_moe_ops()
        if w13_scale is not None:
            return self._fused_moe_w8a8(
                hidden_states, w13_weight, w2_weight, topk_weights, topk_ids,
                w13_scale, w2_scale,
            )
        tn = _torch_npu()
        hidden_states = hidden_states.contiguous()
        num_tokens = hidden_states.shape[0]
        num_experts = w13_weight.shape[0]
        top_k = topk_ids.shape[-1]
        topk_weights = topk_weights.to(hidden_states.dtype)
        topk_ids = topk_ids.to(torch.int32)
        # 专家权重已在 optimize_moe_weights 里预转置为 [E, H, N] + NZ，故 gmm 不再转置。
        sorted_hs, expanded_row_idx, expert_tokens, _ = tn.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=num_experts,
            drop_pad_mode=0,
            active_expert_range=[0, num_experts],
            quant_mode=-1,
            row_idx_type=0,
        )
        group_list = expert_tokens.to(torch.int64)
        gate_up = tn.npu_grouped_matmul(
            x=[sorted_hs], weight=[w13_weight], split_item=2,
            group_list_type=0, group_type=0, group_list=group_list)[0]
        gate_up = tn.npu_swiglu(gate_up)
        projected = tn.npu_grouped_matmul(
            x=[gate_up], weight=[w2_weight], split_item=2,
            group_list_type=0, group_type=0, group_list=group_list)[0]
        return tn.npu_moe_token_unpermute(
            permuted_tokens=projected,
            sorted_indices=expanded_row_idx.abs(),
            probs=topk_weights,
        )

    def _fused_moe_w8a8(self, hidden_states, w13_weight, w2_weight,
                        topk_weights, topk_ids, w13_scale, w2_scale):
        """W8A8_DYNAMIC 专家前向：量化 grouped matmul + dequant（pertoken 激活量化）。"""
        tn = _torch_npu()
        if not hasattr(tn, "npu_quant_grouped_matmul_dequant"):
            raise RuntimeError(
                "W8A8 MoE requires torch_npu.npu_quant_grouped_matmul_dequant "
                "(upgrade torch_npu/CANN)."
            )
        hidden_states = hidden_states.contiguous()
        num_tokens = hidden_states.shape[0]
        num_experts = w13_weight.shape[0]
        top_k = topk_ids.shape[-1]
        topk_weights = topk_weights.to(hidden_states.dtype)
        topk_ids = topk_ids.to(torch.int32)
        sorted_hs, expanded_row_idx, expert_tokens, _ = tn.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=num_experts,
            drop_pad_mode=0,
            active_expert_range=[0, num_experts],
            quant_mode=-1,
            row_idx_type=0,
        )
        # init_routing / swiglu 可能输出 FRACTAL_NZ 内部格式；分组量化矩阵乘
        # （内部 lower 到 QuantBatchMatmulV3）要求激活为 ND——否则 token 数 >1
        # 时 tiling 直接 aicore 异常（507015）。已是 ND 时为元数据操作。
        sorted_hs = tn.npu_format_cast(sorted_hs, _acl_format_nd())
        group_list = expert_tokens.to(torch.int64)
        gate_up = tn.npu_quant_grouped_matmul_dequant(
            x=sorted_hs, quantized_weight=w13_weight, weight_scale=w13_scale,
            group_list=group_list, quant_mode="pertoken")
        gate_up = tn.npu_swiglu(gate_up)
        gate_up = tn.npu_format_cast(gate_up, _acl_format_nd())
        projected = tn.npu_quant_grouped_matmul_dequant(
            x=gate_up, quantized_weight=w2_weight, weight_scale=w2_scale,
            group_list=group_list, quant_mode="pertoken")
        return tn.npu_moe_token_unpermute(
            permuted_tokens=projected,
            sorted_indices=expanded_row_idx.abs(),
            probs=topk_weights,
        )

    def optimize_moe_weights(self, model) -> int:
        """专家权重的 310P 后处理：fp16 = transpose+NZ；W8A8 = NZ 权重 + 尺度展平。"""
        tn = _torch_npu()
        acl_format = _acl_format_nz()
        converted = 0
        for module in model.modules():
            if not getattr(module, "is_moe_experts", False):
                continue
            if getattr(module, "quantized", False):
                module.w13_weight.data = tn.npu_format_cast(module.w13_weight.data, acl_format)
                module.w2_weight.data = tn.npu_format_cast(module.w2_weight.data, acl_format)
                module.w13_weight_scale.data = module.w13_weight_scale.data.reshape(
                    module.w13_weight_scale.shape[0], -1)
                module.w2_weight_scale.data = module.w2_weight_scale.data.reshape(
                    module.w2_weight_scale.shape[0], -1)
                module.w13_weight_offset.data = module.w13_weight_offset.data.reshape(
                    module.w13_weight_offset.shape[0], -1)
                module.w2_weight_offset.data = module.w2_weight_offset.data.reshape(
                    module.w2_weight_offset.shape[0], -1)
            else:
                w13 = module.w13_weight.data.transpose(1, 2).contiguous()
                module.w13_weight.data = tn.npu_format_cast(w13, acl_format)
                w2 = module.w2_weight.data.transpose(1, 2).contiguous()
                module.w2_weight.data = tn.npu_format_cast(w2, acl_format)
            converted += 2
        return converted

    # ---- W8A8 int8 量化（对齐 vLLM-Ascend 310P w8a8_static / w8a8_dynamic）----
    def prepare_quant_linear(self, module, scheme: str) -> None:
        tn = _torch_npu()
        acl_format = _acl_format_nz()
        if scheme == "W8A8":
            expanding = module.weight.data.shape[1]
            module.aclnn_input_scale = module.input_scale.data.repeat(expanding)
            module.aclnn_input_scale_reciprocal = 1.0 / module.aclnn_input_scale
            module.aclnn_input_offset = module.input_offset.data.repeat(expanding).to(
                module.aclnn_input_scale.dtype)
            module.weight.data = tn.npu_format_cast(
                module.weight.data, acl_format).transpose(0, 1)
            module.weight_scale.data = module.weight_scale.data.flatten()
            module.weight_offset.data = module.weight_offset.data.flatten()
        else:
            module.weight.data = tn.npu_format_cast(
                module.weight.data, acl_format).transpose(0, 1)
            module.weight_scale.data = module.weight_scale.data.flatten()
            module.weight_offset.data = module.weight_offset.data.flatten()

    def quant_linear_static(self, x, weight, deq_scale, quant_bias,
                            input_scale, input_scale_reciprocal, input_offset,
                            out_dtype, bias):
        tn = _torch_npu()
        # 前置融合算子（rms_norm/add_rms_norm 等）可能输出 FRACTAL_NZ 内部格式，
        # 而 npu_quant_matmul 的 x1 必须是 ND——否则图捕获/多批解码时
        # QuantBatchMatmulV3 tiling 直接 aicore 异常（507015，异步不可恢复）。
        # 已是 ND 时 npu_format_cast 为元数据操作。
        x = tn.npu_format_cast(x, _acl_format_nd())
        if x.dtype != torch.int8:
            x = tn.npu_quantize(
                x.contiguous(), input_scale_reciprocal, input_offset,
                torch.qint8, -1, False)
            # npu_quantize 的输出格式在图捕获期可能被判为 NZ 内部格式，而
            # npu_quant_matmul 的 x1 必须 ND（QuantBatchMatmulV3 tiling 拒绝）。
            x = tn.npu_format_cast(x, _acl_format_nd())
        return tn.npu_quant_matmul(
            x, weight, deq_scale, bias=quant_bias, output_dtype=out_dtype)

    def quant_linear_dynamic(self, x, weight, weight_scale, out_dtype, bias):
        tn = _torch_npu()
        # 同上：动态量化路径的激活同样须为 ND。
        x = tn.npu_format_cast(x, _acl_format_nd())
        quantized_x, pertoken_scale = tn.npu_dynamic_quant(x.contiguous())
        quantized_x = tn.npu_format_cast(quantized_x, _acl_format_nd())
        squeeze = False
        if pertoken_scale.dim() == 2:
            squeeze = True
            quantized_x = quantized_x.squeeze(dim=1)
            pertoken_scale = pertoken_scale.squeeze(dim=1)
        out = tn.npu_quant_matmul(
            quantized_x, weight, weight_scale,
            pertoken_scale=pertoken_scale, bias=bias, output_dtype=out_dtype)
        if squeeze:
            out = out.unsqueeze(dim=1)
        return out

    def _dflash_noncausal_mask(self, query_tokens: int, max_seq_len: int, device):
        cache = self._dflash_mask
        key = (query_tokens, max_seq_len, str(device), self._splitfuse_v2 is not None)
        mask = _lru_get(cache, key)
        if mask is not None:
            return mask
        if self._splitfuse_v2 is not None:
            mask = torch.zeros(
                (_COMPRESSED_MASK_SEQ_LEN, _COMPRESSED_MASK_SEQ_LEN),
                dtype=torch.float16,
                device=device,
            )
        else:
            mask_nd = torch.zeros(
                (query_tokens, max_seq_len), dtype=torch.float16, device=device
            )
            mask = self._nd_to_nz_spec(mask_nd)
        _lru_put(cache, key, mask, _DFLASH_MASK_CACHE_BYTES)
        return mask

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
        if not self.supports_dflash_paged:
            raise RuntimeError("310P DFlash paged attention requires SplitFuse")
        tn = _torch_npu()
        num_reqs = context_lens.numel()
        query_tokens, num_heads, head_dim = q.shape
        hidden = k_cache.shape[1] * 16
        num_kv_heads = hidden // head_dim
        out = self._get_decode_out(query_tokens, num_heads, head_dim, q.dtype, q.device)
        qlens = getattr(self, "_dflash_qlens", None)
        if qlens is None:
            self._dflash_qlens = {}
            qlens = self._dflash_qlens
        qkey = (num_reqs, rows)
        qlens_cpu = qlens.get(qkey)
        if qlens_cpu is None:
            qlens_cpu = torch.full(
                (num_reqs,), rows, dtype=torch.int32, device="cpu", pin_memory=True
            )
            qlens[qkey] = qlens_cpu
        if block_tables.dtype != torch.int32 or context_lens.dtype != torch.int32:
            raise TypeError(
                "310P DFlash graph metadata must be preallocated as int32 "
                f"(block_tables={block_tables.dtype}, context_lens={context_lens.dtype})"
            )
        if not block_tables.is_contiguous() or not context_lens.is_contiguous():
            raise ValueError("310P DFlash graph metadata must be contiguous")
        mask = self._dflash_noncausal_mask(
            query_tokens, max_seq_len, q.device
        )
        if self._splitfuse_v2 is not None:
            self._splitfuse_v2(
                query=q,
                key_cache=k_cache,
                value_cache=v_cache,
                mask=mask,
                block_table=block_tables,
                seq_len=qlens_cpu,
                context_lens=context_lens,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale_value=scale,
                mask_type=_MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION,
                out=out,
            )
        else:
            self._splitfuse_legacy(
                query=q,
                key_cache=k_cache,
                value_cache=v_cache,
                mask=mask,
                block_table=block_tables,
                seq_len=qlens_cpu,
                context_lens=context_lens,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale_value=scale,
                out=out,
            )
        return out

    def _setup_runtime(self):
        npu_config = getattr(torch.npu, "config", None)
        if npu_config is not None and hasattr(npu_config, "allow_internal_format"):
            try:
                # FLASHENGINE_310P_INTERNAL_FORMAT=0 关闭内部格式传播（默认开）。
                # w8a8 MoE 在 310P 上曾因算子内部选 NZ 输入触发 QuantBatchMatmulV3
                # tiling 的 aicore 异常；该开关用于验证/规避。
                enabled = os.getenv("FLASHENGINE_310P_INTERNAL_FORMAT", "1") != "0"
                npu_config.allow_internal_format = enabled
            except Exception as exc:
                print(f"[FlashEngine] 310P: allow_internal_format failed: {exc}", flush=True)
        set_compile_mode = getattr(torch.npu, "set_compile_mode", None)
        if set_compile_mode is not None:
            try:
                set_compile_mode(jit_compile=False)
            except Exception as exc:
                print(f"[FlashEngine] 310P: set_compile_mode failed: {exc}", flush=True)
        set_option = getattr(torch.npu, "set_option", None)
        if set_option is not None:
            try:
                set_option({"NPU_FUZZY_COMPILE_BLACKLIST": "ReduceNansum"})
            except Exception as exc:
                print(f"[FlashEngine] 310P: set_option failed: {exc}", flush=True)

    def set_device(self, rank):
        torch.npu.set_device(rank)

    def synchronize(self):
        torch.npu.synchronize()

    def empty_cache(self):
        torch.npu.empty_cache()

    def shutdown(self):
        try:
            self.synchronize()
        finally:
            self.empty_cache()
        try:
            import torch_npu
            npu_shutdown = getattr(torch_npu, "_npu_shutdown", None)
            if callable(npu_shutdown):
                npu_shutdown()
            else:
                print(
                    "[FlashEngine] Warning: torch_npu._npu_shutdown is unavailable; "
                    "the NPU runtime will be released by its interpreter-exit hook.",
                    flush=True,
                )
        except Exception as exc:
            print(f"[FlashEngine] Warning: NPU runtime shutdown failed: {exc}", flush=True)

    def mem_get_info(self):
        # 310P uses unified (host) memory: torch.npu.mem_get_info() returns
        # MemFree, which excludes reclaimable page cache, so KV budgeting sees
        # a bogus "used" and fails. Use MemAvailable instead.
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k.strip()] = int(v.split()[0]) * 1024
            return info["MemAvailable"], info["MemTotal"]
        except Exception:
            return torch.npu.mem_get_info()

    def memory_stats(self):
        return torch.npu.memory_stats()

    def reset_peak_memory_stats(self):
        torch.npu.reset_peak_memory_stats()

    def new_graph(self):
        return torch.npu.NPUGraph()

    def create_decode_graph_executor(self, plan):
        from flashengine.backends.ascend310p.graph import NPUGraphDecodeRunner
        return NPUGraphDecodeRunner(plan)

    def prefix_cache_disable_reason(self) -> str | None:
        if not self.supports_splitfuse:
            return "the runtime lacks paged-attention SplitFuse"
        return None

    def dense_noncausal_attention(self, q, k, v, mask, scale: float):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(probs, v)

    @contextmanager
    def graph_ctx(self, graph, pool=None):
        kwargs = {}
        if pool is not None:
            kwargs["pool"] = pool
        with torch.npu.graph(graph, **kwargs):
            yield

    def graph_pool(self, graph):
        pool = getattr(graph, "pool", None)
        return pool() if callable(pool) else None

    def graph_pad_block(self, num_blocks: int) -> int:
        return num_blocks - 1

    def graph_pad_context_len(self) -> int:
        return 1

    def reserve_dummy_kv_block(self) -> bool:
        return True

    def graph_warmup_iters(self) -> int:
        return 3

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
        tn = _torch_npu()
        hidden = num_kv_heads * head_dim
        assert dtype == torch.float16, f"310P KV fp16 only, got {dtype}"
        assert head_dim % 16 == 0, f"310P NZ needs head_dim % 16 == 0, got {head_dim}"
        assert hidden % 16 == 0, f"310P NZ needs hidden % 16 == 0, got {hidden}"
        assert block_size % 16 == 0 and block_size <= 128, (
            f"310P paged attention needs block_size in {{64,128}}, got {block_size}"
        )
        assert block_size * head_dim <= 128 * 128, (
            f"310P paged attention needs block_size * head_dim <= 128*128, "
            f"got {block_size} * {head_dim}"
        )
        acl_format = _acl_format_nz()
        nz_shape = (num_blocks, hidden // 16, block_size, 16)
        layers = []
        for _ in range(num_layers):
            k_cache = tn.empty_with_format(
                size=nz_shape, dtype=dtype, device=device, acl_format=acl_format
            )
            v_cache = tn.empty_with_format(
                size=nz_shape, dtype=dtype, device=device, acl_format=acl_format
            )
            k_cache.zero_()
            v_cache.zero_()
            layers.append((k_cache, v_cache))
        return layers

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping):
        tn = _torch_npu()
        slots = slot_mapping.reshape(-1).to(dtype=torch.int32).contiguous()
        tn._npu_reshape_and_cache(
            key=key.contiguous(),
            value=value.contiguous(),
            key_cache=k_cache,
            value_cache=v_cache,
            slot_indices=slots,
        )

    def prefill_attention(self, q, k, v, context, scale):
        seq_lens = context.host_seq_lens_q
        if seq_lens is None:
            seq_lens = (context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]).to(
                device="cpu", dtype=torch.int32
            )
        seq_lens = seq_lens.to(device="cpu", dtype=torch.int32).contiguous()
        real_tokens = q.shape[0]
        num_heads, head_dim = q.shape[-2], q.shape[-1]
        num_kv_heads = k.shape[-2]
        aligned = (real_tokens + 15) // 16 * 16
        if aligned != real_tokens:
            pad = aligned - real_tokens
            q_a = torch.zeros((aligned, num_heads, head_dim), dtype=q.dtype, device=q.device)
            k_a = torch.zeros((aligned, num_kv_heads, head_dim), dtype=k.dtype, device=k.device)
            v_a = torch.zeros((aligned, num_kv_heads, head_dim), dtype=v.dtype, device=v.device)
            q_a[:real_tokens].copy_(q)
            k_a[:real_tokens].copy_(k)
            v_a[:real_tokens].copy_(v)
            q, k, v = q_a, k_a, v_a
            seq_lens = seq_lens.clone()
            seq_lens[-1] += pad

        flash_op = os.getenv("FLASHENGINE_310P_FLASH_OP", "auto").lower()
        if flash_op not in {"auto", "v3", "atb_v2", "v2", "legacy"}:
            raise ValueError(
                "FLASHENGINE_310P_FLASH_OP must be auto/v3/atb_v2/v2/legacy, got "
                + flash_op
            )
        out = torch.empty((q.shape[0], num_heads, head_dim), dtype=q.dtype, device=q.device)
        if flash_op == "v3" or (flash_op == "auto" and self._compressed_mask_supported):
            self._flash_v3(q, k, v, seq_lens, scale, num_heads, num_kv_heads, out)
        elif flash_op in {"atb_v2", "v2"}:
            self._flash_v2(q, k, v, seq_lens, scale, num_heads, num_kv_heads, out)
        else:
            self._flash_legacy(q, k, v, seq_lens, context, scale, num_heads, num_kv_heads, out)
        return out[:real_tokens]

    def _forward_meta(self, context, kind, build):
        scratch = context.attn_scratch
        if scratch is None:
            scratch = context.attn_scratch = {}
        item = scratch.get(kind)
        if item is None:
            item = scratch[kind] = build()
        return item

    def paged_context_attention(self, q, k_cache, v_cache, context, scale):
        cl = context.context_lens
        bt = context.block_tables
        if (cl is not None and bt is not None
                and cl.shape[0] == q.shape[0] and bt.shape[0] == q.shape[0]):
            if cl.dtype != torch.int32:
                cl = cl.to(dtype=torch.int32)
            return self._run_paged_attention(q, k_cache, v_cache, bt, cl.contiguous(), scale)
        if self.supports_splitfuse:
            context.splitfuse_used = True
            return self._run_splitfuse(q, k_cache, v_cache, context, scale)
        block_table, context_lens = self._forward_meta(
            context, "expanded_pa_meta",
            lambda: self._build_expanded_pa_meta(context, q))
        return self._run_paged_attention(q, k_cache, v_cache, block_table, context_lens, scale)

    def _build_expanded_pa_meta(self, context, q):
        bt = context.block_tables
        n = bt.shape[0]
        qlens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
        if q.shape[0] % n or not bool(torch.all(qlens == qlens[0]).item()):
            raise RuntimeError(
                "310P variable-length paged prefill requires "
                "_npu_paged_attention_splitfuse(_v2)"
            )
        rows = int(qlens[0].item())
        seq_ids, j = self._paged_expand_ids(n, rows, q.device)
        kv_lens = (context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]).to(dtype=torch.int32)
        context_lens = (kv_lens[seq_ids] - rows + j + 1).to(torch.int32).clamp_min_(1).contiguous()
        return bt.index_select(0, seq_ids).contiguous(), context_lens

    def _run_splitfuse(self, q, k_cache, v_cache, context, scale):
        qlens, context_lens, block_table = self._forward_meta(
            context, "splitfuse_meta",
            lambda: self._build_splitfuse_meta(context, q.device))
        _, num_heads, head_dim = q.shape
        hidden = k_cache.shape[1] * 16
        num_kv_heads = hidden // head_dim
        out = self._get_decode_out(q.shape[0], num_heads, head_dim, q.dtype, q.device)

        if self._splitfuse_v2 is not None:
            self._splitfuse_v2(
                query=q,
                key_cache=k_cache,
                value_cache=v_cache,
                mask=self._compressed_splitfuse_mask(q.device),
                block_table=block_table,
                seq_len=qlens,
                context_lens=context_lens,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale_value=scale,
                mask_type=_MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION,
                out=out,
            )
            return out

        mask = self._forward_meta(
            context, "splitfuse_legacy_mask",
            lambda: self._build_splitfuse_legacy_mask(qlens, context_lens))
        self._splitfuse_legacy(
            query=q,
            key_cache=k_cache,
            value_cache=v_cache,
            mask=mask,
            block_table=block_table,
            seq_len=qlens,
            context_lens=context_lens,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            out=out,
        )
        return out

    def _build_splitfuse_meta(self, context, device):
        qlens = (context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]).to(
            device="cpu", dtype=torch.int32
        ).contiguous()
        context_lens = (context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]).to(
            device=device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        return qlens, context_lens, context.block_tables.to(dtype=torch.int32).contiguous()

    def _build_splitfuse_legacy_mask(self, qlens, context_lens):
        max_seq_len = max(int(context_lens.max().item()), 1)
        qlens_list = qlens.tolist()
        context_lens_list = context_lens.to(device="cpu").tolist()
        positions = [
            pos
            for qlen, context_len in zip(qlens_list, context_lens_list)
            for pos in range(context_len - qlen, context_len)
        ]
        position = torch.tensor(positions, dtype=torch.int32, device=context_lens.device)
        mask_nd = self._causal_mask_nd(max_seq_len, context_lens.device).index_select(0, position)
        return self._nd_to_nz_spec(mask_nd)

    def _paged_expand_ids(self, n, rows, device):
        cache = getattr(self, '_pa_expand', None)
        if cache is None:
            self._pa_expand = {}
            cache = self._pa_expand
        key = (n, rows, str(device))
        hit = cache.get(key)
        if hit is None:
            seq_ids = torch.arange(n, device=device, dtype=torch.int64).repeat_interleave(rows)
            j = torch.arange(rows, device=device, dtype=torch.int32).repeat(n)
            hit = (seq_ids, j)
            cache[key] = hit
        return hit

    def decode_attention(self, q, k_cache, v_cache, context, scale):
        out = self._run_paged_attention(
            q, k_cache, v_cache, context.block_tables, context.context_lens, scale,
        )
        return out.unsqueeze(1)

    def _run_paged_attention(self, q, k_cache, v_cache, block_table, context_lens, scale):
        tn = _torch_npu()
        bs, num_heads, head_dim = q.shape
        hidden = k_cache.shape[1] * 16
        num_kv_heads = hidden // head_dim
        out = self._get_decode_out(bs, num_heads, head_dim, q.dtype, q.device)
        from flashengine.backends.ascend310p.aclgraph_attn import (
            capture_paged_attention,
            is_attn_capturing,
            supports_pa_graph_update,
        )
        if is_attn_capturing() and supports_pa_graph_update():
            capture_paged_attention(
                query=q,
                key_cache=k_cache,
                value_cache=v_cache,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale=scale,
                block_table=block_table,
                context_lens=context_lens,
                out=out,
            )
            return out
        tn._npu_paged_attention(
            query=q,
            key_cache=k_cache,
            value_cache=v_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            block_table=block_table,
            context_lens=context_lens,
            out=out,
        )
        return out

    def _get_decode_out(self, bs, num_heads, head_dim, dtype, device):
        key = (bs, num_heads, head_dim, dtype, str(device))
        buf = self._decode_out.get(key)
        if buf is None:
            buf = torch.empty((bs, num_heads, head_dim), dtype=dtype, device=device)
            self._decode_out[key] = buf
        return buf

    def _causal_mask_nz(self, max_seq_len, device):
        seq_len = _bucket_mask_len(max_seq_len)
        cached = _lru_get(self._nz_mask, seq_len)
        if cached is not None:
            return cached
        tn = _torch_npu()
        aligned = (seq_len + 15) // 16 * 16
        mask_nd = torch.zeros((aligned, aligned), dtype=torch.float16, device=device)
        upper = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
        mask_nd[:seq_len, :seq_len].masked_fill_(upper, float("-inf"))
        mask = mask_nd.reshape(1, aligned, aligned // 16, 16).permute(0, 2, 1, 3).contiguous()
        mask = tn.npu_format_cast(mask, _acl_format_nz())
        _lru_put(self._nz_mask, seq_len, mask, _MASK_CACHE_BYTES)
        return mask

    def _causal_mask_nd(self, max_seq_len, device):
        seq_len = _bucket_mask_len(max_seq_len)
        cached = _lru_get(self._nd_mask, seq_len)
        if cached is not None:
            return cached
        mask = torch.zeros((seq_len, seq_len), dtype=torch.float16, device=device)
        upper = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
        mask.masked_fill_(upper, float("-inf"))
        _lru_put(self._nd_mask, seq_len, mask, _MASK_CACHE_BYTES)
        return mask

    def _compressed_splitfuse_mask(self, device):
        key = str(device)
        cached = self._splitfuse_mask.get(key)
        if cached is None:
            cached = torch.triu(
                torch.ones(
                    (_COMPRESSED_MASK_SEQ_LEN, _COMPRESSED_MASK_SEQ_LEN),
                    dtype=torch.float16,
                    device=device,
                ),
                diagonal=1,
            ).mul_(_PAGED_ATTENTION_COMPRESSED_MASK_VALUE)
            self._splitfuse_mask[key] = cached
        return cached

    @staticmethod
    def _nd_to_nz_spec(mask):
        rows, cols = mask.shape
        rows_aligned = (rows + 15) // 16 * 16
        cols_aligned = (cols + 15) // 16 * 16
        padded = torch.zeros(
            (1, rows_aligned, cols_aligned), dtype=mask.dtype, device=mask.device
        )
        padded[0, :rows, :cols].copy_(mask)
        nz = padded.reshape(1, rows_aligned, cols_aligned // 16, 16)
        nz = nz.permute(0, 2, 1, 3).contiguous()
        return _torch_npu().npu_format_cast(nz, _acl_format_nz())

    def _flash_legacy(self, q, k, v, seq_lens, context, scale, num_heads, num_kv_heads, out):
        tn = _torch_npu()
        max_seq_len = max(
            int(context.max_context_len or 0),
            int(context.max_seqlen_k or 0),
            int(seq_lens.max().item()),
        )
        tn._npu_flash_attention(
            query=q,
            key=k,
            value=v,
            mask=self._causal_mask_nz(max_seq_len, q.device),
            seq_len=seq_lens,
            scale_value=scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            out=out,
        )

    def _flash_v3(self, q, k, v, seq_lens, scale, num_heads, num_kv_heads, out):
        if not self._compressed_mask_supported:
            raise RuntimeError(
                "FLASHENGINE_310P_FLASH_OP=v3 requires "
                "_npu_flash_attention_v3 and _npu_paged_attention_splitfuse_v2"
            )
        _torch_npu()._npu_flash_attention_v3(
            query=q,
            key=k,
            value=v,
            mask=self._causal_mask_nz(_COMPRESSED_MASK_SEQ_LEN, q.device),
            seq_len=seq_lens,
            scale_value=scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mask_type=_MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION,
            out=out,
        )

    def _flash_v2(self, q, k, v, seq_lens, scale, num_heads, num_kv_heads, out):
        tn = _torch_npu()
        atb = getattr(tn, "atb", None)
        flash_v2 = getattr(atb, "_npu_flash_attention_v2", None) if atb is not None else None
        if flash_v2 is None:
            raise RuntimeError(
                "FLASHENGINE_310P_FLASH_OP=atb_v2 but torch_npu.atb._npu_flash_attention_v2 missing"
            )
        mask_len = max(int(seq_lens.max().item()), 1)
        try:
            mask = self._causal_mask_nz(mask_len, q.device)
        except Exception:
            mask = self._causal_mask_nd(mask_len, q.device)
        flash_v2(
            q, k, v, seq_lens.tolist(),
            mask=mask,
            kernel_type=0,
            mask_type=1,
            scale_value=scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            out=out,
        )

    @staticmethod
    def rms_norm(x, weight, eps):
        x, _ = _torch_npu().npu_rms_norm(x, weight, eps)
        return x

    @staticmethod
    def add_rms_norm(x, residual, weight, eps):
        x, _, residual = _torch_npu().npu_add_rms_norm(x, residual, weight, eps)
        return x, residual

    @staticmethod
    def silu_and_mul(x):
        if x.shape[-1] % 32 == 0:
            return _torch_npu().npu_swiglu(x)
        h = x.shape[-1] // 2
        return F.silu(x[..., :h]) * x[..., h:]

    def rope(self, positions, q, k, cos_sin_cache):
        tn = _torch_npu()
        apply_op = getattr(tn, "npu_apply_rotary_pos_emb", None)
        rope_op = getattr(tn, "_npu_rotary_embedding", None)
        if apply_op is None and rope_op is None:
            return BackendBase.rope(positions, q, k, cos_sin_cache)

        q_shape, k_shape = q.shape, k.shape
        rotary_dim = q_shape[-1]
        cache = cos_sin_cache
        if cache.device != q.device or cache.dtype != q.dtype:
            cache_key = (cache.data_ptr(), str(q.device), q.dtype)
            converted = self._rope_cache.get(cache_key)
            if converted is None:
                converted = cache.to(device=q.device, dtype=q.dtype)
                self._rope_cache[cache_key] = converted
            cache = converted

        if apply_op is not None and rotary_dim in (64, 128):
            from flashengine.utils.context import get_context

            context = get_context()
            num_tokens = q_shape[0]
            forward_key = (
                cache.data_ptr(), positions.data_ptr(), num_tokens,
                rotary_dim, str(q.device), q.dtype,
            )
            if context.rope_cache_key != forward_key:
                cos_sin = cache.index_select(0, positions)
                cos_half, sin_half = cos_sin.chunk(2, dim=-1)
                cos = torch.cat((cos_half, cos_half), dim=-1)
                sin = torch.cat((sin_half, sin_half), dim=-1)
                buffer_key = (
                    cache.data_ptr(), num_tokens, rotary_dim,
                    str(q.device), q.dtype,
                )
                buffers = _lru_get(self._rope_apply_cache, buffer_key)
                if buffers is None:
                    buffers = (
                        torch.empty(
                            (1, num_tokens, 1, rotary_dim),
                            dtype=q.dtype, device=q.device,
                        ),
                        torch.empty(
                            (1, num_tokens, 1, rotary_dim),
                            dtype=q.dtype, device=q.device,
                        ),
                    )
                    _lru_put(self._rope_apply_cache, buffer_key, buffers,
                             _ROPE_BUFFER_CACHE_BYTES)
                context.rope_cos, context.rope_sin = buffers
                context.rope_cos.copy_(cos.contiguous().view_as(context.rope_cos))
                context.rope_sin.copy_(sin.contiguous().view_as(context.rope_sin))
                context.rope_cache_key = forward_key
            q4 = q.contiguous().view(1, num_tokens, -1, rotary_dim)
            k4 = k.contiguous().view(1, num_tokens, -1, rotary_dim)
            q4, k4 = apply_op(
                q4, k4, context.rope_cos, context.rope_sin, rotary_mode="half"
            )
            return q4.view(q_shape), k4.view(k_shape)

        if rope_op is None:
            return BackendBase.rope(positions, q, k, cos_sin_cache)
        q = q.contiguous().view(q_shape[0], -1)
        k = k.contiguous().view(k_shape[0], -1)
        rope_op(
            positions.contiguous(),
            q,
            k,
            rotary_dim,
            cache,
            True,
        )
        return q.view(q_shape), k.view(k_shape)
