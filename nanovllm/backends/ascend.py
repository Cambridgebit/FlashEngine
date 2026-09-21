# Ascend 310P（Atlas 300I Duo / 200I Pro）后端。
#
# 硬件约束：仅 fp16、不支持 triton、block_size ∈ {64,128}、
# block_size * head_dim <= 128*128。算子路径对齐 nano-vllm-ascend 实机：
#   KV      = FRACTAL_NZ 4D (blocks, hidden/16, block_size, 16)
#   store   = _npu_reshape_and_cache(..., slot_indices=)
#   Prefill = _npu_flash_attention + 16-aligned tokens + NZ causal mask
#   Decode  = _npu_paged_attention (static out)
#   Spec verify/refill = multi-query PA on NZ cache (whole-round NPUGraph)
#   DFlash draft = private paged NZ KV + non-causal SplitFuse
# 图 padding 用末块 dummy（block=N-1、ctx=1），不用 -1。
#
# torch_npu 懒加载：模块顶部只 import torch。

import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .base import BackendBase

_npu = None

_COMPRESSED_MASK_SEQ_LEN = 2048
_MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION = 3
_MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION = 5
_PAGED_ATTENTION_COMPRESSED_MASK_VALUE = -10000.0

_REQUIRED_OPS = (
    "_npu_reshape_and_cache",
    "_npu_paged_attention",
    "_npu_flash_attention",
    "npu_rms_norm",
    "npu_add_rms_norm",
    "npu_swiglu",
)


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


class Ascend310PBackend(BackendBase):

    name = "ascend310p"
    dist_name = "hccl"
    supports_spec = True
    supports_spec_graph = True
    supports_prefix_cache_attn = False
    supports_paged_context_attn = True
    supports_splitfuse = False
    device_name = "npu"
    default_dtype = torch.float16
    graph_sample_default = False
    # 310P spec 图：host 预展开 PA meta + 图内 greedy argmax（graph_sample=False）
    # + float/static 验收（禁 int Add/arange/Gumbel）。默认尝试；冒烟失败降级。
    # NANOVLLM_SPEC_GRAPH=0 关闭尝试。
    spec_graph_try_default = True

    def __init__(self):
        tn = _torch_npu()
        self._setup_runtime()
        self._nz_mask = {}
        self._nd_mask = {}
        self._splitfuse_mask = {}
        self._decode_out = {}
        self._rope_cache = {}
        self._rope_apply_cache = {}
        self._splitfuse_v2 = getattr(tn, "_npu_paged_attention_splitfuse_v2", None)
        self._splitfuse_legacy = getattr(tn, "_npu_paged_attention_splitfuse", None)
        self.supports_splitfuse = (
            self._splitfuse_v2 is not None or self._splitfuse_legacy is not None
        )
        dflash_paged_flag = os.getenv(
            "NANOVLLM_310P_DFLASH_PAGED", "auto").strip().lower()
        if dflash_paged_flag not in {"auto", "1", "true", "on", "0", "false", "off"}:
            raise ValueError(
                "NANOVLLM_310P_DFLASH_PAGED must be auto/1/0, got "
                + dflash_paged_flag
            )
        force_paged = dflash_paged_flag in {"1", "true", "on"}
        disable_paged = dflash_paged_flag in {"0", "false", "off"}
        if force_paged and not self.supports_splitfuse:
            raise RuntimeError(
                "NANOVLLM_310P_DFLASH_PAGED=1 requires "
                "_npu_paged_attention_splitfuse(_v2)"
            )
        self.supports_dflash_paged = self.supports_splitfuse and not disable_paged
        # These operators shipped together with the compressed-mask ABI.
        # Checking both avoids selecting an incompatible intermediate build.
        self._compressed_mask_supported = (
            hasattr(tn, "_npu_flash_attention_v3")
            and hasattr(tn, "_npu_paged_attention_splitfuse_v2")
        )
        requested_flash = os.getenv("NANOVLLM_310P_FLASH_OP", "auto").lower()
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
                "Linear/LM-head weights=FRACTAL_NZ"
            ),
            "dflash_backend": (
                f"paged-KV/SplitFuse-{splitfuse}" if self.supports_dflash_paged
                else "dense-KV/SDPA-fallback"
            ),
        }

    def runtime_info(self) -> dict[str, str]:
        return dict(self._runtime_info)

    def optimize_model_weights(self, model) -> int:
        """Match vLLM-Ascend's mandatory 310P FRACTAL_NZ weight policy.

        Regular linear weights can replace their storage in place because
        ``F.linear`` preserves the logical two-dimensional shape.  The LM head
        keeps a separate NZ view: its parameter may be tied to the token
        embedding, and ``F.embedding`` still needs the original ND tensor.
        """
        from nanovllm.layers.embed_head import ParallelLMHead
        from nanovllm.layers.linear import LinearBase

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
            if isinstance(module, LinearBase):
                weight = module.weight.data
                if (weight.dtype != torch.float32 and weight.ndim >= 2
                        and min(weight.shape[-2:]) > 1):
                    module.weight.data = tn.npu_format_cast(weight, acl_format)
                    converted += 1
        return converted

    def optimize_dflash_weights(self, draft_model, target_lm_head) -> int:
        """Convert DFlash's non-module fused/shared GEMM weights to FRACTAL_NZ.

        ``optimize_model_weights`` sees regular Linear modules, but DFlash also
        owns a concatenated all-layer context-KV projection and directly shares
        the target LM-head tensor. Both are hot-path GEMMs on 310P.
        """
        tn = _torch_npu()
        acl_format = _acl_format_nz()
        converted = 0
        fused = getattr(draft_model, "_fused_kv_weight", None)
        if fused is not None and fused.ndim == 2:
            draft_model._fused_kv_weight = tn.npu_format_cast(fused, acl_format)
            converted += 1
        lm_weight = getattr(target_lm_head, "weight_nz", None)
        if lm_weight is not None:
            # ``lm_head_weight`` was constructed from the target Parameter, so
            # nn.Module registered that name in ``_parameters``. Replacing it
            # with a plain NPU tensor raises TypeError. Keep the shared ND
            # Parameter intact for embedding/checkpoint semantics and expose
            # the preconverted NZ tensor through a separate, non-parameter
            # runtime attribute used only by the DFlash projection.
            draft_model.lm_head_weight_nz = lm_weight
        return converted

    def _dflash_noncausal_mask(self, query_tokens: int, max_seq_len: int, device):
        cache = getattr(self, "_dflash_mask", None)
        if cache is None:
            self._dflash_mask = {}
            cache = self._dflash_mask
        key = (query_tokens, max_seq_len, str(device), self._splitfuse_v2 is not None)
        mask = cache.get(key)
        if mask is not None:
            return mask
        if self._splitfuse_v2 is not None:
            # Compressed-mask ABI consumes a fixed ND matrix. All zeros means
            # every query row can see the complete context and the whole draft
            # block, which is DFlash's non-causal block-attention contract.
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
        cache[key] = mask
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
        """310P DFlash non-causal block attention on paged FRACTAL_NZ KV.

        Query K/V have already been written with ``_npu_reshape_and_cache``.
        SplitFuse reads the shared block table once per request and evaluates
        all ``rows`` block queries in one operator, avoiding dense context
        gathers, GQA expansion, concatenation, and generic SDPA decomposition.
        """
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
                npu_config.allow_internal_format = True
            except Exception as exc:
                print(f"[nano-vllm] 310P: allow_internal_format failed: {exc}", flush=True)
        set_compile_mode = getattr(torch.npu, "set_compile_mode", None)
        if set_compile_mode is not None:
            try:
                set_compile_mode(jit_compile=False)
            except Exception as exc:
                print(f"[nano-vllm] 310P: set_compile_mode failed: {exc}", flush=True)
        set_option = getattr(torch.npu, "set_option", None)
        if set_option is not None:
            try:
                set_option({"NPU_FUZZY_COMPILE_BLACKLIST": "ReduceNansum"})
            except Exception as exc:
                print(f"[nano-vllm] 310P: set_option failed: {exc}", flush=True)

    def set_device(self, rank):
        torch.npu.set_device(rank)

    def synchronize(self):
        torch.npu.synchronize()

    def empty_cache(self):
        torch.npu.empty_cache()

    def mem_get_info(self):
        return torch.npu.mem_get_info()

    def memory_stats(self):
        return torch.npu.memory_stats()

    def reset_peak_memory_stats(self):
        torch.npu.reset_peak_memory_stats()

    def new_graph(self):
        return torch.npu.NPUGraph()

    @contextmanager
    def graph_ctx(self, graph, pool=None):
        kwargs = {}
        if pool is not None:
            kwargs["pool"] = pool
        with torch.npu.graph(graph, **kwargs):
            yield

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
        # ATB ReshapeAndCache requires flat int32 slots. All 310P graph paths
        # pre-expand padding to the reserved final KV block, so a dynamic
        # ``where(slots < 0)`` is unnecessary and would add an integer operator
        # to every captured attention layer (notably 5x per DFlash round).
        slots = slot_mapping.reshape(-1).to(dtype=torch.int32).contiguous()
        tn._npu_reshape_and_cache(
            key=key.contiguous(),
            value=value.contiguous(),
            key_cache=k_cache,
            value_cache=v_cache,
            slot_indices=slots,
        )

    def prefill_attention(self, q, k, v, context, scale):
        seq_lens = (context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]).to(
            device="cpu", dtype=torch.int32
        ).contiguous()
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

        flash_op = os.getenv("NANOVLLM_310P_FLASH_OP", "auto").lower()
        if flash_op not in {"auto", "v3", "atb_v2", "v2", "legacy"}:
            raise ValueError(
                "NANOVLLM_310P_FLASH_OP must be auto/v3/atb_v2/v2/legacy, got "
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

    def paged_context_attention(self, q, k_cache, v_cache, context, scale):
        # spec verify/refill：优先走 host 预展开的 per-query lens/table（与 decode
        # 同构直通 PA）。设备端 index/Add 会进 AICPU，NPUGraph replay 报 507011。
        cl = context.context_lens
        bt = context.block_tables
        if (cl is not None and bt is not None
                and cl.shape[0] == q.shape[0] and bt.shape[0] == q.shape[0]):
            if cl.dtype != torch.int32:
                cl = cl.to(dtype=torch.int32)
            return self._run_paged_attention(q, k_cache, v_cache, bt, cl.contiguous(), scale)
        # Prefix-hit / chunked prefill keeps one block-table row per sequence.
        # SplitFuse consumes variable query lengths and reads the complete NZ
        # paged cache after the new K/V rows have been stored.
        if self.supports_splitfuse:
            return self._run_splitfuse(q, k_cache, v_cache, context, scale)
        # eager 回退：无 SplitFuse 时现场展开为 per-query PA。此路径只支持
        # 等长 query 段（spec verify/refill）；prefix-hit 的变长段必须禁用。
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
        block_table = bt.index_select(0, seq_ids).contiguous()
        return self._run_paged_attention(q, k_cache, v_cache, block_table, context_lens, scale)

    def _run_splitfuse(self, q, k_cache, v_cache, context, scale):
        qlens = (context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]).to(
            device="cpu", dtype=torch.int32
        ).contiguous()
        context_lens = (context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]).to(
            device=q.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        block_table = context.block_tables.to(dtype=torch.int32).contiguous()
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

        max_seq_len = max(int(context_lens.max().item()), 1)
        qlens_list = qlens.tolist()
        context_lens_list = context_lens.to(device="cpu").tolist()
        positions = [
            pos
            for qlen, context_len in zip(qlens_list, context_lens_list)
            for pos in range(context_len - qlen, context_len)
        ]
        position = torch.tensor(positions, dtype=torch.int32, device=q.device)
        mask_nd = self._causal_mask_nd(max_seq_len, q.device).index_select(0, position)
        mask = self._nd_to_nz_spec(mask_nd)
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
        from nanovllm.engine.aclgraph_attn import (
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
        cached = self._nz_mask.get(max_seq_len)
        if cached is not None:
            return cached
        tn = _torch_npu()
        aligned = (max_seq_len + 15) // 16 * 16
        mask_nd = torch.zeros((aligned, aligned), dtype=torch.float16, device=device)
        upper = torch.triu(
            torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
        mask_nd[:max_seq_len, :max_seq_len].masked_fill_(upper, float("-inf"))
        mask = mask_nd.reshape(1, aligned, aligned // 16, 16).permute(0, 2, 1, 3).contiguous()
        mask = tn.npu_format_cast(mask, _acl_format_nz())
        self._nz_mask[max_seq_len] = mask
        return mask

    def _causal_mask_nd(self, max_seq_len, device):
        cached = self._nd_mask.get(max_seq_len)
        if cached is not None:
            return cached
        mask = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16, device=device)
        upper = torch.triu(
            torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
        mask.masked_fill_(upper, float("-inf"))
        self._nd_mask[max_seq_len] = mask
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
                "NANOVLLM_310P_FLASH_OP=v3 requires "
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
                "NANOVLLM_310P_FLASH_OP=atb_v2 but torch_npu.atb._npu_flash_attention_v2 missing"
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
        """Use the 310P in-place fused RoPE kernel when the runtime exposes it."""
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

        # Official 310P fast path for the common Qwen rotary dimensions.  The
        # expensive position gather/cos-sin expansion is done only by the first
        # layer in a model forward and reused by every following layer.
        if apply_op is not None and rotary_dim in (64, 128):
            from nanovllm.utils.context import get_context

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
                buffers = self._rope_apply_cache.get(buffer_key)
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
                    self._rope_apply_cache[buffer_key] = buffers
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
