"""CUDA 上的 FlashInfer 注意力适配；与 vLLM 使用同一套公开 wrapper API。"""

from dataclasses import dataclass
from itertools import accumulate

import torch

from .graph import DecodeGraphExecutor
from flashengine.utils.context import get_context, reset_context, set_context


def paged_metadata(block_tables, lengths, page_size):
    """只压入有效物理页，末页满时长度为 page_size，而不是零。"""
    counts = [(length + page_size - 1) // page_size for length in lengths]
    indptr = torch.tensor([0, *accumulate(counts)], dtype=torch.int32, device="cpu")
    indices = torch.tensor(
        [page for table, count in zip(block_tables, counts) for page in table[:count]],
        dtype=torch.int32, device="cpu",
    )
    last = torch.tensor(
        [(length - 1) % page_size + 1 if length else 0 for length in lengths],
        dtype=torch.int32, device="cpu",
    )
    if indices.numel() != int(indptr[-1]):
        raise ValueError("FlashInfer block table 未覆盖上下文长度")
    return indptr, indices, last


class FlashInferAttention:
    def __init__(self):
        try:
            import flashinfer
        except ImportError as exc:
            raise ImportError("FlashInfer 路径需要在 CUDA 环境安装 flashinfer-python") from exc
        self.api = flashinfer
        self.workspace = None
        self.wrappers = {}

    def workspace_buffer(self):
        if self.workspace is None:
            # 在模型预热阶段分配，计入 KV 池预算；各层在同一流串行复用。
            self.workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
        return self.workspace

    def close(self):
        self.wrappers.clear()
        self.workspace = None

    def _wrapper(self, kind):
        if kind not in self.wrappers:
            cls = {
                "ragged": self.api.BatchPrefillWithRaggedKVCacheWrapper,
                "paged": self.api.BatchPrefillWithPagedKVCacheWrapper,
                "decode": self.api.BatchDecodeWithPagedKVCacheWrapper,
            }[kind]
            kwargs = {"use_tensor_cores": True} if kind == "decode" else {}
            self.wrappers[kind] = cls(self.workspace_buffer(), "NHD", backend="fa2", **kwargs)
        return self.wrappers[kind]

    def _planned(self, context, kind, q, k, scale):
        if context.attn_scratch is None:
            context.attn_scratch = {}
        key = ("flashinfer", kind, q.shape[1], k.shape[-2], q.shape[-1], q.dtype, scale)
        if key in context.attn_scratch:
            return context.attn_scratch[key]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FlashInfer plan 必须在 CUDA Graph 外更新")
        wrapper = self._wrapper(kind)
        kwargs = dict(q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=scale)
        if kind == "ragged":
            wrapper.plan(context.cu_seqlens_q.cpu(), context.cu_seqlens_k.cpu(),
                         q.shape[1], k.shape[-2], q.shape[-1], causal=True, **kwargs)
        else:
            lengths = (context.context_lens.cpu().tolist() if kind == "decode"
                       else context.cu_seqlens_k.cpu().diff().tolist())
            metadata = paged_metadata(context.block_tables.cpu().tolist(), lengths, k.shape[1])
            if kind == "decode":
                wrapper.plan(*metadata, q.shape[1], k.shape[-2], q.shape[-1], k.shape[1], **kwargs)
            else:
                wrapper.plan(context.cu_seqlens_q.cpu(), *metadata, q.shape[1],
                             k.shape[-2], q.shape[-1], k.shape[1], causal=True, **kwargs)
        context.attn_scratch[key] = wrapper
        return wrapper

    def prefill(self, q, k, v, context, scale):
        kind = "ragged" if context.block_tables is None else "paged"
        wrapper = self._planned(context, kind, q, k, scale)
        return wrapper.run(q, k, v) if kind == "ragged" else wrapper.run(q, (k, v))

    def decode(self, q, k, v, context, scale):
        wrapper = (context.attn_scratch or {}).get("flashinfer_graph_decode")
        if wrapper is None:
            wrapper = self._planned(context, "decode", q, k, scale)
        return wrapper.run(q, (k, v))


@dataclass
class _DecodeEntry:
    batch_size: int
    wrapper: object
    ids: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    graph: object = None
    logits: torch.Tensor = None
    tokens: torch.Tensor = None


class FlashInferDecodeGraphRunner(DecodeGraphExecutor):
    """每桶固定 wrapper 地址，在 replay 前从主机页表重新 plan。"""

    def __init__(self, plan, attention):
        self.plan = plan
        self.attention = attention
        self.entries = {}
        self.pool = None
        sizes = [n for n in [1, 2, 4, 8, *range(16, plan.max_batch_size + 1, 16)]
                 if n <= plan.max_batch_size]
        if not sizes or sizes[-1] != plan.max_batch_size:
            sizes.append(plan.max_batch_size)
        self.sizes = sizes
        self.stats = dict(capture_success=0, capture_failure=0, replay=0,
                          fallback=0, bucket_hits={})

    @property
    def capture_sizes(self):
        return tuple(sorted(self.entries))

    def is_available(self):
        return bool(self.entries)

    def snapshot(self):
        return {**self.stats, "bucket_hits": dict(self.stats["bucket_hits"])}

    def reset(self):
        self.stats.update(replay=0, fallback=0, bucket_hits={})

    def _plan(self, entry, tables, lengths):
        cfg = self.plan.config
        mc = cfg.model_config
        tp = cfg.tensor_parallel_size
        head_dim = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
        entry.wrapper.plan(
            *paged_metadata(tables, lengths, self.plan.block_size),
            mc.num_attention_heads // tp, mc.num_key_value_heads // tp,
            head_dim, self.plan.block_size, q_data_type=mc.dtype,
            kv_data_type=mc.dtype, sm_scale=head_dim ** -0.5,
        )

    def _bind(self, entry):
        set_context(False, slot_mapping=entry.slots, use_cuda_graph=True,
                    block_size=self.plan.block_size)
        get_context().attn_scratch = {"flashinfer_graph_decode": entry.wrapper}

    def _forward(self, entry):
        self._bind(entry)
        logits = self.plan.model.compute_logits(self.plan.model(entry.ids, entry.positions))
        return logits, logits.argmax(dim=-1)

    @torch.inference_mode()
    def capture_all(self):
        if self.plan.config.tensor_parallel_size != 1:
            self.stats["capture_failure"] = len(self.sizes)
            print("[FlashEngine] FlashInfer decode Graph 当前要求 TP=1，回退 eager。", flush=True)
            return
        workspace = self.attention.workspace_buffer()
        for bs in reversed(self.sizes):
            try:
                device = self.plan.device
                wrapper = self.attention.api.BatchDecodeWithPagedKVCacheWrapper(
                    workspace, "NHD", use_cuda_graph=True, use_tensor_cores=True,
                    paged_kv_indptr_buffer=torch.empty(bs + 1, dtype=torch.int32, device=device),
                    paged_kv_indices_buffer=torch.empty(
                        bs * self.plan.max_block_columns, dtype=torch.int32, device=device),
                    paged_kv_last_page_len_buffer=torch.empty(bs, dtype=torch.int32, device=device),
                    backend="fa2",
                )
                entry = _DecodeEntry(
                    bs, wrapper, torch.zeros(bs, dtype=torch.int64, device=device),
                    torch.zeros(bs, dtype=torch.int64, device=device),
                    torch.full((bs,), -1, dtype=torch.int32, device=device),
                )
                # 捕获与 padding 读取有效零页；slot=-1 跳过写，不覆盖真实请求。
                for module in self.plan.model.modules():
                    if hasattr(module, "k_cache") and module.k_cache.numel():
                        module.k_cache[0].zero_()
                        module.v_cache[0].zero_()
                self._plan(entry, [[0]] * bs, [1] * bs)
                for _ in range(self.plan.warmup_iters):
                    self._forward(entry)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool):
                    entry.logits, entry.tokens = self._forward(entry)
                self.pool = graph.pool()
                entry.graph = graph
                self.entries[bs] = entry
                self.stats["capture_success"] += 1
            except Exception as exc:
                self.stats["capture_failure"] += 1
                print(f"[FlashEngine] FlashInfer decode 图捕获失败 bs={bs}，回退 eager: {exc}", flush=True)
            finally:
                reset_context()

    @torch.inference_mode()
    def try_run(self, seqs):
        n = len(seqs)
        if not n:
            return None
        bs = next((size for size in sorted(self.entries) if size >= n), None)
        if bs is None:
            self.stats["fallback"] += 1
            return None
        entry = self.entries[bs]
        pad = bs - n
        lengths = [len(seq) for seq in seqs] + [1] * pad
        tables = [seq.block_table for seq in seqs] + [[0]] * pad
        slots = [seq.block_table[-1] * self.plan.block_size + seq.last_block_num_tokens - 1
                 for seq in seqs] + [-1] * pad
        # 拷贝和 plan 均在图外；plan 更新 wrapper 自有的稳定设备缓冲区。
        for target, values in (
            (entry.ids, [seq.last_token for seq in seqs] + [0] * pad),
            (entry.positions, [len(seq) - 1 for seq in seqs] + [0] * pad),
            (entry.slots, slots),
        ):
            target.copy_(torch.tensor(values, dtype=target.dtype, device="cpu"))
        try:
            self._plan(entry, tables, lengths)
        except Exception as exc:
            # 计划失败时不回放旧图，后续同桶使用正常 eager 路径。
            del self.entries[bs]
            self.stats["fallback"] += 1
            print(f"[FlashEngine] FlashInfer plan 更新失败 bs={bs}，禁用该桶: {exc}", flush=True)
            return None
        entry.graph.replay()
        self.stats["replay"] += 1
        hits = self.stats["bucket_hits"]
        hits[bs] = hits.get(bs, 0) + 1
        if all(seq.temperature <= 1e-10 for seq in seqs):
            return entry.tokens[:n], True
        return entry.logits[:n], False
