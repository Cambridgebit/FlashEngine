# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project

"""310P decode 用 NPUGraph（ACLGraph）runner，对齐 nano-vllm-ascend bench_ttft_tps --mode aclgraph。

每个 bucket 永久持有固定地址静态 Tensor；runtime 只 copy_ 内容。
Sampler 在图外；图内是 model.forward + compute_logits。
捕获失败按 bucket 回退 eager。CUDA/CoreX 不走本模块。
"""
from dataclasses import dataclass
import traceback

import torch

from nanovllm.utils.context import set_context, reset_context


def npu_graph_api():
    if hasattr(torch, "npu") and hasattr(torch.npu, "NPUGraph"):
        return torch.npu
    import torch_npu
    return torch_npu.npu


@dataclass
class DecodeGraphEntry:
    graph_bs: int
    graph: object | None = None
    input_ids: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    host_input_ids: torch.Tensor | None = None
    host_positions: torch.Tensor | None = None
    host_slot_mapping: torch.Tensor | None = None
    host_context_lens: torch.Tensor | None = None
    host_block_tables: torch.Tensor | None = None
    host_input_ids_np: object | None = None
    host_positions_np: object | None = None
    host_slot_mapping_np: object | None = None
    host_context_lens_np: object | None = None
    host_block_tables_np: object | None = None
    logits: torch.Tensor | None = None
    greedy_token_ids: torch.Tensor | None = None
    attn_captures: list | None = None
    enabled: bool = True
    capture_error: str | None = None

class NPUGraphDecodeRunner:

    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.model = model_runner.model
        self.config = model_runner.config
        self.device = model_runner.device
        self.block_size = model_runner.block_size
        max_bs = model_runner._graph_max_bs
        sizes = [x for x in ([1, 2, 4, 8] + list(range(16, max_bs + 1, 16))) if x <= max_bs]
        if not sizes or sizes[-1] != max_bs:
            sizes.append(max_bs)
        self.graph_capture_sizes = sizes
        self.max_block_cols = model_runner._pin_W
        self.warmup_iters = model_runner.backend.graph_warmup_iters()
        self.dummy_kv_block = self.config.num_kvcache_blocks - 1
        self.entries: dict[int, DecodeGraphEntry] = {}
        self._npu = npu_graph_api()
        self.graph_pool = self._create_pool_handle()
        self.capture_stream = torch.npu.Stream()
        self.update_stream = torch.npu.Stream()
        for stream in (self.capture_stream, self.update_stream):
            with torch.npu.stream(stream):
                torch.empty(1, device=self.device)
        from nanovllm.engine.aclgraph_attn import supports_pa_graph_update
        self._pa_update_ok = supports_pa_graph_update()
        self.stats = {
            "capture_success": 0,
            "capture_failure": 0,
            "replay": 0,
            "fallback": 0,
            "bucket_hits": {},
        }

    def _create_pool_handle(self):
        handle_fn = getattr(self._npu, "graph_pool_handle", None)
        if handle_fn is None:
            return None
        try:
            return handle_fn()
        except Exception as exc:
            print(f"[nano-vllm] NPUGraph graph-pool creation failed: {exc}", flush=True)
            return None

    def select_graph_bs(self, real_bs: int):
        for graph_bs in self.graph_capture_sizes:
            if real_bs <= graph_bs:
                return graph_bs
        return None

    def get_stats(self) -> dict:
        stats = dict(self.stats)
        stats["bucket_hits"] = dict(stats["bucket_hits"])
        return stats

    def reset_runtime_stats(self):
        """Keep capture results but start a fresh runtime measurement window."""
        self.stats["replay"] = 0
        self.stats["fallback"] = 0
        self.stats["bucket_hits"] = {}

    def _bind(self, entry: DecodeGraphEntry):
        set_context(
            False,
            slot_mapping=entry.slot_mapping,
            context_lens=entry.context_lens,
            block_tables=entry.block_tables,
            max_context_len=self.config.max_model_len,
            use_cuda_graph=True,
            block_size=self.block_size,
        )

    def _allocate_entry(self, graph_bs: int) -> DecodeGraphEntry:
        entry = DecodeGraphEntry(graph_bs=graph_bs)
        entry.input_ids = torch.zeros((graph_bs,), dtype=torch.int64, device=self.device)
        entry.positions = torch.zeros((graph_bs,), dtype=torch.int64, device=self.device)
        dummy_slot = self.dummy_kv_block * self.block_size
        entry.slot_mapping = torch.full((graph_bs,), dummy_slot, dtype=torch.int32, device=self.device)
        entry.context_lens = torch.ones((graph_bs,), dtype=torch.int32, device=self.device)
        entry.block_tables = torch.full(
            (graph_bs, self.max_block_cols),
            self.dummy_kv_block,
            dtype=torch.int32,
            device=self.device,
        )
        # Reuse pinned staging buffers.  The old path rebuilt five tensors and
        # a nested Python block-table list on every decode token.
        # The model runner sets the global default device to NPU while it is
        # being initialized.  Pinned tensors must nevertheless be dense CPU
        # tensors; relying on the default device makes torch-npu try to pin an
        # NPU tensor and raises "Only dense CPU tensors can be pinned".
        entry.host_input_ids = torch.empty(
            (graph_bs,), dtype=torch.int64, device="cpu", pin_memory=True
        )
        entry.host_positions = torch.empty(
            (graph_bs,), dtype=torch.int64, device="cpu", pin_memory=True
        )
        entry.host_slot_mapping = torch.empty(
            (graph_bs,), dtype=torch.int32, device="cpu", pin_memory=True
        )
        entry.host_context_lens = torch.empty(
            (graph_bs,), dtype=torch.int32, device="cpu", pin_memory=True
        )
        entry.host_block_tables = torch.empty(
            (graph_bs, self.max_block_cols),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        # NumPy views share the pinned storage and make row slices cheap without
        # allocating transient CPU tensors.  Torch itself already depends on
        # NumPy in the supported runtime image.
        entry.host_input_ids_np = entry.host_input_ids.numpy()
        entry.host_positions_np = entry.host_positions.numpy()
        entry.host_slot_mapping_np = entry.host_slot_mapping.numpy()
        entry.host_context_lens_np = entry.host_context_lens.numpy()
        entry.host_block_tables_np = entry.host_block_tables.numpy()
        return entry

    def _fill_dummy(self, entry: DecodeGraphEntry):
        dummy_slot = self.dummy_kv_block * self.block_size
        entry.input_ids.zero_()
        entry.positions.zero_()
        entry.context_lens.fill_(1)
        entry.slot_mapping.fill_(dummy_slot)
        entry.block_tables.fill_(self.dummy_kv_block)
        entry.block_tables[:, 0] = self.dummy_kv_block

    def build_metadata(self, entry: DecodeGraphEntry, seqs):
        dummy_slot = self.dummy_kv_block * self.block_size
        input_ids = entry.host_input_ids_np
        positions = entry.host_positions_np
        context_lens = entry.host_context_lens_np
        slot_mapping = entry.host_slot_mapping_np
        block_tables = entry.host_block_tables_np
        input_ids.fill(0)
        positions.fill(0)
        context_lens.fill(1)
        slot_mapping.fill(dummy_slot)
        block_tables.fill(self.dummy_kv_block)
        for row, seq in enumerate(seqs):
            input_ids[row] = seq.last_token
            positions[row] = len(seq) - 1
            context_lens[row] = len(seq)
            slot_mapping[row] = (
                seq.block_table[-1] * self.block_size
                + (seq.last_block_num_tokens - 1)
            )
            table = seq.block_table
            cols = min(len(table), self.max_block_cols)
            block_tables[row, :cols] = table[:cols]

    def _copy_into_static(self, entry: DecodeGraphEntry):
        entry.input_ids.copy_(entry.host_input_ids, non_blocking=True)
        entry.positions.copy_(entry.host_positions, non_blocking=True)
        entry.slot_mapping.copy_(entry.host_slot_mapping, non_blocking=True)
        entry.context_lens.copy_(entry.host_context_lens, non_blocking=True)
        entry.block_tables.copy_(entry.host_block_tables, non_blocking=True)

    def _forward(self, entry: DecodeGraphEntry):
        self._bind(entry)
        hidden_states = self.model(entry.input_ids, entry.positions)
        return self.model.compute_logits(hidden_states)

    def capture_all(self):
        for graph_bs in self.graph_capture_sizes:
            self.capture(graph_bs)

    @torch.inference_mode()
    def capture(self, graph_bs: int):
        entry = DecodeGraphEntry(graph_bs=graph_bs)
        try:
            # Keep staging/static-buffer allocation inside the guarded region.
            # Runtime-specific allocation failures should disable this bucket
            # and fall back to eager decode instead of aborting engine startup.
            entry = self._allocate_entry(graph_bs)
            self._fill_dummy(entry)
            from nanovllm.engine.aclgraph_attn import (
                begin_attn_capture,
                finish_attn_capture,
            )
            for _ in range(self.warmup_iters):
                self._forward(entry)
            torch.npu.synchronize()

            graph = self._npu.NPUGraph()
            graph_kwargs = {}
            if self.graph_pool is not None:
                graph_kwargs["pool"] = self.graph_pool

            if self._pa_update_ok:
                begin_attn_capture()
            try:
                with self._npu.graph(graph, **graph_kwargs):
                    logits = self._forward(entry)
                    # The common 310P path is greedy.  Keep argmax in the graph
                    # so it does not pay the generic sampler's full-vocabulary
                    # argsort, RNG, masks, and sampling-metadata copies.
                    greedy_token_ids = logits.argmax(dim=-1)
                torch.npu.synchronize()
                if self._pa_update_ok:
                    entry.attn_captures = finish_attn_capture()
                else:
                    entry.attn_captures = None
            except Exception:
                if self._pa_update_ok:
                    finish_attn_capture()
                raise

            entry.graph = graph
            entry.logits = logits
            entry.greedy_token_ids = greedy_token_ids
            self.entries[graph_bs] = entry
            self.stats["capture_success"] += 1
            if self.graph_pool is None:
                self.graph_pool = self._try_get_pool(graph)
        except Exception as exc:
            entry.enabled = False
            entry.capture_error = str(exc)
            self.entries[graph_bs] = entry
            self.stats["capture_failure"] += 1
            try:
                torch.npu.synchronize()
                torch.npu.empty_cache()
            except Exception:
                pass
            print(
                f"[nano-vllm] NPUGraph capture failed for batch_size={graph_bs}; "
                "falling back to eager mode: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            reset_context()

    def _try_get_pool(self, graph):
        try:
            pool_fn = getattr(graph, "pool", None)
            if pool_fn is not None:
                return pool_fn()
        except Exception:
            pass
        return None

    @torch.inference_mode()
    def try_run(self, seqs):
        if not seqs:
            return None
        real_bs = len(seqs)
        graph_bs = self.select_graph_bs(real_bs)
        if graph_bs is None:
            self.stats["fallback"] += 1
            return None

        entry = self.entries.get(graph_bs)
        if entry is None:
            self.capture(graph_bs)
            entry = self.entries.get(graph_bs)
        if entry is None or not entry.enabled or entry.graph is None:
            self.stats["fallback"] += 1
            return None

        self.build_metadata(entry, seqs)
        self._copy_into_static(entry)
        self._bind(entry)
        # H2D metadata copies and graph replay are enqueued on the same current
        # stream, so stream ordering is sufficient. A host-side synchronize
        # here serialized every decode token; result.tolist() provides the
        # required completion point after replay.
        if entry.attn_captures:
            from nanovllm.engine.aclgraph_attn import update_paged_attention_params
            ok = update_paged_attention_params(
                entry.attn_captures,
                block_table=entry.block_tables,
                context_lens=entry.context_lens,
                update_stream=self.update_stream,
            )
            if not ok:
                entry.enabled = False
                entry.capture_error = "PA graph-task update unsupported/failed"
                entry.attn_captures = None
                self.stats["fallback"] += 1
                print(
                    f"[nano-vllm] NPUGraph disabled for batch_size={graph_bs} after "
                    "a paged-attention update failure; falling back to eager mode.",
                    flush=True,
                )
                return None
        entry.graph.replay()
        self.stats["replay"] += 1
        self.stats["bucket_hits"][graph_bs] = self.stats["bucket_hits"].get(graph_bs, 0) + 1
        if all(seq.temperature <= 1e-10 for seq in seqs):
            return entry.greedy_token_ids[:real_bs], True
        return entry.logits[:real_bs], False
