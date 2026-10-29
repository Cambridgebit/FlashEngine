
from dataclasses import dataclass
import os
import traceback

import torch

from flashengine.backends.graph import DecodeGraphExecutor, DecodeGraphPlan
from flashengine.utils.context import set_context, reset_context, get_context


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
    # 已用真实输入做过多少次图内外 logits 对比（每桶只查前两次，避免常驻开销）。
    validated_steps: int = 0

class NPUGraphDecodeRunner(DecodeGraphExecutor):

    def __init__(self, plan: DecodeGraphPlan, capture_greedy: bool = True):
        self.model = plan.model
        self.state_runtime = plan.state_runtime
        self.config = plan.config
        self.device = plan.device
        self.block_size = plan.block_size
        # greedy 归约在图外（LITEQ_GREEDY_BACKEND=cpu）时，图内 argmax 无人使用，
        # 改为图内只出 logits，省一次图内归约。
        self.capture_greedy = capture_greedy and not plan.config.enable_constraints
        max_bs = plan.max_batch_size
        sizes = [x for x in ([1, 2, 4, 8] + list(range(16, max_bs + 1, 16))) if x <= max_bs]
        if not sizes or sizes[-1] != max_bs:
            sizes.append(max_bs)
        self.graph_capture_sizes = sizes
        self.max_block_cols = plan.max_block_columns
        self.warmup_iters = plan.warmup_iters
        self.dummy_kv_block = self.config.num_kvcache_blocks - 1
        self.entries: dict[int, DecodeGraphEntry] = {}
        self._npu = npu_graph_api()
        self.graph_pool = self._create_pool_handle()
        self.capture_stream = torch.npu.Stream()
        self.update_stream = torch.npu.Stream()
        for stream in (self.capture_stream, self.update_stream):
            with torch.npu.stream(stream):
                torch.empty(1, device=self.device)
        from flashengine.backends.ascend310p.aclgraph_attn import supports_pa_graph_update
        self._pa_update_ok = supports_pa_graph_update()
        self._compare_logits_fn = None
        self._compare_resolved = False
        # 图内外复算对比要求「多跑一次前向没有副作用」。Qwen3.5/GDN 的解码会把
        # conv/ssm 设备侧状态就地推进，多跑一次前向就多消费一个 token，故这类
        # 模型只保留按桶回退，不做复算比对（行为与主项目一致）。
        # 另外主项目文档记录过 310P 上图内外 ULP 差可致 argmax 翻转：数值门禁
        # 可能把本来可用的桶判失效并静默回退 eager，故提供 `=0` 关闭口。
        self._numeric_guard_enabled = os.environ.get(
            "FLASHENGINE_GRAPH_NUMERIC_GUARD", "1") != "0"
        self._numeric_guard_safe = plan.state_runtime is None and self._numeric_guard_enabled
        self.stats = {
            "capture_success": 0,
            "capture_failure": 0,
            "replay": 0,
            "fallback": 0,
            "bucket_hits": {},
            # 图内外数值对比（捕获自检 + 每桶前两次真实输入）
            "numeric_checks": 0,
            "numeric_failures": 0,
        }

    @property
    def capture_sizes(self) -> tuple[int, ...]:
        return tuple(self.graph_capture_sizes)

    def is_available(self) -> bool:
        return any(
            entry.enabled and entry.graph is not None
            for entry in self.entries.values()
        )

    def _create_pool_handle(self):
        handle_fn = getattr(self._npu, "graph_pool_handle", None)
        if handle_fn is None:
            return None
        try:
            return handle_fn()
        except Exception as exc:
            print(f"[FlashEngine] NPUGraph graph-pool creation failed: {exc}", flush=True)
            return None

    def select_graph_bs(self, real_bs: int):
        for graph_bs in self.graph_capture_sizes:
            if real_bs <= graph_bs:
                return graph_bs
        return None

    def snapshot(self) -> dict:
        stats = dict(self.stats)
        stats["bucket_hits"] = dict(stats["bucket_hits"])
        stats["captured_greedy"] = self.capture_greedy
        # 复算守卫是否对当前模型启用（递归状态模型恒 False，见 _numeric_guard_safe）。
        stats["numeric_guard"] = self._numeric_guard_safe
        # 只有 enabled 且真的捕到图的桶才算「在用图」；失效桶靠 enabled=False
        # + capture_error 表达，不再出现在 active_buckets 里被误报为在用。
        stats["active_buckets"] = [
            size for size, entry in self.entries.items()
            if entry.enabled and entry.graph is not None
        ]
        return stats

    def reset(self):
        self.stats["replay"] = 0
        self.stats["fallback"] = 0
        self.stats["bucket_hits"] = {}
        self.stats["numeric_checks"] = 0
        self.stats["numeric_failures"] = 0

    def _resolve_compare_logits(self):
        """惰性取 compare_logits；模块不可用时返回 None（守卫关闭=主项目原行为）。"""
        if not self._compare_resolved:
            self._compare_resolved = True
            try:
                from flashengine.validation import compare_logits
                self._compare_logits_fn = compare_logits
            except Exception as exc:  # noqa: BLE001 -- 守卫是可选增强，缺模块不影响出图
                print(
                    "[FlashEngine] 310P 图数值守卫不可用，跳过图内外 logits 对比："
                    f"{exc}",
                    flush=True,
                )
        return self._compare_logits_fn

    def _compare_numeric(self, reference, candidate):
        """对比两份 logits：通过返回 None，不通过返回失败描述（不抛）。

        compare_logits 内部统一转 CPU float32 再比（310P 上避免设备侧 fp16
        舍入差），失败会抛 RuntimeError，这里转成描述串交给调用方决定回退。
        """
        compare_logits = self._resolve_compare_logits()
        if compare_logits is None:
            return None
        try:
            compare_logits(reference, candidate)
        except Exception as exc:  # noqa: BLE001 -- 由调用方决定回退策略
            return f"{type(exc).__name__}: {exc}"
        return None

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
        if self.state_runtime:
            self.state_runtime.bind_decode(get_context(), entry.graph_bs)

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
        if self.state_runtime:
            self.state_runtime.prepare_capture()
        for graph_bs in self.graph_capture_sizes:
            self.capture(graph_bs)

    @torch.inference_mode()
    def capture(self, graph_bs: int):
        entry = DecodeGraphEntry(graph_bs=graph_bs)
        try:
            entry = self._allocate_entry(graph_bs)
            self._fill_dummy(entry)
            from flashengine.backends.ascend310p.aclgraph_attn import (
                begin_attn_capture,
                finish_attn_capture,
            )
            reference = None
            for _ in range(max(1, self.warmup_iters)):
                reference = self._forward(entry)
            # 捕获前的 eager 结果留作自检基准（clone 出来，避免与图内输出缓冲同块）。
            if reference is not None:
                reference = reference.detach().clone()
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
                    greedy_token_ids = (
                        logits.argmax(dim=-1) if self.capture_greedy else None)
                torch.npu.synchronize()
                if self._pa_update_ok:
                    entry.attn_captures = finish_attn_capture()
                else:
                    entry.attn_captures = None
            except Exception:
                if self._pa_update_ok:
                    finish_attn_capture()
                raise

            # 捕获后立即回放一次做自检：不再把「捕到了但数值不对」的桶登记成可用。
            # 递归状态模型跳过（回放会推进设备侧 conv/ssm 状态，见 _numeric_guard_safe）。
            if reference is not None and self._numeric_guard_safe:
                graph.replay()
                torch.npu.synchronize()
                failure = self._compare_numeric(reference, logits)
                reference = None
                if failure is None:
                    if self._compare_logits_fn is not None:
                        self.stats["numeric_checks"] += 1
                else:
                    self.stats["numeric_failures"] += 1
                    raise RuntimeError(
                        f"捕获后回放数值自检失败（batch_size={graph_bs}）：{failure}"
                    )

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
                f"[FlashEngine] NPUGraph capture failed for batch_size={graph_bs}; "
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
        if self.state_runtime:
            self.state_runtime.prepare_decode(seqs)
        self._copy_into_static(entry)
        self._bind(entry)
        if entry.attn_captures:
            from flashengine.backends.ascend310p.aclgraph_attn import update_paged_attention_params
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
                    f"[FlashEngine] NPUGraph disabled for batch_size={graph_bs} after "
                    "a paged-attention update failure; falling back to eager mode.",
                    flush=True,
                )
                return None
        entry.graph.replay()
        if self._numeric_guard_safe and entry.validated_steps < 2:
            # 每桶前两次「真实输入」再复核一次：捕获自检只覆盖 dummy 元数据，
            # 真实 block_table/context_lens 走的是 PA 元数据刷新通路，只有真实
            # 输入才能暴露该通路没生效（图内读旧元数据）这类问题。复算只对无副作用
            # 的模型做：Qwen3.5/GDN 多跑一次前向会推进 conv/ssm 状态。
            graph_logits = entry.logits.detach().clone()
            reference = self._forward(entry)
            failure = self._compare_numeric(reference, graph_logits)
            del graph_logits, reference
            if failure is None:
                entry.validated_steps += 1
                if self._compare_logits_fn is not None:
                    self.stats["numeric_checks"] += 1
            else:
                entry.enabled = False
                entry.capture_error = f"真实输入数值检查失败：{failure}"
                self.stats["numeric_failures"] += 1
                self.stats["fallback"] += 1
                print(
                    f"[FlashEngine] NPUGraph disabled for batch_size={graph_bs} "
                    f"after a real-input numeric check failure; falling back to "
                    f"eager mode: {failure}",
                    flush=True,
                )
                return None
        self.stats["replay"] += 1
        self.stats["bucket_hits"][graph_bs] = self.stats["bucket_hits"].get(graph_bs, 0) + 1
        if self.capture_greedy and all(seq.temperature <= 1e-10 for seq in seqs):
            return entry.greedy_token_ids[:real_bs], True
        return entry.logits[:real_bs], False
