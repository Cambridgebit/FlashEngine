import gc
import os
import pickle
import time
import traceback
import tempfile
from pathlib import Path
from collections import OrderedDict
from itertools import accumulate

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from flashengine.backends.graph import DecodeGraphPlan
from flashengine.config import Config
from flashengine.engine.block_manager import BlockManager
from flashengine.engine.sequence import Sequence
from flashengine.engine.memory import MemoryBudgetError, plan_kv_memory
from flashengine.layers.sampler import Sampler
from flashengine.utils.context import set_context, get_context, reset_context


def _bucket(table, n):
    return next((x for x in table if x >= n), None)


def _graph_buckets(max_bs):
    buckets = [x for x in ([1, 2, 4, 8] + list(range(16, max_bs + 1, 16))) if x <= max_bs]
    if not buckets or buckets[-1] != max_bs:
        buckets.append(max_bs)
        print(
            f"[FlashEngine] Added graph bucket {max_bs} because max_num_seqs={max_bs} "
            "is not covered by the standard bucket set.",
            flush=True,
        )
    return buckets


def _refill_ids(t2d, m_idx, rows: int, k: int, ar_f=None, shift_idx=None):
    bonus = t2d.gather(1, m_idx[:, None])
    if shift_idx is None:
        t_ext = torch.cat([t2d[:, 1:], t2d[:, k:k + 1]], dim=1)
    else:
        t_ext = t2d.gather(1, shift_idx.expand(t2d.size(0), rows))
    if ar_f is None:
        ar = torch.arange(rows, device=t2d.device)
        return torch.where(ar[None, :] == m_idx[:, None], bonus, t_ext).reshape(-1)
    m_f = m_idx.to(torch.float32)[:, None]
    return torch.where(ar_f[None, :] == m_f, bonus, t_ext).reshape(-1)


class MetaBlob:

    def __init__(self, i64_segments, f32_segments, device):
        self.device = device
        self._i64_off, self._i64_size = {}, {}
        pos = 0
        for name, size in i64_segments:
            self._i64_off[name], self._i64_size[name] = pos, size
            pos += size
        self.i64 = torch.zeros(pos, dtype=torch.int64, device='cpu', pin_memory=True)
        self._f32_off, self._f32_size = {}, {}
        pos = 0
        for name, size in f32_segments:
            self._f32_off[name], self._f32_size[name] = pos, size
            pos += size
        self.f32 = torch.zeros(pos, dtype=torch.float32, device='cpu', pin_memory=True)
        self._views = {}
        self._dev = {}

    def offset(self, name):
        return self._i64_off.get(name, self._f32_off.get(name))

    def _slice(self, plane, name):
        if plane == 'i64':
            return self._i64_off[name], self._i64_size[name]
        return self._f32_off[name], self._f32_size[name]

    def _view(self, plane, name, shape=None, device=False):
        key = ("d" if device else "") + plane, name, shape
        v = self._views.get(key)
        if v is None:
            off, size = self._slice(plane, name)
            source = self._device(plane) if device else getattr(self, plane)
            v = source[off:off + size]
            if shape is not None:
                v = v.view(*shape)
            self._views[key] = v
        return v

    def view_i64(self, name, shape=None):
        return self._view('i64', name, shape)

    def view_f32(self, name, shape=None):
        return self._view('f32', name, shape)

    def _device(self, plane):
        if self._dev.get(plane) is None:
            source = getattr(self, plane)
            self._dev[plane] = torch.zeros(source.numel(), dtype=source.dtype, device=self.device)
        return self._dev[plane]

    def device_i64(self):
        return self._device('i64')

    def device_f32(self):
        return self._device('f32')

    def device_view_i64(self, name, shape=None):
        return self._view('i64', name, shape, True)

    def device_view_f32(self, name, shape=None):
        return self._view('f32', name, shape, True)

    def h2d(self, non_blocking=True):
        self.device_i64().copy_(self.i64, non_blocking=non_blocking)
        self.device_f32().copy_(self.f32, non_blocking=non_blocking)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        default_dtype = torch.get_default_dtype()
        default_device = torch.get_default_device()
        try:
            self._initialize(config, rank, event)
        except Exception:
            if getattr(self, "_owns_process_group", False):
                self.exit()
            else:
                # 尚未获得进程组时，不触碰其他模型的上下文或后端运行时。
                store_dir = getattr(self, "_store_dir", None)
                self.__dict__.clear()
                self._closed = True
                if store_dir is not None:
                    store_dir.cleanup()
            raise
        finally:
            torch.set_default_device(default_device)
            torch.set_default_dtype(default_dtype)

    def _initialize(self, config: Config, rank: int, event: Event | list[Event]):
        startup_t0 = time.perf_counter()
        self.startup_stats: dict[str, float] = {}
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.backend = config.backend_impl
        self.dflash_paged = config.use_paged_draft
        self.paged_context_attn = self.backend.supports_paged_context_attn
        draft_rows = (
            config.spec_num_tokens
            if config.spec_sample_from_anchor
            else config.spec_num_tokens + 1
        )
        if self.dflash_paged and draft_rows > self.block_size:
            raise ValueError(
                "Ascend 310P paged DFlash requires draft block_size <= "
                f"KV-cache block size ({draft_rows} > {self.block_size})."
            )
        if config.enable_prefix_cache:
            disable_reason = self.backend.prefix_cache_disable_reason()
            if disable_reason is not None:
                print(
                    f"[FlashEngine] Prefix cache disabled by backend: {disable_reason}.",
                    flush=True,
                )
                config.enable_prefix_cache = False
        self.device = f"{self.backend.device_name}:{config.device_ids[rank]}"
        self.dtype = self.backend.default_dtype or config.model_config.dtype
        self.spec_enabled = bool(config.spec_model)
        self.spec_runtime = None
        if self.spec_enabled and not self.backend.supports_spec:
            raise RuntimeError(
                f"Backend {self.backend.name} does not support speculative decoding "
                "(supports_spec=False); remove spec_model."
            )
        self.spec_hidden: dict[int, torch.Tensor] = {}
        self.spec_carry: dict[int, tuple[torch.Tensor, int]] = {}
        self.df_prefix_snapshot_stats = {"saved": 0, "restored": 0, "evicted": 0}
        self._df_prefix_snapshot_bytes = 0
        _env_sample = os.environ.get("FLASHENGINE_GRAPH_SAMPLE")
        self.graph_sample = (
            self.world_size == 1
            and (self.backend.graph_sample_default if _env_sample is None
                 else _env_sample != "0")
        )
        self._graph_bs_active = 0
        self.graphs = None
        self.decode_greedy_graphs = {}
        self._decode_batch_greedy = False
        self.graph_pool = None
        self.graph_runner = None
        self._max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self._graph_max_bs = min(config.max_num_seqs, 512)
        M = self._graph_max_bs
        W = self._max_num_blocks + 1
        self._pin_M, self._pin_W = M, W
        self._dblob = MetaBlob(
            [('ids', M), ('pos', M), ('slots', M), ('ctx', M),
             ('ctx_cu', M + 1), ('bt', M * W), ('topks', M)],
            [('temperatures', M), ('top_ps', M)], device=self.device)

        self.backend.set_device(config.device_ids[rank])
        self._owns_process_group = False
        if dist.is_initialized():
            raise RuntimeError("每个引擎必须在独立进程中拥有自己的分布式进程组")
        if self.world_size == 1:
            self._store_dir = tempfile.TemporaryDirectory(prefix="flashengine-dist-")
            rendezvous = (Path(self._store_dir.name) / "store").as_uri()
        else:
            rendezvous = f"tcp://localhost:{config.hccl_port}"
        dist.init_process_group(self.backend.dist_name, rendezvous,
                                world_size=self.world_size, rank=rank)
        self._owns_process_group = True
        self._memory_baseline = self.backend.memory_stats()["allocated_bytes.all.current"]
        self.memory_stages = {}
        default_dtype = torch.get_default_dtype()
        if self.backend.default_dtype is not None and self.backend.default_dtype != config.model_config.dtype:
            print(
                f"[FlashEngine] Backend {self.backend.name} requires "
                f"dtype={self.backend.default_dtype}; model dtype={config.model_config.dtype}. "
                "Weights will be converted while loading.",
                flush=True,
            )
        torch.set_default_dtype(self.dtype)
        torch.set_default_device(self.device)
        model_load_t0 = time.perf_counter()
        self.target_adapter = config.target_adapter
        self.model = self.target_adapter.build(hf_config)
        self.target_adapter.load_weights(
            self.model, config.model,
            dequant=config.dequant_int8_weights, dtype=self.dtype)
        self.state_runtime = self.target_adapter.create_state_runtime(self.model, config)
        if self.spec_enabled:
            self.model.model.aux_hidden_state_layers = tuple(config.spec_aux_layers)
            self.draft_config = config.draft_config
            if getattr(self.draft_config, "draft_vocab_size", None) is None:
                self.draft_config.draft_vocab_size = config.model_config.vocab_size
            self.draft_model = config.spec_adapter.build_draft(config, self.model)
            self.spec_runtime = config.spec_adapter.build_runtime(self)
            if config.spec_is_block_draft:
                d = self.draft_config
                self.draft_model.use_paged_kv = self.dflash_paged
                self.draft_model._build_fused_kv_buffers()
                hd = getattr(d, "head_dim", d.hidden_size // d.num_attention_heads)
                if not self.dflash_paged:
                    ctx_shape = (config.max_num_seqs + 1, config.max_model_len,
                                 d.num_hidden_layers, d.num_key_value_heads, hd)
                    self.draft_model.ctx_k_cache = torch.zeros(ctx_shape, dtype=self.dtype)
                    self.draft_model.ctx_v_cache = torch.zeros(ctx_shape, dtype=self.dtype)
                    self._df_prefix_snapshots = OrderedDict()
                    self._df_prefix_snapshot_limit = 1 << 30  # 1 GiB host RAM
                self._df_pad_ctx_slot = config.max_num_seqs
                self.draft_ctx_slots: dict[int, int] = {}
                self._df_ctx_free = set(range(config.max_num_seqs))
                self._df_prefix_restored: set[int] = set()
                self.spec_ctx_pending: dict[int, torch.Tensor | None] = {}
        weight_nz_count = self.backend.optimize_model_weights(self.model)
        if self.config.is_moe:
            # MoE 专家权重不走 LinearBase 路径（stacked [E,...] 参数），单独后处理。
            weight_nz_count += self.backend.optimize_moe_weights(self.model)
        if self.spec_enabled:
            weight_nz_count += self.backend.optimize_model_weights(self.draft_model)
            if self.config.spec_is_block_draft:
                weight_nz_count += self.backend.optimize_block_draft_weights(
                    self.draft_model, self.model.lm_head
                )
        self.startup_stats["weight_layout_conversions"] = float(weight_nz_count)
        self.startup_stats["model_load_s"] = time.perf_counter() - model_load_t0
        self.sampler = Sampler(device_rng=self.backend.supports_device_sampling_rng)
        warmup_t0 = time.perf_counter()
        self.warmup_model()
        self.startup_stats["model_warmup_s"] = time.perf_counter() - warmup_t0
        stats = self.backend.memory_stats()
        self._warmup_workspace = max(0, stats["allocated_bytes.all.peak"]
                                     - stats["allocated_bytes.all.current"])
        self._record_memory("weights_states_warmup")
        # 先物化静态缓冲，再按实际常驻量规划 KV；临时 padding 在分配后重置。
        requested_blocks = config.num_kvcache_blocks
        config.num_kvcache_blocks = 2
        self._dblob.device_i64()
        self._dblob.device_f32()
        if self.spec_enabled:
            k = self.config.spec_num_tokens
            rows = k + 1
            M, W = self._pin_M, self._pin_W
            df_segments = ([('df_l1', M), ('df_c', M), ('df_ctxslot', M),
                             ('df_ctx_pos', rows * M), ('df_ctx_phys_slots', rows * M),
                            ('df_total_lens', M)]
                           if self.config.spec_is_block_draft else [])
            self._sblob = MetaBlob(
                [('chain_tokens', M), ('chain_pos', k * M), ('chain_ctxlen', k * M),
                 ('chain_slots', k * M), ('v_pos', rows * M), ('v_slots', rows * M),
                 ('v_cu_k', M + 1), ('ctopks', M), ('v_topks', rows * M), ('btables', M * W),
                 ('v_ctx_lens', rows * M), ('v_btables_exp', rows * M * W)]
                + df_segments + [('chain_cu_k', k * (M + 1))],
                [('ctemps', M), ('ctopps', M), ('v_temps', rows * M), ('v_topps', rows * M)], device=self.device)
            self._sg = self._alloc_spec_graph_vars(self._graph_max_bs, rows, k)
            if self.config.spec_is_block_draft:
                self._sg.update(self._alloc_df_graph_vars(self._graph_max_bs, rows))
            self._sblob.device_i64()
            self._sblob.device_f32()
        config.num_kvcache_blocks = requested_blocks
        self._record_memory("static_buffers")
        kv_t0 = time.perf_counter()
        self.allocate_kv_cache()
        self.startup_stats["kv_cache_allocation_s"] = time.perf_counter() - kv_t0
        if self.spec_enabled:
            for name in ("chain_btables", "v_btables", "v_btables_exp"):
                self._sg[name].fill_(self._graph_pad_block())
            for name in ("chain_slots", "v_slots", "df_ctx_slots"):
                if name in self._sg:
                    self._sg[name].fill_(self._graph_pad_slot())
        self._record_memory("kv_cache")
        self.spec_graphs = None
        self.spec_greedy_graphs = {}
        self._spec_capture_greedy = False
        self._spec_batch_greedy = False
        self.spec_graph_pool = None
        self._spec_graph_active = 0
        self.spec_graph_stats = {
            "capture_success": 0,
            "capture_failure": 0,
            "replay": 0,
            "eager_rounds": 0,
            "sampling_fallbacks": 0,
            "greedy_replays": 0,
            "greedy_capture_success": 0,
            "context_limited_rounds": 0,
            "bucket_hits": {},
        }
        self.spec_accept_stats = {
            "rounds": 0,
            "committed": 0,
            "accepted": 0,
            "positions": [0] * config.spec_num_tokens,
        }
        self.decode_graph_stats = {
            "capture_success": 0,
            "capture_failure": 0,
            "replay": 0,
            "fallback": 0,
            "greedy_replays": 0,
            "bucket_hits": {},
        }
        self.prefill_cache_stats = {
            "batches": 0,
            "cache_hit_batches": 0,
            "query_tokens": 0,
            "reused_prefix_tokens": 0,
            "splitfuse_batches": 0,
            "prefix_recomputed_batches": 0,
            "prefix_recomputed_tokens": 0,
        }
        self.graph_capture_attempted = False
        graph_t0 = time.perf_counter()
        if not self.enforce_eager:
            if not self.spec_enabled:
                graph_plan = DecodeGraphPlan(
                    model=self.model,
                    config=self.config,
                    device=self.device,
                    block_size=self.block_size,
                    max_batch_size=self._graph_max_bs,
                    max_block_columns=self._pin_W,
                    warmup_iters=self.backend.graph_warmup_iters(),
                    state_runtime=self.state_runtime,
                )
                self.graph_runner = self.backend.create_decode_graph_executor(graph_plan)
                if self.graph_runner is not None:
                    self.graph_capture_attempted = True
                    self.graph_runner.capture_all()
                    if not self.graph_runner.is_available():
                        failed_stats = self.graph_runner.snapshot()
                        self.decode_graph_stats.update(failed_stats)
                        print(
                            "[FlashEngine] All backend graph buckets failed to capture; "
                            "decode will use eager mode.",
                            flush=True,
                        )
                        self.graph_runner = None
                else:
                    self.graph_capture_attempted = True
                    self.capture_cudagraph()
                    if self.graphs:
                        self.decode_graph_stats["capture_success"] = len(self.graphs)
                    else:
                        self.decode_graph_stats["capture_failure"] = 1
            else:
                _env_spec = os.environ.get("FLASHENGINE_SPEC_GRAPH")
                if (self.world_size == 1 and self.backend.supports_spec_graph
                        and (_env_spec != "0" if _env_spec is not None
                             else self.backend.spec_graph_try_default)):
                    self.graph_capture_attempted = True
                    self.spec_runtime.capture_graph()
        self.startup_stats["graph_capture_s"] = time.perf_counter() - graph_t0
        self._record_memory("graphs")
        for key in self.prefill_cache_stats:
            self.prefill_cache_stats[key] = 0
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        self.startup_stats["model_runner_total_s"] = time.perf_counter() - startup_t0

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=config.shm_name, create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=config.shm_name)
                self.loop()

    def graph_mode(self) -> str:
        if self.enforce_eager:
            return "eager"
        if self.spec_enabled:
            if self.spec_graphs:
                return "spec_graph"
            return "eager_fallback" if self.graph_capture_attempted else "eager"
        if self.graph_runner is not None or self.graphs is not None:
            return "decode_graph"
        return "eager_fallback" if self.graph_capture_attempted else "eager"

    def reset_measurement_stats(self):
        if self.graph_runner is not None:
            self.graph_runner.reset()
        for stats, keys in (
            (self.decode_graph_stats, ("replay", "fallback", "bucket_hits", "greedy_replays")),
            (self.spec_graph_stats,
             ("replay", "eager_rounds", "sampling_fallbacks", "bucket_hits", "context_limited_rounds", "greedy_replays")),
            (self.prefill_cache_stats, tuple(self.prefill_cache_stats)),
        ):
            for key in keys:
                stats[key] = {} if key == "bucket_hits" else 0
        for key in ("rounds", "committed", "accepted"):
            self.spec_accept_stats[key] = 0
        self.spec_accept_stats["positions"] = [0] * self.config.spec_num_tokens
        for key in self.df_prefix_snapshot_stats:
            self.df_prefix_snapshot_stats[key] = 0
        if self.state_runtime:
            for key in self.state_runtime.prefix_stats:
                self.state_runtime.prefix_stats[key] = 0

    def runtime_snapshot(self) -> dict:
        source = (self.spec_graph_stats if self.spec_enabled else
                  self.graph_runner.snapshot() if self.graph_runner is not None else
                  self.decode_graph_stats)
        graph = dict(source)
        graph["bucket_hits"] = dict(graph["bucket_hits"])
        spec = dict(self.spec_accept_stats)
        spec["positions"] = list(spec["positions"])
        return {
            "recurrent": self.state_runtime.snapshot() if self.state_runtime else None,
            "mode": self.graph_mode(),
            "graph": graph,
            "prefill": dict(self.prefill_cache_stats),
            "spec": {
                "enabled": self.spec_enabled,
                "method": self.config.spec_method,
                "num_tokens": self.config.spec_num_tokens if self.spec_enabled else 0,
                **spec,
            },
            "draft_prefix": {
                **self.df_prefix_snapshot_stats,
                "bytes": self._df_prefix_snapshot_bytes,
            },
        }

    def graph_buckets(self):
        if self.spec_enabled and self.spec_graphs:
            return tuple(self.spec_graph_bs)
        if self.graph_runner is not None:
            return tuple(self.graph_runner.capture_sizes)
        return tuple(self.graph_bs) if self.graphs else ()

    def execution_path(self, temperature: float) -> str:
        if self.enforce_eager:
            return "eager"
        if self.spec_enabled:
            if not self.spec_graphs:
                return "eager_spec"
            if self.graph_sample or temperature <= 1e-10:
                return "spec_graph"
            return "eager_spec(sampling)"
        return "decode_graph" if self.graph_runner is not None or self.graphs else "eager"

    def prefix_cache_validator(self):
        return self.spec_runtime.prefix_cache_validator() if self.spec_runtime else None

    def after_spec_prefill(self, seqs):
        self.spec_runtime.after_prefill(seqs)

    def startup_details(self) -> list[tuple[str, str]]:
        details = self.spec_runtime.startup_details() if self.spec_runtime else []
        plan = self.memory_plan
        details.append(("Device", self.device))
        details.append(("Memory budget", f"engine={plan.engine_budget_bytes / 2**20:.0f} MiB "
                        f"(0=auto), workspace={plan.workspace_bytes / 2**20:.0f} MiB, "
                        f"graph={plan.graph_reserve_bytes / 2**20:.0f} MiB, "
                        f"external reserve={plan.device_reserve_bytes / 2**20:.0f} MiB"))
        if self.state_runtime:
            details.append(("Recurrent state", f"{self.state_runtime.slot_count} slots / {self.state_runtime.bytes / 2**20:.1f} MiB"))
            if self.config.enable_prefix_cache:
                details.append(("Recurrent prefix state", f"bounded CPU snapshots ({self.state_runtime.prefix_limit / 2**20:.0f} MiB)"))
        return details

    def exit(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        backend = self.backend
        owns_group = getattr(self, "_owns_process_group", False)
        store_dir = getattr(self, "_store_dir", None)
        world_size = self.world_size
        rank = self.rank
        if self.world_size > 1 and hasattr(self, "shm"):
            self.shm.close()
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                self.shm.unlink()
        try:
            backend.synchronize()
        except Exception as exc:
            print(f"[FlashEngine] Warning: device synchronization during shutdown failed: {exc}",
                  flush=True)

        reset_context()
        self.__dict__.clear()
        self._closed = True
        gc.collect()
        try:
            backend.empty_cache()
        except Exception as exc:
            print(f"[FlashEngine] Warning: device cache cleanup failed: {exc}", flush=True)
        if owns_group and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as exc:
                print(f"[FlashEngine] Warning: process-group shutdown failed: {exc}",
                      flush=True)
        try:
            backend.release_engine()
        except Exception as exc:
            print(f"[FlashEngine] Warning: device runtime shutdown failed: {exc}", flush=True)
        if store_dir is not None:
            store_dir.cleanup()

    def _record_memory(self, stage):
        stats = self.backend.memory_stats()
        current = max(0, stats["allocated_bytes.all.current"] - self._memory_baseline)
        peak = max(0, stats["allocated_bytes.all.peak"] - self._memory_baseline)
        self.memory_stages[stage] = {"resident_bytes": current, "peak_bytes": peak}
        if self.config.memory_budget_bytes and peak > self.config.memory_budget_bytes:
            raise MemoryBudgetError(
                f"{stage} 显存峰值 {peak} 超过引擎预算 {self.config.memory_budget_bytes}")

    def memory_snapshot(self):
        stats = self.backend.memory_stats()
        return {"plan": self.memory_plan.snapshot(), "stages": dict(self.memory_stages),
                "allocated_bytes": max(0, stats["allocated_bytes.all.current"] - self._memory_baseline),
                "peak_bytes": max(0, stats["allocated_bytes.all.peak"] - self._memory_baseline),
                "device": self.device}

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        self.backend.empty_cache()
        self.backend.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        if self.spec_enabled:
            self.run_prefill_spec(seqs)
            self.spec_hidden.clear()
        else:
            self.run(seqs, True)
        self.backend.empty_cache()
        if self.state_runtime:
            self.state_runtime.clear()
        if self.config.spec_is_block_draft:
            # Reclaim draft context slots held by warmup-only sequences.
            self.spec_ctx_pending.clear()
            self._df_prefix_restored.clear()
            self.draft_ctx_slots.clear()
            self._df_ctx_free = set(range(self.config.max_num_seqs))

    def allocate_kv_cache(self):
        config = self.config
        tc = config.model_config
        num_attn_layers = getattr(self.model.model, "num_attention_layers", tc.num_hidden_layers)
        num_kv_heads = tc.num_key_value_heads // self.world_size
        head_dim = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
        block_bytes = 2 * num_attn_layers * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize
        if self.spec_enabled and (
                not self.config.spec_is_block_draft or self.dflash_paged):
            d = self.draft_config
            draft_kv_heads = d.num_key_value_heads // self.world_size
            draft_head_dim = getattr(d, "head_dim", d.hidden_size // d.num_attention_heads)
            draft_layers = (
                d.num_hidden_layers if self.config.spec_is_block_draft else 1
            )
            block_bytes += (2 * draft_layers * self.block_size * draft_kv_heads
                            * draft_head_dim * self.dtype.itemsize)
        requested = config.num_kvcache_blocks
        for _ in range(6):
            free, total = self.backend.mem_get_info()
            stats = self.backend.memory_stats()
            current = stats["allocated_bytes.all.current"]
            try:
                self.memory_plan = plan_kv_memory(
                    total=total, free=free, resident=max(0, current - self._memory_baseline),
                    workspace=max(config.memory_workspace_bytes, self._warmup_workspace),
                    block_bytes=block_bytes, utilization=config.gpu_memory_utilization,
                    budget_bytes=config.memory_budget_bytes,
                    reserve_bytes=config.device_memory_reserve_bytes,
                    graph_bytes=0 if self.enforce_eager else config.graph_memory_reserve_bytes,
                    requested_blocks=requested,
                    minimum_blocks=2 if self.backend.reserve_dummy_kv_block() else 1)
                config.num_kvcache_blocks = self.memory_plan.num_blocks
                break
            except MemoryBudgetError:
                if requested > 0 or config.memory_budget_bytes or _ == 5:
                    raise
            time.sleep(5)
        layers = self.backend.allocate_kv_layers(
            num_layers=num_attn_layers,
            num_blocks=config.num_kvcache_blocks,
            block_size=self.block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        if layers is not None:
            attn_mods = [m for m in self.model.modules()
                         if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
            assert len(attn_mods) == len(layers) == num_attn_layers, (
                f"attn {len(attn_mods)} vs NZ {len(layers)}/{num_attn_layers}")
            for module, (k_cache, v_cache) in zip(attn_mods, layers):
                module.k_cache = k_cache
                module.v_cache = v_cache
            if self.spec_enabled and (
                    not self.config.spec_is_block_draft or self.dflash_paged):
                d = self.draft_config
                draft_attns = [m for m in self.draft_model.modules()
                                if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
                draft_kv_heads = d.num_key_value_heads // self.world_size
                draft_head_dim = getattr(
                    d, "head_dim", d.hidden_size // d.num_attention_heads)
                draft_layers = self.backend.allocate_kv_layers(
                    num_layers=len(draft_attns),
                    num_blocks=config.num_kvcache_blocks,
                    block_size=self.block_size,
                    num_kv_heads=draft_kv_heads,
                    head_dim=draft_head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                assert draft_layers is not None and len(draft_layers) == len(draft_attns)
                for module, (k_cache, v_cache) in zip(draft_attns, draft_layers):
                    module.k_cache = k_cache
                    module.v_cache = v_cache
            return
        shape = self.backend.get_kv_cache_shape(
            num_layers=num_attn_layers,
            num_blocks=config.num_kvcache_blocks,
            block_size=self.block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.kv_cache = torch.empty(shape)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        if self.spec_enabled and not self.config.spec_is_block_draft:
            d = self.draft_config
            draft_kv_heads = d.num_key_value_heads // self.world_size
            draft_head_dim = getattr(d, "head_dim", d.hidden_size // d.num_attention_heads)
            draft_shape = self.backend.get_kv_cache_shape(
                num_layers=d.num_hidden_layers,
                num_blocks=config.num_kvcache_blocks,
                block_size=self.block_size,
                num_kv_heads=draft_kv_heads,
                head_dim=draft_head_dim,
            )
            self.draft_kv_cache = torch.empty(draft_shape)
            attn = self.draft_model.midlayer.self_attn.attn
            attn.k_cache = self.draft_kv_cache[0, 0]
            attn.v_cache = self.draft_kv_cache[1, 0]

    def drop_spec_state(self, seq_id: int):
        self.drop_seq_state(seq_id)

    def drop_seq_state(self, seq_id: int):
        if self.state_runtime:
            self.state_runtime.drop(seq_id)
        if self.spec_runtime:
            self.spec_runtime.drop_state(seq_id)

    def _graph_pad_block(self) -> int:
        return self.backend.graph_pad_block(self.config.num_kvcache_blocks)

    def _graph_pad_slot(self) -> int:
        b = self._graph_pad_block()
        return -1 if b < 0 else b * self.block_size

    def _graph_pad_ctx(self) -> int:
        return self.backend.graph_pad_context_len()

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        pad = self._graph_pad_block()
        block_tables = [seq.block_table + [pad] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        return block_tables

    def _slot_of(self, seq: Sequence, p: int) -> int:
        return seq.block_table[p // self.block_size] * self.block_size + p % self.block_size

    def _fill_bt_row(self, buf, i: int, table: list[int]):
        pad = self._graph_pad_block()
        buf[i, :len(table)] = torch.tensor(table, dtype=torch.int64)
        buf[i, len(table):] = pad

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        prefix_cache_hit_blocks = sum(
            getattr(seq, "_prefix_cache_hit_pending", 0) for seq in seqs
        )
        prefill_query_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        prefix_reused_tokens = sum(seq.num_cached_tokens for seq in seqs)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache / prior chunk
            block_tables = self.prepare_block_tables(seqs)
        for seq in seqs:
            seq._prefix_cache_hit_pending = 0
        paged_context = bool(
            block_tables is not None
            and self.backend.supports_paged_context_attn
        )
        prefix_recomputed = bool(
            prefix_reused_tokens
            and not (
                paged_context
                or self.backend.supports_prefix_cache_attn
            )
        )
        input_ids = torch.tensor(
            input_ids, dtype=torch.int64, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        positions = torch.tensor(
            positions, dtype=torch.int64, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping,
            dtype=self.backend.graph_slot_mapping_dtype(),
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        host_seq_lens_q = torch.tensor(
            [seq.num_scheduled_tokens for seq in seqs], dtype=torch.int32, device="cpu")
        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=block_tables,
            block_size=self.block_size,
            host_seq_lens_q=host_seq_lens_q,
            prefill_cache_hit=prefix_cache_hit_blocks > 0,
            prefix_cache_hit_blocks=prefix_cache_hit_blocks,
            prefill_query_tokens=prefill_query_tokens,
            prefix_reused_tokens=prefix_reused_tokens,
            prefix_recomputed=prefix_recomputed,
        )
        if self.state_runtime:
            self.state_runtime.bind_prefill(get_context(), seqs)
        return input_ids, positions

    def _record_prefill_cache_state(self, seqs: list[Sequence]):
        context = get_context()
        stats = getattr(self, "prefill_cache_stats", None)
        if stats is None:
            stats = self.prefill_cache_stats = {
                "batches": 0,
                "cache_hit_batches": 0,
                "query_tokens": 0,
                "reused_prefix_tokens": 0,
                "splitfuse_batches": 0,
                "prefix_recomputed_batches": 0,
                "prefix_recomputed_tokens": 0,
            }
        stats["batches"] += 1
        if context.prefill_cache_hit:
            stats["cache_hit_batches"] += 1
        stats["query_tokens"] += context.prefill_query_tokens
        stats["reused_prefix_tokens"] += context.prefix_reused_tokens
        if context.splitfuse_used:
            stats["splitfuse_batches"] += 1
        if context.prefix_recomputed:
            stats["prefix_recomputed_batches"] += 1
            stats["prefix_recomputed_tokens"] += context.prefix_reused_tokens
        for seq in seqs:
            seq.prefill_query_tokens += seq.num_scheduled_tokens
            seq.prefill_splitfuse_used |= context.splitfuse_used
            seq.prefill_prefix_recomputed |= context.prefix_recomputed

    def prepare_decode(self, seqs: list[Sequence]):
        n = len(seqs)
        if self.state_runtime:
            self.state_runtime.prepare_decode(seqs)
        block_size = self.block_size
        blob = self._dblob
        ids_v = blob.view_i64('ids')
        pos_v = blob.view_i64('pos')
        ctx_v = blob.view_i64('ctx')
        ctx_cu_v = blob.view_i64('ctx_cu')
        slots_v = blob.view_i64('slots')
        bt = blob.view_i64('bt', (self._pin_M, self._pin_W))
        for i, seq in enumerate(seqs):
            ids_v[i] = seq.last_token
            pos_v[i] = len(seq) - 1
            ctx_v[i] = len(seq)
            slots_v[i] = seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
            self._fill_bt_row(bt, i, seq.block_table)
        ctx_cu_v[0] = 0
        for i in range(n):
            ctx_cu_v[i + 1] = ctx_cu_v[i] + ctx_v[i]
        if self.graphs is not None:
            gv = self.graph_vars
            bucket = _bucket(self.graph_bs, n)
            if bucket is not None:
                with torch.inference_mode():
                    if n < bucket:
                        pad_slot = self._graph_pad_slot()
                        pad_ctx = self._graph_pad_ctx()
                        pad_block = self._graph_pad_block()
                        slots_v[n:bucket].fill_(pad_slot)
                        ctx_v[n:bucket].fill_(pad_ctx)
                        ids_v[n:bucket].zero_()
                        pos_v[n:bucket].zero_()
                        bt[n:, :].fill_(pad_block)
                        ctx_cu_v[n + 1:bucket + 1].fill_(ctx_cu_v[n])
                    else:
                        ctx_cu_v[bucket + 1:].zero_()
                    self._graph_bs_active = bucket
                    set_context(False,
                                slot_mapping=gv['slot_mapping'][:bucket],
                                context_lens=gv['context_lens'][:bucket],
                                block_tables=gv['block_tables'][:bucket],
                                cu_seqlens_q=gv['cu_seqlens_q'][:bucket + 1],
                                cu_seqlens_k=gv['cu_seqlens_k'][:bucket + 1],
                                max_context_len=self.config.max_model_len,
                                use_cuda_graph=True,
                                block_size=self.block_size)
                    if self.state_runtime:
                        self.state_runtime.bind_decode(get_context(), bucket)
                return gv['input_ids'][:n], gv['positions'][:n]
        with torch.inference_mode():
            slot_dtype = self.backend.graph_slot_mapping_dtype()
            input_ids = ids_v[:n].to(self.device, non_blocking=True)
            positions = pos_v[:n].to(self.device, non_blocking=True)
            slot_mapping = slots_v[:n].to(self.device, non_blocking=True).to(slot_dtype)
            context_lens = ctx_v[:n].to(self.device, non_blocking=True).to(torch.int32)
            block_tables = bt[:n].to(self.device, non_blocking=True).to(torch.int32)
            cu_seqlens_q = torch.arange(n + 1, dtype=torch.int32, device=self.device)
            cu_seqlens_k = ctx_cu_v[:n + 1].to(
                self.device, non_blocking=True
            ).to(torch.int32)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                    block_tables=block_tables,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_context_len=self.config.max_model_len,
                    block_size=self.block_size)
        if self.state_runtime:
            self.state_runtime.bind_decode(get_context(), n)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], is_prefill: bool = False):
        vocab = self.config.model_config.vocab_size
        n = len(seqs)
        blob = self._dblob
        M = self._pin_M
        temps_v = blob.view_f32('temperatures')
        topps_v = blob.view_f32('top_ps')
        topks_v = blob.view_i64('topks')
        for i, seq in enumerate(seqs):
            temps_v[i] = seq.temperature
            topps_v[i] = seq.top_p
            topks_v[i] = seq.top_k if seq.top_k > 0 else vocab
        if not is_prefill and self.graphs is not None:
            gv = self.graph_vars
            return gv['temperatures'][:n], gv['top_ps'][:n], gv['top_ks'][:n]
        with torch.inference_mode():
            off = blob.offset('topks')
            return (blob.f32[:n].to(self.device, non_blocking=True),
                    blob.f32[M:M + n].to(self.device, non_blocking=True),
                    blob.i64[off:off + n].to(self.device, non_blocking=True))

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, bool]:
        if is_prefill or self.enforce_eager or input_ids.size(0) > self._graph_max_bs:
            return self.model.compute_logits(self.model(input_ids, positions)), False
        if self.graphs is None:    # 捕获失败降级：decode 全部走 eager 前向
            if self.graph_capture_attempted:
                self.decode_graph_stats["fallback"] += 1
            return self.model.compute_logits(self.model(input_ids, positions)), False

        bs = input_ids.size(0)
        gv = self.graph_vars

        graph_bs = _bucket(self.graph_bs, bs)
        if graph_bs is None:
            self.decode_graph_stats["fallback"] += 1
            return self.model.compute_logits(self.model(input_ids, positions)), False
        if input_ids.data_ptr() != gv["input_ids"].data_ptr():
            self.decode_graph_stats["fallback"] += 1
            return self.model.compute_logits(self.model(input_ids, positions)), False

        self._dblob.h2d()
        greedy = self._decode_batch_greedy and graph_bs in self.decode_greedy_graphs
        graphs = self.decode_greedy_graphs if greedy else self.graphs
        graphs[graph_bs].replay()
        if greedy:
            self.decode_graph_stats["greedy_replays"] += 1
        self.decode_graph_stats["replay"] += 1
        bucket_hits = self.decode_graph_stats["bucket_hits"]
        bucket_hits[graph_bs] = bucket_hits.get(graph_bs, 0) + 1
        if self.graph_sample:
            return gv["token_ids"][:bs], True
        return self.model.compute_logits(gv["outputs"][:bs]), False

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        self._decode_batch_greedy = all(seq.temperature <= 1e-10 for seq in seqs)
        if not is_prefill and self.graph_runner is not None:
            graph_result = self.graph_runner.try_run(seqs)
            if graph_result is not None:
                result, sampled = graph_result
                if self.rank == 0:
                    if sampled:
                        token_ids = result.tolist()
                    else:
                        sample = self.prepare_sample(seqs, False)
                        token_ids = self._sample(result, seqs, sample).tolist()
                else:
                    token_ids = None
                reset_context()
                return token_ids
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        sample = self.prepare_sample(seqs, is_prefill) if self.rank == 0 else None
        result, sampled = self.run_model(input_ids, positions, is_prefill)
        if is_prefill:
            if self.state_runtime:
                self.state_runtime.finish_prefill()
            self._record_prefill_cache_state(seqs)
        if self.rank == 0:
            token_ids = (result.tolist() if sampled
                         else self._sample(result, seqs, sample).tolist())
        else:
            token_ids = None
        reset_context()
        return token_ids

    def _sample(self, logits, seqs, prepared=None):
        """由 host 请求参数决定纯 greedy；混合/随机请求保留完整采样语义。"""
        if all(seq.temperature <= 1e-10 for seq in seqs):
            return logits.argmax(dim=-1)
        if prepared is None:
            prepared = self.prepare_sample(seqs)
        return self.sampler(logits, *prepared)

    @torch.inference_mode()
    def run_prefill_spec(self, seqs: list[Sequence]) -> list[int]:
        return self.spec_runtime.prefill(seqs)

    @torch.inference_mode()
    def _run_prefill_eagle(self, seqs: list[Sequence]) -> list[int]:
        assert self.rank == 0
        input_ids, positions = self.prepare_prefill(seqs)
        hidden, aux = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden)
        sample = self.prepare_sample(seqs)
        token_ids = self._sample(logits, seqs, sample).tolist()
        offsets = list(accumulate([0] + [seq.num_scheduled_tokens for seq in seqs]))
        seed = self.draft_model.project_aux(aux)
        shifted = []
        for i, seq in enumerate(seqs):
            off0, off1 = offsets[i], offsets[i + 1]
            prompt_end = seq.num_cached_tokens + seq.num_scheduled_tokens
            tail_token = (
                seq.token_ids[prompt_end]
                if prompt_end < seq.num_tokens
                else token_ids[i]
            )
            shifted.append(torch.cat([input_ids[off0 + 1:off1],
                                      input_ids.new_tensor(tail_token).view(1)]))
        self.draft_model.forward_prefill(positions, seed, torch.cat(shifted))
        self._record_prefill_cache_state(seqs)
        reset_context()
        for i, seq in enumerate(seqs):
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                self.spec_hidden[seq.seq_id] = aux[offsets[i + 1] - 1:offsets[i + 1]].clone()
                seq.spec_cold = True
        return token_ids

    @torch.inference_mode()
    def spec_round(self, seqs: list[Sequence]) -> list[list[int]]:
        self._spec_batch_greedy = all(seq.temperature <= 1e-10 for seq in seqs)
        sampling_compatible = (
            self.graph_sample
            or self._spec_batch_greedy
        )
        result = self.spec_runtime.round(seqs, sampling_compatible)
        self._record_spec_acceptance(result)
        return result

    def _record_spec_acceptance(self, result):
        stats = self.spec_accept_stats
        for tokens in result:
            accepted = len(tokens) - 1
            stats["rounds"] += 1
            stats["committed"] += len(tokens)
            stats["accepted"] += accepted
            for position in range(accepted):
                stats["positions"][position] += 1

    def _bucket_for_spec(self, batch_size):
        return _bucket(self.spec_graph_bs, batch_size)

    @torch.inference_mode()
    def context_limited_spec_round(self, seqs):
        # 固定宽度 verify 不应越过窗口；剩余不足一整轮时只消费当前 anchor。
        ids, positions = self.prepare_decode(seqs)
        hidden, _ = self.model(ids, positions)
        logits = self.model.compute_logits(hidden)
        tokens = self._sample(logits, seqs).tolist()
        result = [[token] for token in tokens]
        self._record_spec_acceptance(result)
        self.spec_graph_stats["eager_rounds"] += 1
        self.spec_graph_stats["context_limited_rounds"] += 1
        reset_context()
        return result

    def _fill_spec_meta(self, seqs: list[Sequence]) -> int:
        k = self.config.spec_num_tokens
        rows = k + 1
        n = len(seqs)
        lens = [len(seq) for seq in seqs]
        blob = self._sblob
        M, W = self._pin_M, self._pin_W
        draft_vocab = self.draft_config.draft_vocab_size
        vocab = self.config.model_config.vocab_size
        chain_tokens = blob.view_i64('chain_tokens')
        ctopks = blob.view_i64('ctopks')
        ctemps = blob.view_f32('ctemps')
        ctopps = blob.view_f32('ctopps')
        cp = blob.view_i64('chain_pos', (k, M))
        cc = blob.view_i64('chain_ctxlen', (k, M))
        cs = blob.view_i64('chain_slots', (k, M))
        vp = blob.view_i64('v_pos', (M, rows))
        vs = blob.view_i64('v_slots', (M, rows))
        vt = blob.view_f32('v_temps', (M, rows))
        vpp = blob.view_f32('v_topps', (M, rows))
        cu_k = blob.view_i64('v_cu_k')
        v_topks = blob.view_i64('v_topks')
        bt = blob.view_i64('btables', (M, W))

        for i, seq in enumerate(seqs):
            chain_tokens[i] = seq.last_token
            ctopks[i] = seq.top_k if 0 < seq.top_k < draft_vocab else draft_vocab
            ctemps[i] = seq.temperature
            ctopps[i] = seq.top_p
        for i in range(1, k):
            for j, (seq, l) in enumerate(zip(seqs, lens)):
                p = l + i - 2
                cp[i, j] = p
                cc[i, j] = l + i - 1
                cs[i, j] = self._slot_of(seq, p)
        cu_k[0] = 0
        acc = 0
        for i, (seq, l) in enumerate(zip(seqs, lens)):
            for j in range(rows):
                p = l - 1 + j
                vp[i, j] = p
                vs[i, j] = self._slot_of(seq, p)
                vt[i, j] = seq.temperature
                vpp[i, j] = seq.top_p
                v_topks[i * rows + j] = seq.top_k if seq.top_k > 0 else vocab
            acc += l + k
            cu_k[i + 1] = acc
        for i, seq in enumerate(seqs):
            self._fill_bt_row(bt, i, seq.block_table)
        v_ctx = blob.view_i64('v_ctx_lens', (M, rows))
        v_bte = blob.view_i64('v_btables_exp', (M, rows, W))
        for i, l in enumerate(lens):
            for j in range(rows):
                v_ctx[i, j] = max(l + j, 1)
                v_bte[i, j, :] = bt[i]
        return n

    def _load_spec_carry(self, seqs: list[Sequence], n: int):
        g = self._sg
        g['carry_seed'][:n] = torch.cat([self.spec_carry[seq.seq_id][0] for seq in seqs], dim=0)
        g['drafted_ids'][0][:n] = torch.tensor(
            [self.spec_carry[seq.seq_id][1] for seq in seqs], dtype=torch.int64)

    def _push_spec_meta(self):
        # 链式 decode 的累计长度在 host 展开，图内仅复制固定形状元数据。
        blob = self._sblob
        k, M = self.config.spec_num_tokens, self._pin_M
        lengths = blob.view_i64('chain_ctxlen', (k, M))
        cumulative = blob.view_i64('chain_cu_k', (k, M + 1))
        cumulative[:, 0] = 0
        torch.cumsum(lengths, dim=1, out=cumulative[:, 1:])
        self._sblob.h2d()

    def _readback_spec_commit(self, seqs: list[Sequence]) -> list[list[int]]:
        g = self._sg
        rows = self.config.spec_num_tokens + 1
        n = len(seqs)
        rows_out = g['commit_buf'][:n].tolist()
        results = []
        for i, seq in enumerate(seqs):
            row = rows_out[i]
            results.append(row[:row[rows]])
            self.spec_carry[seq.seq_id] = (g['carry_seed'][i:i + 1].clone(),
                                           row[rows + 1])
            base = i * rows + row[rows] - 1
            self.spec_hidden[seq.seq_id] = g['v_aux'][base:base + 1].clone()
            seq.num_scheduled_tokens = 0
        return results

    @torch.inference_mode()
    def _spec_round_eager(self, seqs: list[Sequence]) -> list[list[int]]:
        assert self.rank == 0
        n = len(seqs)
        if n == 0:
            return []
        self.spec_graph_stats["eager_rounds"] += 1
        lens = [len(seq) for seq in seqs]

        draft_vocab = self.draft_config.draft_vocab_size
        temperatures = torch.tensor(
            [seq.temperature for seq in seqs],
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        top_ps = torch.tensor(
            [seq.top_p for seq in seqs],
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        top_ks_draft = torch.tensor(
            [seq.top_k if 0 < seq.top_k < draft_vocab else draft_vocab for seq in seqs],
            dtype=torch.int64, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        h_seed = torch.cat([self.spec_hidden[seq.seq_id] for seq in seqs], dim=0)
        seed = self.draft_model.project_aux(h_seed)
        slot_dtype = self.backend.graph_slot_mapping_dtype()
        p = torch.tensor(
            [l - 2 for l in lens], dtype=torch.int64, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        ctx = torch.tensor(
            [l - 1 for l in lens], dtype=torch.int32, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        # 1D slot_indices（Ascend ATB 硬性要求；[n,1] 会 ReshapeAndCache setup failed）
        slots = torch.tensor(
            [self._slot_of(seq, l - 2) for seq, l in zip(seqs, lens)],
            dtype=slot_dtype,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        tokens = torch.tensor(
            [seq.last_token for seq in seqs],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        set_context(False, slot_mapping=slots, context_lens=ctx,
                    block_tables=self.prepare_block_tables(seqs),
                    max_context_len=self.config.max_model_len,
                    block_size=self.block_size)
        row_m_hidden, draft_logits = self.draft_model.forward_decode(p, seed, tokens)
        row_m_hidden = row_m_hidden.clone()
        draft_ids = self.sampler.forward_native(draft_logits, temperatures, top_ps, top_ks_draft)
        draft1 = self.draft_model.map_to_target_ids(draft_ids)

        g = self._sg
        g['carry_seed'][:n] = seed
        g['drafted_ids'][0][:n] = draft1

        self._fill_spec_meta(seqs)
        self._push_spec_meta()
        self._sg_round(n, use_cuda_graph=False)
        results = self._readback_spec_commit(seqs)
        for i, seq in enumerate(seqs):
            self.spec_carry[seq.seq_id] = (row_m_hidden[i:i + 1], draft1[i].item())
        reset_context()
        for seq in seqs:
            seq.spec_cold = False
        return results


    def _df_ctx_get_or_alloc(self, seq: Sequence) -> int:
        slot = self.draft_ctx_slots.get(seq.seq_id)
        if slot is None:
            try:
                slot = self._df_ctx_free.pop()
            except KeyError:
                raise RuntimeError(
                    "Draft-context slots are exhausted; live sequences exceeded "
                    f"max_num_seqs={self.config.max_num_seqs}."
                ) from None
            self.draft_ctx_slots[seq.seq_id] = slot
        return slot

    def has_dflash_prefix_snapshot(self, block_id: int, block_hash: int) -> bool:
        snapshot = self._df_prefix_snapshots.get(block_id)
        return snapshot is not None and snapshot[0] == block_hash

    def save_dflash_prefix_snapshots(self, seqs: list[Sequence]) -> None:
        if self.dflash_paged or not self.config.enable_prefix_cache:
            return
        for seq in seqs:
            if not seq.block_table or seq.seq_id not in self.draft_ctx_slots:
                continue
            slot = self.draft_ctx_slots[seq.seq_id]
            block_hash = -1
            # 分块 prefill 只保存已经投影的完整块，不能缓存尚未写入的草稿 KV。
            computed = seq.num_cached_tokens + seq.num_scheduled_tokens
            for i in range(computed // self.block_size):
                block_hash = BlockManager.compute_hash(seq.block(i), block_hash)
                block_id = seq.block_table[i]
                if self.has_dflash_prefix_snapshot(block_id, block_hash):
                    continue
                begin = i * self.block_size
                end = begin + self.block_size
                k = self.draft_model.ctx_k_cache[slot, begin:end].to("cpu").contiguous()
                v = self.draft_model.ctx_v_cache[slot, begin:end].to("cpu").contiguous()
                size = (k.numel() * k.element_size() + v.numel() * v.element_size())
                old = self._df_prefix_snapshots.pop(block_id, None)
                if old is not None:
                    self._df_prefix_snapshot_bytes -= old[3]
                if size > self._df_prefix_snapshot_limit:
                    continue
                while self._df_prefix_snapshot_bytes + size > self._df_prefix_snapshot_limit:
                    _, evicted = self._df_prefix_snapshots.popitem(last=False)
                    self._df_prefix_snapshot_bytes -= evicted[3]
                    self.df_prefix_snapshot_stats["evicted"] += 1
                self._df_prefix_snapshots[block_id] = (block_hash, k, v, size)
                self._df_prefix_snapshot_bytes += size
                self.df_prefix_snapshot_stats["saved"] += 1

    def _restore_dflash_prefix(self, seq: Sequence, slot: int) -> None:
        if not seq.num_cached_tokens or self.dflash_paged:
            return
        block_hash = -1
        for i in range(seq.num_cached_tokens // self.block_size):
            block_hash = BlockManager.compute_hash(seq.block(i), block_hash)
            block_id = seq.block_table[i]
            if not self.has_dflash_prefix_snapshot(block_id, block_hash):
                raise RuntimeError("DFlash prefix snapshot missing after cache admission")
            _, k, v, _ = self._df_prefix_snapshots[block_id]
            self._df_prefix_snapshots.move_to_end(block_id)
            begin = i * self.block_size
            end = begin + self.block_size
            self.draft_model.ctx_k_cache[slot, begin:end].copy_(k)
            self.draft_model.ctx_v_cache[slot, begin:end].copy_(v)
            self.df_prefix_snapshot_stats["restored"] += 1

    @torch.inference_mode()
    def _run_prefill_spec_block_draft(self, seqs: list[Sequence]) -> list[int]:
        assert self.rank == 0
        input_ids, positions = self.prepare_prefill(seqs)
        hidden, aux = self.model(input_ids, positions)
        if self.state_runtime:
            self.state_runtime.finish_prefill()
        logits = self.model.compute_logits(hidden)
        token_ids = self._sample(logits, seqs).tolist()
        self._record_prefill_cache_state(seqs)
        reset_context()
        offsets = list(accumulate([0] + [seq.num_scheduled_tokens for seq in seqs]))
        for i, seq in enumerate(seqs):
            if (self.dflash_paged
                    and (not seq.block_table
                         or not self.draft_model.layers[0].self_attn.k_cache.numel())):
                self.spec_ctx_pending[seq.seq_id] = None
                continue
            slot = self._df_ctx_get_or_alloc(seq)
            if seq.seq_id not in self._df_prefix_restored:
                self._restore_dflash_prefix(seq, slot)
                self._df_prefix_restored.add(seq.seq_id)
            p0, p1 = offsets[i], offsets[i + 1]
            if self.dflash_paged:
                start = seq.num_cached_tokens
                context_slots = torch.tensor(
                    [self._slot_of(seq, p) for p in range(start, start + (p1 - p0))],
                    dtype=self.backend.graph_slot_mapping_dtype(),
                    device="cpu",
                    pin_memory=True,
                ).to(self.device, non_blocking=True)
            else:
                context_slots = torch.full(
                    (p1 - p0,), slot, dtype=torch.int64, device=self.device
                )
            self.draft_model.project_context_kv(
                aux[p0:p1], positions[p0:p1],
                context_slots)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                self.spec_ctx_pending[seq.seq_id] = None
        return token_ids

    def _df_sample_from_anchor(self) -> bool:
        return self.draft_model.sample_from_anchor

    def _df_query_rows(self) -> int:
        return self.draft_model.query_rows(self.config.spec_num_tokens)

    def _df_readback_commit(self, seqs: list[Sequence]) -> list[list[int]]:
        rows = self.config.spec_num_tokens + 1
        rows_out = self._sg['commit_buf'][:len(seqs)].tolist()
        results = []
        for i, (seq, row) in enumerate(zip(seqs, rows_out)):
            count = row[rows]
            results.append(row[:count])
            self.spec_ctx_pending[seq.seq_id] = self._sg['v_aux'][
                i * rows:i * rows + count].clone()
            seq.num_scheduled_tokens = 0
        if self.state_runtime and not self.state_runtime.device_commit:
            self.state_runtime.commit_verify(seqs, [len(tokens) for tokens in results])
        return results

    @torch.inference_mode()
    def _df_round_eager(self, seqs: list[Sequence]) -> list[list[int]]:
        if not seqs:
            return []
        self.spec_graph_stats['eager_rounds'] += 1
        size = self._fill_spec_meta(seqs)
        self._fill_df_meta(seqs)
        self._df_load_ctx_aux(seqs, size)
        self._push_spec_meta()
        self._df_round_body(size, use_cuda_graph=False)
        return self._df_readback_commit(seqs)
    def _verify_block_tables(self, g, bs: int, rows: int):
        if self.paged_context_attn:
            return g['v_btables_exp'][:bs * rows]
        return g['v_btables'][:bs]

    def _alloc_df_graph_vars(self, max_bs: int, rows: int) -> dict:
        aux_width = self.config.model_config.hidden_size * len(self.config.spec_aux_layers)
        slot_dtype = self.backend.graph_slot_mapping_dtype()
        return dict(
            df_ctx_aux=torch.zeros(max_bs * rows, aux_width, dtype=self.dtype),
            df_ctx_positions=torch.zeros(max_bs * rows, dtype=torch.int64),
            df_ctx_slots=torch.full(
                (max_bs * rows,), self._graph_pad_slot(), dtype=slot_dtype),
            df_total_lens=torch.full(
                (max_bs,), self._graph_pad_ctx(), dtype=torch.int32),
        )

    def _fill_df_meta(self, seqs: list[Sequence]):
        if self.state_runtime:
            self.state_runtime.prepare_verify(seqs)
        blob = self._sblob
        l1 = blob.view_i64('df_l1')
        c = blob.view_i64('df_c')
        cs = blob.view_i64('df_ctxslot')
        rows = self.config.spec_num_tokens + 1
        query_rows = self._df_query_rows()
        M = self._pin_M
        ctx_pos = blob.view_i64('df_ctx_pos', (M, rows))
        ctx_phys = blob.view_i64('df_ctx_phys_slots', (M, rows))
        total_lens = blob.view_i64('df_total_lens')
        for i, seq in enumerate(seqs):
            l1_value = len(seq) - 1
            l1[i] = l1_value
            p = self.spec_ctx_pending.get(seq.seq_id)
            pending_count = 0 if p is None else p.size(0)
            c[i] = pending_count
            cs[i] = self.draft_ctx_slots[seq.seq_id]
            start = l1_value - pending_count
            total_lens[i] = l1_value + query_rows
            for j in range(rows):
                pos = start + j
                ctx_pos[i, j] = pos
                ctx_phys[i, j] = (
                    self._slot_of(seq, pos) if self.dflash_paged else int(cs[i])
                )

    def _df_load_ctx_aux(self, seqs: list[Sequence], bs: int):
        g = self._sg
        rows = self.config.spec_num_tokens + 1
        buf = g['df_ctx_aux']
        buf[:bs * rows].zero_()
        for i, seq in enumerate(seqs):
            p = self.spec_ctx_pending.pop(seq.seq_id, None)
            if p is not None:
                buf[i * rows:i * rows + p.size(0)].copy_(p)

    def _df_scatter(self, bs: int):
        g = self._sg
        blob = self._sblob
        rows = self.config.spec_num_tokens + 1
        M = self._pin_M
        g['df_ctx_positions'][:bs * rows].copy_(
            blob.device_view_i64('df_ctx_pos', (M, rows))[:bs].reshape(-1))
        g['df_ctx_slots'][:bs * rows].copy_(
            blob.device_view_i64('df_ctx_phys_slots', (M, rows))[:bs].reshape(-1))
        g['df_total_lens'][:bs].copy_(
            blob.device_view_i64('df_total_lens')[:bs])

    def _df_append_context_graph(self, bs: int):
        g = self._sg
        rows = self.config.spec_num_tokens + 1
        aux = g['df_ctx_aux'][:bs * rows]
        pos = g['df_ctx_positions'][:bs * rows]
        slots = g['df_ctx_slots'][:bs * rows]
        self.draft_model.project_context_kv(aux, pos, slots)

    def _df_draft_block_graph(self, bs: int, use_cuda_graph: bool):
        g = self._sg
        blob = self._sblob
        k = self.config.spec_num_tokens
        rows = k + 1
        query_rows = self._df_query_rows()
        l1 = blob.device_view_i64('df_l1')[:bs]
        block_pos = g['v_positions'][:bs * rows].view(bs, rows)[:, :query_rows]
        query_slots = None
        if self.dflash_paged:
            query_slots = (g['v_slots'][:bs * rows].view(bs, rows)[:, :query_rows]
                           .contiguous().reshape(-1))
        logits = self.draft_model.forward_block(
            g['chain_tokens'][:bs], block_pos,
            blob.device_view_i64('df_ctxslot')[:bs], l1,
            self.config.max_model_len, query_rows,
            query_slots=query_slots,
            block_tables=(g['v_btables'][:bs] if self.dflash_paged else None),
            total_lens=(g['df_total_lens'][:bs]
                        if self.dflash_paged else None),
            sample_from_anchor=self._df_sample_from_anchor())
        ids = self.draft_model.sample_proposals(
            logits,
            g['chain_tokens'][:bs],
            k,
            g['chain_temps'][:bs],
            g['chain_topps'][:bs],
            g['chain_topks'][:bs],
            lambda values, temps, topps, topks: self._sg_sample(
                values, temps, topps, topks, use_cuda_graph
            ),
        )
        if self.draft_model.reduced_vocab:
            ids = self.draft_model.map_to_target_ids(ids)
        g['drafted_ids'][:k, :bs] = ids.t()

    def _df_round_body(self, bs: int, use_cuda_graph: bool = True):
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        self._sg_scatter(bs)
        self._df_scatter(bs)
        self._df_append_context_graph(bs)
        self._df_draft_block_graph(bs, use_cuda_graph)
        v = g['v_input_ids'][:bs * rows].view(bs, rows)
        v[:, 0] = g['chain_tokens'][:bs]
        v[:, 1:].copy_(g['drafted_ids'][:k, :bs].t())
        self._sg_verify(bs, use_cuda_graph)
        self._sg_acceptance(bs)
        if self.state_runtime:
            self.state_runtime.commit_verify_device(g['commit_buf'], bs)

    def _capture_spec_graphs(self, prepare, body, label):
        self.spec_graph_bs = _graph_buckets(self._graph_max_bs)
        self.spec_graphs = {}
        self.spec_greedy_graphs = {}
        try:
            prepare()
            self._push_spec_meta()
            for bs in reversed(self.spec_graph_bs):
                self._sg['v_cu_k'][:bs + 1] = self._sg['v_cu_q'][:bs + 1]
                for _ in range(self.backend.graph_warmup_iters()):
                    body(bs)
                    reset_context()
                    self.backend.synchronize()
                graph = self.backend.new_graph()
                with self.backend.graph_ctx(graph, pool=self.spec_graph_pool):
                    body(bs)
                if self.spec_graph_pool is None:
                    self.spec_graph_pool = self.backend.graph_pool(graph)
                self.spec_graphs[bs] = graph
                reset_context()
                self._push_spec_meta()
                graph.replay()
                self.backend.synchronize()
                self.spec_graph_stats["capture_success"] += 1
            # 保留原随机采样图；另捕获纯 argmax 图，不在 replay 中读取设备标量。
            if (self.graph_sample and self.backend.supports_greedy_graph_variants
                    and os.getenv("FLASHENGINE_SPEC_GREEDY_GRAPH", "1") != "0"):
                for bs in reversed(self.spec_graph_bs):
                    try:
                        self._spec_capture_greedy = True
                        for _ in range(self.backend.graph_warmup_iters()):
                            body(bs)
                            reset_context()
                            self.backend.synchronize()
                        graph = self.backend.new_graph()
                        with self.backend.graph_ctx(graph, pool=self.spec_graph_pool):
                            body(bs)
                        reset_context()
                        graph.replay()
                        self.backend.synchronize()
                        self.spec_greedy_graphs[bs] = graph
                        self.spec_graph_stats["greedy_capture_success"] += 1
                    except Exception as exc:
                        print(f"[FlashEngine] Greedy spec graph bucket {bs} unavailable; "
                              f"using sampling graph: {type(exc).__name__}: {exc}", flush=True)
                        reset_context()
                    finally:
                        self._spec_capture_greedy = False
        except Exception as e:
            print(f"[FlashEngine] {label} graph capture failed; falling back to "
                  f"eager speculative rounds: {type(e).__name__}: {e}", flush=True)
            self.spec_graph_stats["capture_failure"] += 1
            traceback.print_exc()
            try:
                reset_context()
                self.backend.synchronize()
            except Exception:
                pass
            self.spec_graphs = None
            self.spec_greedy_graphs.clear()
            self.spec_graph_pool = None
            try:
                self.backend.empty_cache()
            except Exception:
                pass

    @torch.inference_mode()
    def capture_df_spec_cudagraph(self):
        def prepare():
            M, W = self._pin_M, self._pin_W
            rows = self.config.spec_num_tokens + 1
            blob = self._sblob
            pad_block = self.backend.graph_capture_block(self.config.num_kvcache_blocks)
            pad_base = pad_block * self.block_size
            pad_slots = torch.arange(
                pad_base, pad_base + rows, dtype=torch.int64, device="cpu")
            pad_positions = torch.arange(rows, dtype=torch.int64, device="cpu")
            pad_context_lens = torch.arange(
                1, rows + 1, dtype=torch.int64, device="cpu")
            blob.view_i64('btables', (M, W)).fill_(pad_block)
            blob.view_i64('v_slots', (M, rows))[:] = pad_slots
            blob.view_i64('v_pos', (M, rows))[:] = pad_positions
            blob.view_i64('v_ctx_lens', (M, rows))[:] = pad_context_lens
            blob.view_i64('v_btables_exp', (M, rows, W)).fill_(pad_block)
            blob.view_i64('df_ctx_pos', (M, rows))[:] = pad_positions
            if self.dflash_paged:
                blob.view_i64('df_ctx_phys_slots', (M, rows))[:] = pad_slots
            else:
                blob.view_i64('df_ctx_phys_slots', (M, rows)).fill_(self._df_pad_ctx_slot)
            blob.view_i64('df_total_lens').fill_(self._df_query_rows())
        self._capture_spec_graphs(prepare, self._df_round_body,
                                  "Block-draft speculative")

    @torch.inference_mode()
    def _reset_spec_padding(self, n: int, bucket: int, block_draft: bool):
        if n >= bucket:
            return
        rows = self.config.spec_num_tokens + 1
        M, W = self._pin_M, self._pin_W
        blob = self._sblob
        pad_block = self._graph_pad_block()
        pad_slot = self._graph_pad_slot()
        slots = blob.view_i64('v_slots', (M, rows))
        contexts = blob.view_i64('v_ctx_lens', (M, rows))
        if block_draft:
            pad_slots = (torch.arange(pad_block * self.block_size,
                                      pad_block * self.block_size + rows,
                                      dtype=torch.int64, device="cpu")
                         if pad_block >= 0 else torch.full((rows,), -1, dtype=torch.int64, device="cpu"))
            positions = torch.arange(rows, dtype=torch.int64, device="cpu")
            slots[n:] = pad_slots
            contexts[n:] = positions + 1
            blob.view_i64('v_pos', (M, rows))[n:] = positions
            blob.view_i64('df_ctx_pos', (M, rows))[n:] = positions
            if self.dflash_paged:
                blob.view_i64('df_ctx_phys_slots', (M, rows))[n:] = pad_slots
            else:
                blob.view_i64('df_ctx_phys_slots', (M, rows))[n:].fill_(self._df_pad_ctx_slot)
            blob.view_i64('df_l1')[n:].zero_()
            blob.view_i64('df_c')[n:].zero_()
            blob.view_i64('df_ctxslot')[n:].fill_(self._df_pad_ctx_slot)
            blob.view_i64('df_total_lens')[n:].fill_(self._df_query_rows())
        else:
            slots[n:].fill_(pad_slot)
            contexts[n:].fill_(self._graph_pad_ctx())
        cu_k = blob.view_i64('v_cu_k')
        cu_k[n + 1:] = cu_k[n]
        blob.view_i64('btables', (M, W))[n:].fill_(pad_block)
        blob.view_i64('v_btables_exp', (M, rows, W))[n:].fill_(pad_block)

    def _replay_spec_graph(self, bucket: int):
        self._spec_graph_active = bucket
        self._push_spec_meta()
        greedy = self._spec_batch_greedy and bucket in self.spec_greedy_graphs
        graphs = self.spec_greedy_graphs if greedy else self.spec_graphs
        graphs[bucket].replay()
        if greedy:
            self.spec_graph_stats["greedy_replays"] += 1
        self.spec_graph_stats["replay"] += 1
        hits = self.spec_graph_stats["bucket_hits"]
        hits[bucket] = hits.get(bucket, 0) + 1

    @torch.inference_mode()
    def _df_round_graph(self, seqs: list[Sequence], bucket: int) -> list[list[int]]:
        n = self._fill_spec_meta(seqs)
        self._fill_df_meta(seqs)
        self._reset_spec_padding(n, bucket, True)
        self._df_load_ctx_aux(seqs, bucket)
        self._replay_spec_graph(bucket)
        return self._df_readback_commit(seqs)


    def _alloc_spec_graph_vars(self, max_bs: int, rows: int, k: int) -> dict:
        tc = self.config.model_config
        vocab = tc.vocab_size
        draft_vocab = self.draft_config.draft_vocab_size
        max_num_blocks = self._pin_W
        slot_dtype = self.backend.graph_slot_mapping_dtype()
        pad_block = self._graph_pad_block()
        pad_slot = self._graph_pad_slot()
        return dict(
            carry_seed=torch.zeros(max_bs, tc.hidden_size, dtype=self.dtype),
            chain_tokens=torch.zeros(max_bs, dtype=torch.int64),
            drafted_ids=torch.zeros(k, max_bs, dtype=torch.int64),
            chain_pos=torch.zeros(k, max_bs, dtype=torch.int64),
            chain_slots=torch.full((k, max_bs), pad_slot, dtype=slot_dtype),
            chain_ctxlen=torch.full((k, max_bs), self._graph_pad_ctx(), dtype=torch.int32),
            chain_cu_q=torch.arange(max_bs + 1, dtype=torch.int32),
            chain_cu_k=torch.zeros(k, max_bs + 1, dtype=torch.int32),
            chain_btables=torch.full((max_bs, max_num_blocks), pad_block, dtype=torch.int32),
            chain_temps=torch.ones(max_bs, dtype=torch.float32),
            chain_topps=torch.ones(max_bs, dtype=torch.float32),
            chain_topks=torch.full((max_bs,), draft_vocab, dtype=torch.int64),
            v_input_ids=torch.zeros(max_bs * rows, dtype=torch.int64),
            v_positions=torch.zeros(max_bs * rows, dtype=torch.int64),
            v_cu_q=(torch.arange(max_bs + 1, dtype=torch.int32) * rows),
            v_cu_k=torch.zeros(max_bs + 1, dtype=torch.int32),
            v_slots=torch.full((max_bs * rows,), pad_slot, dtype=slot_dtype),
            v_btables=torch.full((max_bs, max_num_blocks), pad_block, dtype=torch.int32),
            v_ctx_lens=torch.full((max_bs * rows,), self._graph_pad_ctx(), dtype=torch.int32),
            v_btables_exp=torch.full((max_bs * rows, max_num_blocks), pad_block, dtype=torch.int32),
            v_temps=torch.ones(max_bs * rows, dtype=torch.float32),
            v_topps=torch.ones(max_bs * rows, dtype=torch.float32),
            v_topks=torch.full((max_bs * rows,), vocab, dtype=torch.int64),
            v_aux=torch.zeros(max_bs * rows, tc.hidden_size * len(self.config.spec_aux_layers), dtype=self.dtype),
            v_token_ids=torch.zeros(max_bs * rows, dtype=torch.int64),
            commit_buf=torch.full((max_bs, rows + 2), -1, dtype=torch.int64),
            v_ar_f=torch.arange(rows, dtype=torch.float32),
            v_ar_clamp=torch.arange(rows, dtype=torch.int64).clamp(max=k),
            d_pad=torch.zeros(max_bs, rows, dtype=torch.int64),
            neg1_tok=torch.full((max_bs, rows), -1, dtype=torch.int64),
            accept_m=torch.zeros(max_bs, dtype=torch.int64),
            v_shift_idx=torch.cat([
                torch.arange(1, k + 1, dtype=torch.int64),
                torch.tensor([k], dtype=torch.int64)]),
        )

    def _sg_sample(self, logits, temps, topps, topks, use_cuda_graph: bool):
        if ((use_cuda_graph and (self._spec_capture_greedy or not self.graph_sample))
                or (not use_cuda_graph and self._spec_batch_greedy)):
            return logits.argmax(dim=-1)
        return self.sampler.forward_native(logits, temps, topps, topks)

    def _sg_scatter(self, bs: int):
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        blob = self._sblob
        M, W = self._pin_M, self._pin_W
        g['chain_tokens'][:bs].copy_(blob.device_view_i64('chain_tokens')[:bs])
        cp = blob.device_view_i64('chain_pos', (k, M))
        cc = blob.device_view_i64('chain_ctxlen', (k, M))
        cs = blob.device_view_i64('chain_slots', (k, M))
        for i in range(k):
            g['chain_pos'][i][:bs].copy_(cp[i, :bs])
            g['chain_ctxlen'][i][:bs].copy_(cc[i, :bs])
            g['chain_slots'][i][:bs].copy_(cs[i, :bs])
            g['chain_cu_k'][i, :bs + 1].copy_(
                blob.device_view_i64('chain_cu_k', (k, M + 1))[i, :bs + 1])
        g['v_positions'][:bs * rows].copy_(blob.device_view_i64('v_pos')[:bs * rows])
        g['v_slots'][:bs * rows].copy_(blob.device_view_i64('v_slots')[:bs * rows])
        g['v_cu_k'][:bs + 1].copy_(blob.device_view_i64('v_cu_k')[:bs + 1])
        g['chain_topks'][:bs].copy_(blob.device_view_i64('ctopks')[:bs])
        g['v_topks'][:bs * rows].copy_(blob.device_view_i64('v_topks')[:bs * rows])
        bt = blob.device_view_i64('btables', (M, W))
        g['chain_btables'][:bs].copy_(bt[:bs])
        g['v_btables'][:bs].copy_(bt[:bs])
        g['v_ctx_lens'][:bs * rows].copy_(
            blob.device_view_i64('v_ctx_lens')[:bs * rows])
        g['v_btables_exp'][:bs * rows].copy_(
            blob.device_view_i64('v_btables_exp', (M * rows, W))[:bs * rows])
        g['chain_temps'][:bs].copy_(blob.device_view_f32('ctemps')[:bs])
        g['chain_topps'][:bs].copy_(blob.device_view_f32('ctopps')[:bs])
        g['v_temps'][:bs * rows].copy_(blob.device_view_f32('v_temps')[:bs * rows])
        g['v_topps'][:bs * rows].copy_(blob.device_view_f32('v_topps')[:bs * rows])

    def _sg_draft_chain(self, bs: int, use_cuda_graph: bool):
        g = self._sg
        k = self.config.spec_num_tokens
        seed = g['carry_seed'][:bs]
        for i in range(1, k):
            set_context(False,
                        slot_mapping=g['chain_slots'][i][:bs],
                        context_lens=g['chain_ctxlen'][i][:bs],
                        block_tables=g['chain_btables'][:bs],
                        cu_seqlens_q=g['chain_cu_q'][:bs + 1],
                        cu_seqlens_k=g['chain_cu_k'][i, :bs + 1],
                        max_context_len=self.config.max_model_len,
                        use_cuda_graph=use_cuda_graph,
                        block_size=self.block_size)
            seed, logits = self.draft_model.forward_decode(
                g['chain_pos'][i][:bs], seed, g['drafted_ids'][i - 1][:bs])
            ids = self._sg_sample(
                logits, g['chain_temps'][:bs], g['chain_topps'][:bs], g['chain_topks'][:bs],
                use_cuda_graph)
            g['drafted_ids'][i][:bs] = self.draft_model.map_to_target_ids(ids)

    def _sg_verify(self, bs: int, use_cuda_graph: bool = True):
        g = self._sg
        rows = self.config.spec_num_tokens + 1
        set_context(True,
                    cu_seqlens_q=g['v_cu_q'][:bs + 1],
                    cu_seqlens_k=g['v_cu_k'][:bs + 1],
                    max_seqlen_q=rows,
                    max_seqlen_k=self.config.max_model_len,
                    slot_mapping=g['v_slots'][:bs * rows],
                    context_lens=g['v_ctx_lens'][:bs * rows],
                    block_tables=self._verify_block_tables(g, bs, rows),
                    block_size=self.block_size)
        if self.state_runtime:
            self.state_runtime.bind_verify(get_context(), bs)
        hidden, aux = self.model(g['v_input_ids'][:bs * rows], g['v_positions'][:bs * rows])
        logits = self.backend.linear(
            hidden, getattr(self.model.lm_head, "weight_nz", self.model.lm_head.weight))
        g['v_token_ids'][:bs * rows] = self._sg_sample(
            logits, g['v_temps'][:bs * rows], g['v_topps'][:bs * rows], g['v_topks'][:bs * rows],
            use_cuda_graph)
        g['v_aux'][:bs * rows].copy_(aux)

    def _sg_acceptance(self, bs: int):
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        t = g['v_token_ids'][:bs * rows].view(bs, rows)
        g['d_pad'][:bs, :k].copy_(g['drafted_ids'][:k, :bs].transpose(0, 1))
        g['d_pad'][:bs, k].zero_()
        d = g['d_pad'][:bs, :k]
        match = t[:, :k] == d
        # 310P: avoid AI-CPU cumprod (crashes under current driver/CANN).
        # Equivalent to (match.cumprod(-1).sum(-1)): index of first mismatch (or k if all matched).
        _idx = torch.arange(k, device=match.device).expand_as(match)
        m_f = torch.where(match, torch.full_like(_idx, k), _idx).min(dim=-1).values.to(torch.float32)
        m = m_f.to(torch.int64)
        g['accept_m'][:bs] = m
        t_m = t.gather(1, m[:, None])
        ar_f = g['v_ar_f'][:rows]
        gathered = g['d_pad'][:bs].gather(1, g['v_ar_clamp'][:rows].expand(bs, rows))
        out = torch.where(ar_f.expand(bs, rows) < m_f[:, None], gathered, t_m.expand(bs, rows))
        g['commit_buf'][:bs, :rows] = torch.where(
            ar_f.expand(bs, rows) <= m_f[:, None], out, g['neg1_tok'][:bs])
        g['commit_buf'][:bs, rows] = (m_f + 1.0).to(torch.int64)

    def _sg_refill(self, bs: int, use_cuda_graph: bool = True):
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        seed_r = self.draft_model.project_aux(g['v_aux'][:bs * rows])
        t = g['v_token_ids'][:bs * rows].view(bs, rows)
        m_idx = g['accept_m'][:bs]
        ids_r = _refill_ids(t, m_idx, rows, k, ar_f=g['v_ar_f'][:rows], shift_idx=g['v_shift_idx'][:rows])
        set_context(True,
                    cu_seqlens_q=g['v_cu_q'][:bs + 1],
                    cu_seqlens_k=g['v_cu_k'][:bs + 1],
                    max_seqlen_q=rows,
                    max_seqlen_k=self.config.max_model_len,
                    slot_mapping=g['v_slots'][:bs * rows],
                    context_lens=g['v_ctx_lens'][:bs * rows],
                    block_tables=self._verify_block_tables(g, bs, rows),
                    block_size=self.block_size)
        h = self.draft_model.forward_prefill(
            g['v_positions'][:bs * rows], seed_r, ids_r)
        h_m = h.view(bs, rows, -1).gather(
            1, m_idx[:, None, None].expand(bs, 1, h.size(-1))).squeeze(1)
        logits = self.backend.linear(self.draft_model.norm(h_m), self.draft_model.lm_head.weight)
        ids = self._sg_sample(
            logits, g['chain_temps'][:bs], g['chain_topps'][:bs], g['chain_topks'][:bs],
            use_cuda_graph)
        g['drafted_ids'][0][:bs] = self.draft_model.map_to_target_ids(ids)
        g['carry_seed'][:bs] = h_m
        g['commit_buf'][:bs, rows + 1] = g['drafted_ids'][0][:bs]

    def _sg_round(self, bs: int, use_cuda_graph: bool):
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        self._sg_scatter(bs)
        self._sg_draft_chain(bs, use_cuda_graph)
        v = g['v_input_ids'][:bs * rows].view(bs, rows)
        v[:, 0] = g['chain_tokens'][:bs]
        v[:, 1:].copy_(g['drafted_ids'][:k, :bs].t())
        self._sg_verify(bs, use_cuda_graph)
        self._sg_acceptance(bs)
        self._sg_refill(bs, use_cuda_graph)

    @torch.inference_mode()
    def capture_spec_cudagraph(self):
        def prepare():
            k = self.config.spec_num_tokens
            rows = k + 1
            M, W = self._pin_M, self._pin_W
            blob = self._sblob
            blob.view_i64('chain_ctxlen', (k, M)).fill_(self._graph_pad_ctx())
            blob.view_i64('chain_slots', (k, M)).fill_(self._graph_pad_slot())
            blob.view_i64('btables', (M, W)).fill_(self._graph_pad_block())
            blob.view_i64('v_slots', (M, rows)).fill_(self._graph_pad_slot())
            blob.view_i64('v_ctx_lens').fill_(1)
            blob.view_i64('v_btables_exp', (M, rows, W)).fill_(self._graph_pad_block())
        self._capture_spec_graphs(
            prepare, lambda bs: self._sg_round(bs, use_cuda_graph=True),
            "Speculative")

    @torch.inference_mode()
    def _spec_round_graph(self, seqs: list[Sequence], bucket: int) -> list[list[int]]:
        n = self._fill_spec_meta(seqs)
        self._reset_spec_padding(n, bucket, False)
        if n < bucket:
            k = self.config.spec_num_tokens
            M = self._pin_M
            blob = self._sblob
            blob.view_i64('chain_pos', (k, M))[:, n:].fill_(-1)
            blob.view_i64('chain_ctxlen', (k, M))[:, n:].fill_(self._graph_pad_ctx())
            blob.view_i64('chain_slots', (k, M))[:, n:].fill_(self._graph_pad_slot())
        self._load_spec_carry(seqs, n)
        self._replay_spec_graph(bucket)
        return self._readback_spec_commit(seqs)

    @torch.inference_mode()
    def capture_cudagraph(self):
        try:
            if self.state_runtime:
                self.state_runtime.prepare_capture()
            config = self.config
            tc = config.model_config
            max_bs = self._graph_max_bs
            max_num_blocks = self._pin_W
            io = torch.zeros(2 * max_bs, dtype=torch.int64)
            sc = torch.zeros(max_bs, dtype=self.backend.graph_slot_mapping_dtype())
            pad_slot = self._graph_pad_slot()
            capture_block = self.backend.graph_capture_block(
                config.num_kvcache_blocks
            )
            capture_ctx = self.backend.graph_capture_context_len()
            if capture_ctx > 0 and capture_block >= 0:
                for module in self.model.modules():
                    k_cache = getattr(module, "k_cache", None)
                    v_cache = getattr(module, "v_cache", None)
                    if (isinstance(k_cache, torch.Tensor)
                            and isinstance(v_cache, torch.Tensor)
                            and k_cache.numel() and v_cache.numel()):
                        k_cache[capture_block].zero_()
                        v_cache[capture_block].zero_()
            sc.fill_(pad_slot)
            ctx = torch.zeros(max_bs, dtype=torch.int32)
            ctx.fill_(capture_ctx)
            input_ids, positions = io[:max_bs], io[max_bs:]
            slot_mapping, context_lens = sc, ctx
            block_tables = torch.full(
                (max_bs, max_num_blocks), capture_block, dtype=torch.int32
            )
            cu_seqlens_q = torch.arange(max_bs + 1, dtype=torch.int32)
            cu_seqlens_k = (
                torch.arange(max_bs + 1, dtype=torch.int32) * capture_ctx
            )
            outputs = torch.zeros(max_bs, tc.hidden_size, dtype=self.dtype)
            temperatures = torch.ones(max_bs, dtype=torch.float32)
            top_ps = torch.ones(max_bs, dtype=torch.float32)
            top_ks = torch.full((max_bs,), tc.vocab_size, dtype=torch.int64)
            token_ids = torch.zeros(max_bs, dtype=torch.int64)
            self.graph_bs = _graph_buckets(max_bs)
            self.graphs = {}
            self.graph_pool = None
            ids_d = self._dblob.device_view_i64('ids')
            pos_d = self._dblob.device_view_i64('pos')
            slots_d = self._dblob.device_view_i64('slots')
            ctx_d = self._dblob.device_view_i64('ctx')
            ctx_cu_d = self._dblob.device_view_i64('ctx_cu')
            bt_d = self._dblob.device_view_i64('bt', (max_bs, max_num_blocks))
            topks_d = self._dblob.device_view_i64('topks')
            temps_d = self._dblob.device_view_f32('temperatures')
            topps_d = self._dblob.device_view_f32('top_ps')
            self._dblob.view_i64('slots')[:max_bs] = pad_slot
            self._dblob.view_i64('ctx')[:max_bs] = capture_ctx
            self._dblob.view_i64('ctx_cu')[:max_bs + 1].copy_(
                torch.arange(max_bs + 1, dtype=torch.int64) * capture_ctx
            )
            self._dblob.view_i64(
                'bt', (max_bs, max_num_blocks)
            ).fill_(capture_block)
            self._dblob.h2d()

            def scatter(bs):
                input_ids[:bs].copy_(ids_d[:bs])
                positions[:bs].copy_(pos_d[:bs])
                slot_mapping[:bs].copy_(slots_d[:bs])
                context_lens[:bs].copy_(ctx_d[:bs])
                cu_seqlens_k[:bs + 1].copy_(ctx_cu_d[:bs + 1])
                block_tables[:bs].copy_(bt_d[:bs])
                top_ks[:bs].copy_(topks_d[:bs])
                temperatures[:bs].copy_(temps_d[:bs])
                top_ps[:bs].copy_(topps_d[:bs])

            for bs in reversed(self.graph_bs):
                graph = self.backend.new_graph()
                set_context(
                    False,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs],
                    cu_seqlens_q=cu_seqlens_q[:bs + 1],
                    cu_seqlens_k=cu_seqlens_k[:bs + 1],
                    max_context_len=config.max_model_len,
                    use_cuda_graph=True,
                    block_size=self.block_size,
                )
                if self.state_runtime:
                    self.state_runtime.bind_decode(get_context(), bs)
                scatter(bs)
                for _ in range(self.backend.graph_warmup_iters()):
                    hidden = self.model(input_ids[:bs], positions[:bs])    # warmup
                    if self.graph_sample:
                        token_ids[:bs] = self.sampler(
                            self.model.compute_logits(hidden), temperatures[:bs], top_ps[:bs], top_ks[:bs])
                    self.backend.synchronize()
                with self.backend.graph_ctx(graph, pool=self.graph_pool):
                    scatter(bs)
                    hidden = self.model(input_ids[:bs], positions[:bs])    # capture
                    if self.graph_sample:
                        token_ids[:bs] = self.sampler(
                            self.model.compute_logits(hidden), temperatures[:bs], top_ps[:bs], top_ks[:bs])
                    else:
                        outputs[:bs].copy_(hidden)
                if self.graph_pool is None:
                    self.graph_pool = self.backend.graph_pool(graph)
                self.graphs[bs] = graph
                if (self.graph_sample and self.backend.supports_greedy_graph_variants
                        and os.getenv("FLASHENGINE_DECODE_GREEDY_GRAPH", "1") != "0"):
                    try:
                        greedy_graph = self.backend.new_graph()
                        with self.backend.graph_ctx(greedy_graph, pool=self.graph_pool):
                            scatter(bs)
                            greedy_hidden = self.model(input_ids[:bs], positions[:bs])
                            token_ids[:bs].copy_(self.model.compute_logits(greedy_hidden).argmax(dim=-1))
                        self.backend.synchronize()
                        self.decode_greedy_graphs[bs] = greedy_graph
                    except Exception as exc:
                        print(f"[FlashEngine] Greedy decode graph bucket {bs} unavailable; "
                              f"using sampling graph: {type(exc).__name__}: {exc}", flush=True)
                self.backend.synchronize()
                reset_context()

            self.graph_vars = dict(
                block_tables=block_tables,
                outputs=outputs,
                top_ks=top_ks,
                token_ids=token_ids,
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                temperatures=temperatures,
                top_ps=top_ps,
            )
        except Exception as e:
            print(
                "[FlashEngine] Decode graph capture failed; falling back to eager mode: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()
            reset_context()
            self.backend.synchronize()
            self.graphs = None
            self.decode_greedy_graphs.clear()
            self.graph_pool = None
            self.backend.empty_cache()
