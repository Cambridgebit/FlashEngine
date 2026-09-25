import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        self._closed = False
        self._atexit_callback = self.close
        # Register before model/process initialization so partially constructed
        # engines are cleaned up when loading or graph capture raises.
        atexit.register(self._atexit_callback)
        engine_t0 = perf_counter()
        config_fields = {field.name for field in fields(Config) if field.init}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        print(
            f"[nano-vllm] Initializing engine for model {config.model} "
            f"on backend {config.backend if config.backend != 'auto' else 'auto-detect'}...",
            flush=True,
        )
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        tokenizer_t0 = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        tokenizer_load_s = perf_counter() - tokenizer_t0
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        if (self.model_runner.spec_enabled
                and config.spec_method in ("dflash", "dspark")
                and not self.model_runner.dflash_paged
                and config.enable_prefix_cache):
            self.scheduler.block_manager.cache_hit_validator = (
                self.model_runner.has_dflash_prefix_snapshot)
        # Spec state follows KV lifetime: preemption invalidates both target KV
        # and per-sequence draft carry/context state.
        self.scheduler.on_preempt = self._on_preempt_release
        # 首请求冷启动预热：warmup_model 只跑 prefill，首个 decode / spec 轮会
        # 触发 decode attention kernel（flash-attn with_kvcache）与图首放的一次性
        # 初始化——实测同进程首轮 ~0.55s、其后 ~0.06s。此处跑一次极短生成
        # （max_tokens=2 保证走满一次 decode / spec 轮）把该开销前移到启动期，
        # 使首个真实请求 TTFT 与稳态一致（server 默认无 benchmark 的多次预热）。
        runtime_warmup_t0 = perf_counter()
        self.generate(
            ["warmup"],
            SamplingParams(temperature=0.0, max_tokens=2),
            use_tqdm=False,
        )
        runtime_warmup_s = perf_counter() - runtime_warmup_t0
        self._print_startup_summary(
            config,
            tokenizer_load_s=tokenizer_load_s,
            runtime_warmup_s=runtime_warmup_s,
            engine_startup_s=perf_counter() - engine_t0,
        )

    def _print_startup_summary(
        self,
        config: Config,
        *,
        tokenizer_load_s: float,
        runtime_warmup_s: float,
        engine_startup_s: float,
    ):
        runner = self.model_runner
        architecture = (
            (getattr(config.hf_config, "architectures", None) or [None])[0]
            or getattr(config.model_config, "model_type", None)
            or type(runner.model).__name__
        )
        backend_info = runner.backend.runtime_info()
        speculative = "enabled" if runner.spec_enabled else "disabled"
        draft_id = config.spec_model if runner.spec_enabled else "n/a"
        spec_method = config.spec_method if runner.spec_enabled else "n/a"
        spec_tokens = config.spec_num_tokens if runner.spec_enabled else "n/a"
        stats = runner.startup_stats
        block_manager = self.scheduler.block_manager
        usable_kv_blocks = len(block_manager.blocks)
        kv_capacity_tokens = usable_kv_blocks * config.kvcache_block_size
        max_concurrency = kv_capacity_tokens / config.max_model_len
        graph_buckets = "n/a"
        if runner.spec_enabled and getattr(runner, "spec_graphs", None):
            graph_buckets = getattr(runner, "spec_graph_bs", list(runner.spec_graphs))
        elif getattr(runner, "graph_runner", None) is not None:
            graph_buckets = runner.graph_runner.capture_sizes
        elif getattr(runner, "graphs", None):
            graph_buckets = getattr(runner, "graph_bs", list(runner.graphs))

        fields = [
            ("Model ID", config.model),
            ("Architecture", architecture),
            ("Backend", runner.backend.name),
            ("DType", str(runner.dtype).removeprefix("torch.")),
            ("Tensor parallel size", config.tensor_parallel_size),
            ("Max model length", config.max_model_len),
            ("Max sequences", config.max_num_seqs),
            ("Max batched tokens", config.max_num_batched_tokens),
            ("Continuous batching", "enabled (fair prefill/decode)"),
            ("Chunked prefill", "enabled" if config.enable_chunked_prefill else "disabled"),
            ("Max partial prefills", config.max_num_partial_prefills),
            ("KV-cache block size", config.kvcache_block_size),
            ("KV-cache blocks", usable_kv_blocks),
            ("KV-cache token capacity", f"{kv_capacity_tokens:,}"),
            ("Maximum KV concurrency", f"{max_concurrency:.2f}x @ {config.max_model_len:,} tokens"),
            ("Prefix cache", "enabled" if config.enable_prefix_cache else "disabled"),
            ("Speculative decoding", speculative),
            ("Speculative method", spec_method),
            ("Draft model ID", draft_id),
            ("Speculative tokens", spec_tokens),
            ("Graph mode", runner.graph_mode()),
            ("Graph capture buckets", graph_buckets),
            ("Attention backend", backend_info["attention_backend"]),
            ("Graph backend", backend_info["graph_backend"]),
            ("KV-cache layout", backend_info["kv_cache_layout"]),
            ("Fused ops", backend_info["fused_ops"]),
            ("Weight layout conversions", int(stats["weight_layout_conversions"])),
        ]
        if runner.spec_enabled and runner.config.spec_method in ("dflash", "dspark"):
            fields.insert(
                -1,
                ("Block-draft attention path",
                 backend_info.get("dflash_backend", "dense-KV/SDPA")),
            )
            fields.insert(-1, ("Draft prefix state",
                               "paged KV (anchor block excluded)" if runner.dflash_paged
                               else "bounded CPU snapshots (1024 MiB)"))
        if runner.spec_enabled and runner.config.spec_method == "dspark":
            fields.insert(-1, ("DSpark verification", "fixed-k; adaptive confidence disabled"))
        print("[nano-vllm] Engine configuration:", flush=True)
        for key, value in fields:
            print(f"[nano-vllm]   {key}: {value}", flush=True)

        print("[nano-vllm] Startup timings:", flush=True)
        print(
            f"[nano-vllm]   Model loading: {stats['model_load_s']:.3f} s",
            flush=True,
        )
        print(
            f"[nano-vllm]   Model warmup: {stats['model_warmup_s']:.3f} s",
            flush=True,
        )
        print(
            f"[nano-vllm]   KV-cache allocation: "
            f"{stats['kv_cache_allocation_s']:.3f} s",
            flush=True,
        )
        print(
            f"[nano-vllm]   Graph capture: {stats['graph_capture_s']:.3f} s",
            flush=True,
        )
        print(
            f"[nano-vllm]   Tokenizer loading: {tokenizer_load_s:.3f} s",
            flush=True,
        )
        print(
            f"[nano-vllm]   Runtime warmup: {runtime_warmup_s:.3f} s",
            flush=True,
        )
        print(
            f"[nano-vllm] Engine ready in {engine_startup_s:.3f} s",
            flush=True,
        )

    def close(self):
        """Permanently close the engine and release device/process resources."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        callback = getattr(self, "_atexit_callback", None)
        if callback is not None:
            try:
                atexit.unregister(callback)
            except Exception:
                pass

        runner = getattr(self, "model_runner", None)
        if runner is not None:
            try:
                runner.call("exit")
            except Exception as exc:
                print(f"[nano-vllm] Warning: model runner shutdown failed: {exc}", flush=True)
            finally:
                del self.model_runner
        else:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()

        # Child ranks should leave through ModelRunner.exit. If a failed rank
        # does not cooperate, terminate it so it cannot retain an NPU context.
        for process in getattr(self, "ps", []):
            process.join(timeout=30)
            if process.is_alive():
                print(
                    f"[nano-vllm] Warning: worker process {process.pid} did not exit; "
                    "terminating it.",
                    flush=True,
                )
                process.terminate()
                process.join(timeout=5)
        self.ps = []
        self.events = []

    def exit(self):
        """Backward-compatible alias for :meth:`close`."""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def _on_preempt_release(self, seq):
        """Release per-sequence accelerator state after KV preemption."""
        if self.model_runner.spec_enabled:
            self.model_runner.call("drop_spec_state", seq.seq_id)

    def step(self):
        if self.model_runner.spec_enabled:
            return self.step_spec()
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def step_spec(self):
        """EAGLE3 投机解码引擎步（仅 TP=1）。

        prefill 步与普通路径一致（额外执行 draft prefill 并记录 h_seed）；
        decode 步为一整轮投机：预分配容量 → draft 起草 → target 验证 → 多 token 提交。
        """
        seqs, is_prefill = self.scheduler.schedule()
        if is_prefill:
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
            token_ids = self.model_runner.call("run_prefill_spec", seqs)
            if (self.model_runner.config.spec_method in ("dflash", "dspark")
                    and not self.model_runner.dflash_paged):
                self.model_runner.save_dflash_prefix_snapshots(seqs)
            self.scheduler.postprocess(seqs, token_ids, True)
            self._drop_finished_spec_state(seqs)
            return self._collect_outputs(seqs), num_tokens
        num_draft = self.model_runner.config.spec_num_tokens
        active, capped = self.scheduler.prepare_spec_round(num_draft)
        if capped:
            # 顶格结束的序列：释放其投机跨轮状态，并作为完成输出上报。
            self._drop_finished_spec_state(capped)
        if not active:
            return self._collect_outputs(capped), 0
        committed = self.model_runner.call("spec_round", active)
        finished = self.scheduler.commit_spec(active, committed)
        self._drop_finished_spec_state(finished)
        num_tokens = -sum(len(tokens) for tokens in committed)
        return self._collect_outputs(capped + active), num_tokens

    def _drop_finished_spec_state(self, seqs):
        for seq in seqs:
            if seq.is_finished:
                self.model_runner.call("drop_spec_state", seq.seq_id)

    def _collect_outputs(self, seqs):
        return [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
