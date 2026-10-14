import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from flashengine.config import Config
from flashengine.sampling_params import SamplingParams
from flashengine.engine.sequence import Sequence
from flashengine.engine.scheduler import Scheduler
from flashengine.engine.model_runner import ModelRunner
from flashengine.engine.observer import EngineObserver


class LLMEngine:

    def __init__(self, model, **kwargs):
        self._closed = False
        self._atexit_callback = self.close
        atexit.register(self._atexit_callback)
        engine_t0 = perf_counter()
        config_fields = {field.name for field in fields(Config) if field.init}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        print(
            f"[FlashEngine] Initializing engine for model {config.model} "
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
        self.observer: EngineObserver | None = None
        self._step_impl = (
            self._step_spec if self.model_runner.spec_enabled else self._step_standard
        )
        tokenizer_t0 = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        tokenizer_load_s = perf_counter() - tokenizer_t0
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.scheduler.block_manager.cache_hit_validator = (
            self.model_runner.prefix_cache_validator()
        )
        self.scheduler.block_manager.prefix_state_validator = (
            self.model_runner.state_runtime.has_prefix
            if self.model_runner.state_runtime else None
        )
        self.scheduler.on_preempt = self._on_preempt_release
        runtime_warmup_t0 = perf_counter()
        self.generate(
            ["warmup"],
            SamplingParams(temperature=0.0, max_tokens=2),
            use_tqdm=False,
        )
        runtime_warmup_s = perf_counter() - runtime_warmup_t0
        self.reset_measurement_stats()
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
        architecture = ((getattr(config.hf_config, "architectures", None) or [None])[0]
                        or getattr(config.model_config, "model_type", None)
                        or type(runner.model).__name__)
        backend_info = runner.backend.runtime_info()
        stats = runner.startup_stats
        usable_kv_blocks = len(self.scheduler.block_manager.blocks)
        kv_capacity_tokens = usable_kv_blocks * config.kvcache_block_size
        fields = [
            ("Model ID", config.model),
            ("Architecture", architecture),
            ("Backend", runner.backend.name),
            ("DType", str(runner.dtype).removeprefix("torch.")),
            ("Limits", f"len={config.max_model_len}, seqs={config.max_num_seqs}, "
             f"batch_tokens={config.max_num_batched_tokens}, tp={config.tensor_parallel_size}"),
            ("KV cache", f"{usable_kv_blocks} x {config.kvcache_block_size} = "
             f"{kv_capacity_tokens:,} tokens"),
            ("Prefix cache", "enabled" if config.enable_prefix_cache else "disabled"),
            ("Speculative", f"{config.spec_method or 'off'} "
             f"(k={config.spec_num_tokens if runner.spec_enabled else 0})"),
            ("Graph", f"{runner.graph_mode()} buckets={runner.graph_buckets() or 'n/a'}"),
            ("Backend runtime", ", ".join(f"{k}={v}" for k, v in backend_info.items())),
        ]
        fields.extend(runner.startup_details())
        print("[FlashEngine] Engine configuration:", flush=True)
        for key, value in fields:
            print(f"[FlashEngine]   {key}: {value}", flush=True)

        print("[FlashEngine] Startup timings:", flush=True)
        timings = (
            ("Model loading", stats["model_load_s"]),
            ("Model warmup", stats["model_warmup_s"]),
            ("KV-cache allocation", stats["kv_cache_allocation_s"]),
            ("Graph capture", stats["graph_capture_s"]),
            ("Tokenizer loading", tokenizer_load_s),
            ("Runtime warmup", runtime_warmup_s),
        )
        for label, seconds in timings:
            print(f"[FlashEngine]   {label}: {seconds:.3f} s", flush=True)
        print(f"[FlashEngine] Engine ready in {engine_startup_s:.3f} s", flush=True)

    def close(self):
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
                print(f"[FlashEngine] Warning: model runner shutdown failed: {exc}", flush=True)
            finally:
                del self.model_runner
        else:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()

        for process in getattr(self, "ps", []):
            process.join(timeout=30)
            if process.is_alive():
                print(
                    f"[FlashEngine] Warning: worker process {process.pid} did not exit; "
                    "terminating it.",
                    flush=True,
                )
                process.terminate()
                process.join(timeout=5)
        self.ps = []
        self.events = []

    def exit(self):
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
        self.model_runner.call("drop_seq_state", seq.seq_id)

    def _notify_schedule(self, seqs, is_prefill):
        if self.observer is not None:
            self.observer.on_schedule(seqs, is_prefill, perf_counter())

    def step(self):
        return self._step_impl()

    def _step_standard(self):
        seqs, is_prefill = self.scheduler.schedule()
        self._notify_schedule(seqs, is_prefill)
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        self._drop_finished_spec_state(seqs)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def _step_spec(self):
        seqs, is_prefill = self.scheduler.schedule()
        self._notify_schedule(seqs, is_prefill)
        if is_prefill:
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
            token_ids = self.model_runner.call("run_prefill_spec", seqs)
            self.model_runner.after_spec_prefill(seqs)
            self.scheduler.postprocess(seqs, token_ids, True)
            self._drop_finished_spec_state(seqs)
            return self._collect_outputs(seqs), num_tokens
        num_draft = self.model_runner.config.spec_num_tokens
        active, capped = self.scheduler.prepare_spec_round(num_draft)
        if capped:
            self._drop_finished_spec_state(capped)
        if not active:
            return self._collect_outputs(capped), 0
        limited = [seq for seq in active if self.config.max_model_len - len(seq) < num_draft + 1]
        normal = [seq for seq in active if seq not in limited]
        by_id = {}
        if normal:
            by_id.update(zip((seq.seq_id for seq in normal), self.model_runner.call("spec_round", normal)))
        if limited:
            by_id.update(zip((seq.seq_id for seq in limited), self.model_runner.call("context_limited_spec_round", limited)))
        committed = [by_id[seq.seq_id] for seq in active]
        if self.observer is not None:
            self.observer.on_commit(active, committed, perf_counter())
        finished = self.scheduler.commit_spec(active, committed)
        self._drop_finished_spec_state(finished)
        num_tokens = -sum(len(tokens) for tokens in committed)
        return self._collect_outputs(capped + active), num_tokens

    def step_spec(self):
        return self._step_spec()

    def _drop_finished_spec_state(self, seqs):
        for seq in seqs:
            if seq.is_finished:
                self.model_runner.call("drop_seq_state", seq.seq_id)

    def _collect_outputs(self, seqs):
        return [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

    def is_finished(self):
        return self.scheduler.is_finished()

    def runtime_snapshot(self) -> dict:
        runner = self.model_runner.runtime_snapshot()
        block_manager = self.scheduler.block_manager
        return {
            "engine": {
                "model": self.config.model,
                "backend": self.model_runner.backend.name,
                "backend_info": self.model_runner.backend.runtime_info(),
                "dtype": str(self.model_runner.dtype),
                "graph_mode": runner["mode"],
                "enforce_eager": self.config.enforce_eager,
                "max_model_len": self.model_runner.config.max_model_len,
                "max_num_seqs": self.model_runner.config.max_num_seqs,
                "max_num_batched_tokens": self.model_runner.config.max_num_batched_tokens,
                "chunked_prefill": self.model_runner.config.enable_chunked_prefill,
                "max_num_partial_prefills": self.config.max_num_partial_prefills,
                "prefix_cache": self.model_runner.config.enable_prefix_cache,
                "block_size": self.config.kvcache_block_size,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "spec_model": self.config.spec_model,
            },
            "scheduler": self.scheduler.get_stats(),
            "cache": {
                **block_manager.get_prefix_stats(),
                "used_blocks": len(block_manager.used_block_ids),
                "total_blocks": len(block_manager.blocks),
            },
            **runner,
        }

    def reset_measurement_stats(self):
        self.model_runner.reset_measurement_stats()
        if hasattr(self, "scheduler"):
            self.scheduler.block_manager.reset_prefix_stats()

    def execution_path(self, temperature: float) -> str:
        return self.model_runner.execution_path(temperature)

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
