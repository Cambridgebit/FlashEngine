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
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
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
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # hybrid GDN 状态槽：preempt 释放回调（engine 侧解耦注入；qwen3 路径
        # is_hybrid=False 恒短路，零开销）。finish 侧释放见 step()。
        self.scheduler.on_preempt = self._on_preempt_release
        atexit.register(self.exit)

    def exit(self):
        # 幂等：atexit 会再次调用；init 未完成（如 kv 预算不足抛错）时
        # model_runner 不存在，仅拆除可能已建立的 NCCL 组。
        if not hasattr(self, "model_runner"):
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def _on_preempt_release(self, seq):
        """hybrid GDN 状态槽释放（scheduler.preempt 回调）：抢占即丢状态+KV，
        后续重 prefill 零初始化重建（正确性优先）。"""
        if self.model_runner.is_hybrid:
            self.model_runner.call("drop_seq_state", seq.seq_id)

    def _drop_finished_gdn_state(self, seqs):
        """hybrid：本步内结束的序列释放 GDN 状态槽（沿用 finish 后 call
        drop_spec_state 的既有模式；qwen3 路径由 is_hybrid 门控，不进本函数）。"""
        for seq in seqs:
            if seq.is_finished:
                self.model_runner.call("drop_seq_state", seq.seq_id)

    def step(self):
        if self.model_runner.spec_enabled:
            return self.step_spec()
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if self.model_runner.is_hybrid:
            # hybrid：序列 finish 即释放 GDN 状态槽（spec 与 hybrid 互斥，见
            # model_runner 的 spec 守卫，故两条 drop 路径不会同时生效）。
            self._drop_finished_gdn_state(seqs)
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
            self.scheduler.postprocess(seqs, token_ids, True)
            self._drop_finished_spec_state(seqs)
            return self._collect_outputs(seqs), num_tokens
        num_draft = self.model_runner.config.spec_num_tokens
        active = self.scheduler.prepare_spec_round(num_draft)
        if not active:
            return [], 0
        committed = self.model_runner.call("spec_round", active)
        finished = self.scheduler.commit_spec(active, committed)
        self._drop_finished_spec_state(finished)
        num_tokens = -sum(len(tokens) for tokens in committed)
        return self._collect_outputs(active), num_tokens

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
