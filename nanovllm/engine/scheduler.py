from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.is_hybrid = config.is_hybrid
        # 状态槽后端门控：hybrid（GDN 状态池）与 dense+dflash（draft ctx 稠密缓冲
        # 槽）的 per-seq 状态槽都在 prefill 分配、decode 持有、finish/preempt 释放
        # ——admission 必须保证「持有槽的序列总数 ≤ max_num_seqs」，否则排队请求
        # 会在 decode 开始前把槽耗尽（dense+dflash 机械冒烟实测：ms=2 下 p2 仍占
        # 槽时 p3/p4 被同一 prefill 批放行 → draft ctx 槽耗尽 RuntimeError）。
        # qwen3+eagle3 无固定状态槽（spec_carry 按 seq_id 记账、finish 即释放），
        # 门控关闭，行为零变化。
        self.state_slots_gate = config.is_hybrid or config.spec_method == "dflash"
        # hybrid GDN 状态槽释放回调（llm_engine 注入为 model_runner.drop_seq_state
        # 的包装；回调解耦——scheduler 不持有 model_runner 引用。qwen3 路径恒
        # None，preempt 零开销直通）。
        self.on_preempt = None
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size,
                                          config.enable_prefix_cache)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        # 状态槽后端 admission 门控（hybrid GDN 池 / dense+dflash draft ctx 槽）：
        # 已 prefill 的序列即使在 prefill 批次间停滞（本引擎 prefill/decode 不同
        # 批）也持有槽位，若不限制「持有槽的序列总数」，排队请求会在 decode 开始
        # 前把槽耗尽。故门控后端下追加「步首 running + 本步新 prefill ≤
        # max_num_seqs」（vLLM MambaManager 同思路：无空闲状态槽不放行新请求；
        # preempt/finish 缩减 running 后自然放行）。注意用循环前快照：整段 prefill
        # 的序列在循环内同时进入 running 与 scheduled_seqs，用实时 len(running)
        # 会双计。
        # 非门控后端（qwen3+eagle3 / 非 spec）短路恒真（快照变量不求值），条件与
        # 原式等价，行为零变化。
        running_holders = len(self.running)
        while (self.waiting and len(scheduled_seqs) < self.max_num_seqs
               and (not self.state_slots_gate
                    or running_holders + len(scheduled_seqs) < self.max_num_seqs)):
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # hybrid 核实（阶段1b）：config 断言 max_num_batched_tokens >=
            # max_model_len 下，首序列 remaining 即 ≥ 任意整段 prompt，min() 不截断、
            # 本分支「仅首序列可切分」不可达 → hybrid prefill 恒整段（无 chunk）。
            # （prompt 超过 max_model_len 属用户错误，非本断言保护范围。）
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        if self.on_preempt is not None:
            # hybrid：抢占即丢 GDN 状态槽（正确性优先——状态与 KV 一同作废，
            # 重 prefill 时零初始化重建）。回调由 llm_engine 注入，见 __init__。
            self.on_preempt(seq)
        self.waiting.appendleft(seq)

    def prepare_spec_round(self, num_draft_tokens: int) -> list[Sequence]:
        """投机解码轮准备：为每个 running 序列预分配 len+k 的 block 容量。

        容量不足时抢占该序列（丢弃 KV，退回 waiting 等待重 prefill）。
        """
        active = []
        while self.running:
            seq = self.running.popleft()
            # +1：draft 流最深写槽 = len+k-3，verify 覆盖到 len+k-1，留 1 余量。
            if self.block_manager.ensure_capacity(seq, len(seq) + num_draft_tokens + 1):
                active.append(seq)
            else:
                self.preempt(seq)
        self.running.extend(active)
        return active

    def _maybe_finish(self, seq: Sequence, token_id: int) -> bool:
        """完成判定（commit_spec / postprocess 共用）：命中 eos 或补全长度达 max_tokens。"""
        return (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens

    def commit_spec(self, seqs: list[Sequence], tokens_list: list[list[int]]) -> list[Sequence]:
        """投机解码轮提交：逐 token 追加（遇 eos / max_tokens 提前截断并结束序列）。

        投机解码产生的 token 不做 block hash（前缀缓存仅覆盖 prefill 阶段，
        避免被拒绝的 draft token 污染哈希表）。
        """
        finished = []
        for seq, tokens in zip(seqs, tokens_list):
            for token_id in tokens:
                seq.append_token(token_id)
                seq.num_cached_tokens = seq.num_tokens
                if self._maybe_finish(seq, token_id):
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    finished.append(seq)
                    break
        return finished

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if self._maybe_finish(seq, token_id):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
