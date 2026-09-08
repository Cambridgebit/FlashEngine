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
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
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
