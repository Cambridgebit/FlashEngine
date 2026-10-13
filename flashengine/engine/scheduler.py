from collections import deque

from flashengine.config import Config
from flashengine.engine.block_manager import BlockManager
from flashengine.engine.sequence import Sequence, SequenceStatus


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.max_num_partial_prefills = config.max_num_partial_prefills
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        # 越过 max_model_len 继续写 KV / 取 token 会让注意力读到从未写入的槽位，
        self.max_model_len = config.max_model_len
        self.on_preempt = None
        self._prefer_decode = False
        self.stats = {
            "steps": 0,
            "prefill_steps": 0,
            "decode_steps": 0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "chunked_prefill_steps": 0,
            "admitted_requests": 0,
            "preemptions": 0,
        }

        num_blocks = config.num_kvcache_blocks
        backend = config.backend_impl
        if backend.reserve_dummy_kv_block():
            if num_blocks <= 1:
                raise RuntimeError(
                    "The backend dummy KV block requires num_kvcache_blocks > 1"
                )
            num_blocks -= 1
        self.block_manager = BlockManager(
            num_blocks,
            config.kvcache_block_size,
            config.enable_prefix_cache,
            keep_last_block_unhashed=config.keep_last_block_unhashed,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._lru_clock = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _num_partial_prefills(self) -> int:
        return sum(bool(seq.block_table) for seq in self.waiting)

    def _num_active(self) -> int:
        # A partial prefill owns KV blocks and optional draft state just like a
        return len(self.running) + self._num_partial_prefills()

    def _touch(self, seq: Sequence):
        self._lru_clock += 1
        seq._lru_tick = self._lru_clock

    def _preempt_lru_for(self, seq: Sequence) -> bool:
        while self.block_manager.can_allocate(seq) == -1 and self.running:
            victim = min(self.running, key=lambda item: getattr(item, "_lru_tick", 0))
            self.running.remove(victim)
            self.preempt(victim)
            try:
                self.waiting.remove(victim)
            except ValueError:
                pass
            self.waiting.append(victim)
        return self.block_manager.can_allocate(seq) != -1

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled: list[Sequence] = []
        num_batched_tokens = 0
        partial_prefills = self._num_partial_prefills()
        active = len(self.running) + partial_prefills

        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            was_partial = bool(seq.block_table)
            if not was_partial and active >= self.max_num_seqs:
                break

            remaining_budget = self.max_num_batched_tokens - num_batched_tokens
            if remaining_budget <= 0:
                break

            if not was_partial:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    if not self._preempt_lru_for(seq):
                        break
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                remaining_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_cached_blocks = 0
                remaining_tokens = seq.num_tokens - seq.num_cached_tokens

            will_be_partial = remaining_tokens > remaining_budget
            if will_be_partial:
                if not self.enable_chunked_prefill:
                    break
                if not was_partial and partial_prefills >= self.max_num_partial_prefills:
                    break

            if not was_partial:
                self.block_manager.allocate(seq, num_cached_blocks)
                active += 1
                self.stats["admitted_requests"] += 1

            seq.num_scheduled_tokens = min(remaining_tokens, remaining_budget)
            if seq.num_scheduled_tokens <= 0:
                raise RuntimeError(
                    f"Scheduler made no prefill progress for sequence {seq.seq_id}."
                )
            num_batched_tokens += seq.num_scheduled_tokens
            scheduled.append(seq)
            self._touch(seq)

            complete = (
                seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens
            )
            if complete:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                if was_partial:
                    partial_prefills -= 1
            else:
                if not was_partial:
                    partial_prefills += 1
                break

        if scheduled:
            self.stats["steps"] += 1
            self.stats["prefill_steps"] += 1
            self.stats["prefill_tokens"] += num_batched_tokens
            if any(
                seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens
                for seq in scheduled
            ):
                self.stats["chunked_prefill_steps"] += 1
        return scheduled

    def _schedule_decode(self) -> list[Sequence]:
        scheduled: list[Sequence] = []
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = min(self.running,
                                 key=lambda item: getattr(item, "_lru_tick", 0))
                    self.running.remove(victim)
                    self.preempt(victim)
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled.append(seq)
                self._touch(seq)
        self.running.extendleft(reversed(scheduled))
        if scheduled:
            self.stats["steps"] += 1
            self.stats["decode_steps"] += 1
            self.stats["decode_tokens"] += len(scheduled)
        return scheduled

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.waiting and (not self.running or not self._prefer_decode):
            prefills = self._schedule_prefill()
            if prefills:
                self._prefer_decode = bool(self.running)
                return prefills, True

        decodes = self._schedule_decode()
        if decodes:
            self._prefer_decode = False
            return decodes, False

        prefills = self._schedule_prefill()
        if prefills:
            self._prefer_decode = bool(self.running)
            return prefills, True

        raise RuntimeError(
            "Scheduler cannot make progress: no KV capacity is available for "
            "the waiting request and no running request can be decoded."
        )

    def get_stats(self) -> dict:
        result = dict(self.stats)
        result.update(
            running=len(self.running),
            waiting=len(self.waiting),
            partial_prefills=self._num_partial_prefills(),
            active=self._num_active(),
        )
        return result

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.stats["preemptions"] += 1
        if self.on_preempt is not None:
            self.on_preempt(seq)
        self.waiting.appendleft(seq)

    def prepare_spec_round(self, num_draft_tokens: int) -> tuple[list[Sequence], list[Sequence]]:
        active: list[Sequence] = []
        capped: list[Sequence] = []
        while self.running:
            seq = self.running.popleft()
            if len(seq) >= self.max_model_len:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                capped.append(seq)
                continue
            target = min(len(seq) + num_draft_tokens + 1, self.max_model_len)
            if self.block_manager.ensure_capacity(seq, target):
                active.append(seq)
            else:
                self.preempt(seq)
        self.running.extend(active)
        return active, capped

    def _maybe_finish(self, seq: Sequence, token_id: int) -> bool:
        return (
            (not seq.ignore_eos and token_id == self.eos)
            or seq.num_completion_tokens >= seq.max_tokens
            or seq.num_tokens >= self.max_model_len   # 到上下文顶：硬停，不再越界写 KV
        )

    def commit_spec(
        self, seqs: list[Sequence], tokens_list: list[list[int]]
    ) -> list[Sequence]:
        finished = []
        for seq, tokens in zip(seqs, tokens_list):
            for token_id in tokens:
                if seq.num_tokens >= self.max_model_len:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    finished.append(seq)
                    break
                seq.append_token(token_id)
                seq.num_cached_tokens = seq.num_tokens
                if self._maybe_finish(seq, token_id):
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    finished.append(seq)
                    break
        return finished

    def postprocess(
        self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool
    ):
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
