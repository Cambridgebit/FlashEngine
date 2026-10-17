"""单一驱动线程使用的公开请求句柄、取消和快照接口。"""

import math
from flashengine.engine.sequence import Sequence, SequenceStatus


class RequestCapacityError(RuntimeError):
    pass


class EngineRequests:
    def add_request(self, prompt, sampling_params):
        if getattr(self, "_closed", False):
            raise RuntimeError("引擎已关闭")
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if (not isinstance(prompt, (list, tuple)) or not prompt
                or any(isinstance(t, bool) or not isinstance(t, int)
                       or not 0 <= t < self.config.model_config.vocab_size for t in prompt)):
            raise ValueError("prompt 必须为非空文本或词表内的 token IDs")
        sp = sampling_params
        if (isinstance(sp.max_tokens, bool) or not isinstance(sp.max_tokens, int)
                or sp.max_tokens < 1 or isinstance(sp.temperature, bool)
                or not math.isfinite(sp.temperature) or sp.temperature < 0
                or isinstance(sp.top_p, bool) or not isinstance(sp.ignore_eos, bool)
                or not math.isfinite(sp.top_p) or not 0 < sp.top_p <= 1
                or isinstance(sp.top_k, bool) or not isinstance(sp.top_k, int) or sp.top_k < 0):
            raise ValueError("采样参数或 max_tokens 无效")
        if len(prompt) + sp.max_tokens > self.config.max_model_len:
            raise ValueError("prompt + max_tokens 超过 max_model_len；请显式调整请求")
        capacity = len(self.scheduler.block_manager.blocks) * self.config.kvcache_block_size
        if len(prompt) + sp.max_tokens > capacity:
            raise RequestCapacityError("单请求超出实际 KV 容量；请减少上下文或输出长度")
        if len(self.scheduler.waiting) + len(self.scheduler.running) >= self.config.max_pending_requests:
            raise RequestCapacityError("请求队列已满")
        seq = Sequence(list(prompt), sp)
        self.scheduler.add(seq)
        return seq

    def cancel_request(self, request):
        """须与 step 在同一线程串行调用；同时释放 partial prefill 和投机状态。"""
        seq_id = request.seq_id if isinstance(request, Sequence) else request
        scheduler = self.scheduler
        seq = next((s for s in (*scheduler.waiting, *scheduler.running) if s.seq_id == seq_id), None)
        if seq is None:
            return False
        self._on_preempt_release(seq)
        scheduler.block_manager.deallocate(seq)
        for queue in (scheduler.waiting, scheduler.running):
            try:
                queue.remove(seq)
            except ValueError:
                pass
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = "cancelled"
        return True

    def request_snapshot(self, seq, offset=0):
        """返回拥有独立存储的增量；句柄由调用方持有，引擎不积累已完成请求。"""
        if not 0 <= offset <= seq.num_completion_tokens:
            raise ValueError("增量 offset 无效")
        reason = getattr(seq, "finish_reason", "")
        if seq.is_finished and not reason:
            reason = ("stop" if not seq.ignore_eos and seq.last_token == self.config.eos else "length")
        return {"request_id": seq.seq_id,
                "token_ids": seq.token_ids[seq.num_prompt_tokens + offset:],
                "prompt_tokens": seq.num_prompt_tokens,
                "completion_tokens": seq.num_completion_tokens,
                "finished": seq.is_finished, "finish_reason": reason}
