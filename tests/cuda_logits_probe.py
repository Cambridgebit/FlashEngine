"""实机回归的 logits 观察器，不修改引擎执行方法。"""

from flashengine.engine.observer import EngineObserver


class LogitsProbe(EngineObserver):
    def __init__(self, engine):
        self.seqs = []
        self.is_prefill = False
        self.captured = True
        self.rows = {}
        self.graph_runner = engine.model_runner.graph_runner
        self.handle = engine.model_runner.model.lm_head.register_forward_hook(self._capture)
        engine.observer = self

    def on_schedule(self, seqs, is_prefill, timestamp):
        self._flush_graph()
        self.seqs = seqs
        self.is_prefill = is_prefill
        self.captured = False

    def _capture(self, module, inputs, logits):
        self.captured = True
        self._record(logits)

    def _record(self, logits):
        values, ids = logits.topk(8, dim=-1)
        values, ids = values.float().cpu().tolist(), ids.cpu().tolist()
        for row, seq in enumerate(self.seqs):
            if self.is_prefill and seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens:
                continue
            self.rows.setdefault(seq.seq_id, []).append({"ids": ids[row], "logits": values[row]})

    def _flush_graph(self):
        if self.seqs and not self.captured and self.graph_runner is not None:
            bucket = next(size for size in sorted(self.graph_runner.entries) if size >= len(self.seqs))
            self._record(self.graph_runner.entries[bucket].logits[:len(self.seqs)])
            self.captured = True

    def drain(self):
        self._flush_graph()
        rows, self.rows = self.rows, {}
        return [rows[key] for key in sorted(rows)]

    def close(self):
        self.handle.remove()
