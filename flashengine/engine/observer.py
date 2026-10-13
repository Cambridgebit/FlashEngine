class EngineObserver:
    def on_schedule(self, seqs, is_prefill: bool, timestamp: float) -> None:
        pass

    def on_commit(self, seqs, committed, timestamp: float) -> None:
        pass
