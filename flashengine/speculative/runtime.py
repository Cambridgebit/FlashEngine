

class SpecRuntime:
    block_draft = False

    def __init__(self, runner):
        self.runner = runner

    def prefill(self, seqs):
        return self.runner._run_prefill_eagle(seqs)

    def round(self, seqs, sampling_compatible):
        runner = self.runner
        if (runner.spec_graphs is not None and seqs
                and sampling_compatible
                and not any(seq.spec_cold for seq in seqs)):
            bucket = runner._bucket_for_spec(len(seqs))
            if bucket is not None:
                return runner._spec_round_graph(seqs, bucket)
        if runner.spec_graphs is not None and seqs and not sampling_compatible:
            runner.spec_graph_stats["sampling_fallbacks"] += 1
        return runner._spec_round_eager(seqs)

    def drop_state(self, seq_id):
        self.runner.spec_hidden.pop(seq_id, None)
        self.runner.spec_carry.pop(seq_id, None)

    def after_prefill(self, seqs):
        pass

    def prefix_cache_validator(self):
        return None

    def startup_details(self):
        return []

    def capture_graph(self):
        self.runner.capture_spec_cudagraph()


class BlockDraftRuntime(SpecRuntime):
    block_draft = True

    def prefill(self, seqs):
        return self.runner._run_prefill_spec_block_draft(seqs)

    def round(self, seqs, sampling_compatible):
        runner = self.runner
        if runner.spec_graphs is not None and seqs and sampling_compatible:
            bucket = runner._bucket_for_spec(len(seqs))
            if bucket is not None:
                return runner._df_round_graph(seqs, bucket)
        if runner.spec_graphs is not None and seqs and not sampling_compatible:
            runner.spec_graph_stats["sampling_fallbacks"] += 1
        return runner._df_round_eager(seqs)

    def drop_state(self, seq_id):
        super().drop_state(seq_id)
        runner = self.runner
        runner.spec_ctx_pending.pop(seq_id, None)
        runner._df_prefix_restored.discard(seq_id)
        slot = runner.draft_ctx_slots.pop(seq_id, None)
        if slot is not None:
            runner._df_ctx_free.add(slot)

    def after_prefill(self, seqs):
        if not self.runner.dflash_paged:
            self.runner.save_dflash_prefix_snapshots(seqs)

    def prefix_cache_validator(self):
        runner = self.runner
        if not runner.dflash_paged and runner.config.enable_prefix_cache:
            return runner.has_dflash_prefix_snapshot
        return None

    def startup_details(self):
        runner = self.runner
        info = runner.backend.runtime_info()
        details = [
            ("Block-draft attention path",
             "dense-KV/masked (causal or sliding window)"
             if getattr(runner.config.draft_config, "block_draft_requires_mask", False)
             else info.get("dflash_backend", "dense-KV/SDPA")),
            ("Draft prefix state", "paged KV (anchor block excluded)"
             if runner.dflash_paged else "bounded CPU snapshots (1024 MiB)"),
        ]
        if runner.config.spec_sample_from_anchor:
            details.append(("DSpark verification", "fixed-k; adaptive confidence disabled"))
        return details

    def capture_graph(self):
        self.runner.capture_df_spec_cudagraph()
