import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace


_RUNTIME_PATH = Path(__file__).parents[1] / "flashengine" / "speculative" / "runtime.py"
_SPEC = importlib.util.spec_from_file_location("flashengine_spec_runtime", _RUNTIME_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
BlockDraftRuntime = _MODULE.BlockDraftRuntime
SpecRuntime = _MODULE.SpecRuntime


class FakeRunner:
    def __init__(self):
        self.spec_graphs = {4: object()}
        self.spec_graph_bs = [1, 2, 4]
        self.spec_graph_stats = {"sampling_fallbacks": 0}
        self.spec_hidden = {7: object()}
        self.spec_carry = {7: object()}
        self.spec_ctx_pending = {7: object()}
        self._df_prefix_restored = {7}
        self.draft_ctx_slots = {7: 3}
        self._df_ctx_free = set()
        self.calls = []

    def _bucket_for_spec(self, size):
        return next((value for value in self.spec_graph_bs if value >= size), None)

    def _spec_round_graph(self, seqs, bucket):
        self.calls.append(("eagle_graph", bucket))
        return [[1]]

    def _spec_round_eager(self, seqs):
        self.calls.append(("eagle_eager", len(seqs)))
        return [[2]]

    def _df_round_graph(self, seqs, bucket):
        self.calls.append(("block_graph", bucket))
        return [[3]]

    def _df_round_eager(self, seqs):
        self.calls.append(("block_eager", len(seqs)))
        return [[4]]


class RuntimeContractTests(unittest.TestCase):
    def test_eagle_cold_batch_uses_eager(self):
        runner = FakeRunner()
        runtime = SpecRuntime(runner)
        result = runtime.round([SimpleNamespace(spec_cold=True)], True)
        self.assertEqual(result, [[2]])
        self.assertEqual(runner.calls, [("eagle_eager", 1)])

    def test_block_runtime_uses_bucket_and_sampling_fallback(self):
        runner = FakeRunner()
        runtime = BlockDraftRuntime(runner)
        seqs = [SimpleNamespace(spec_cold=True)] * 3
        self.assertEqual(runtime.round(seqs, True), [[3]])
        self.assertEqual(runner.calls[-1], ("block_graph", 4))
        self.assertEqual(runtime.round(seqs, False), [[4]])
        self.assertEqual(runner.spec_graph_stats["sampling_fallbacks"], 1)

    def test_block_drop_releases_all_per_sequence_state(self):
        runner = FakeRunner()
        BlockDraftRuntime(runner).drop_state(7)
        self.assertNotIn(7, runner.spec_hidden)
        self.assertNotIn(7, runner.spec_carry)
        self.assertNotIn(7, runner.spec_ctx_pending)
        self.assertNotIn(7, runner._df_prefix_restored)
        self.assertNotIn(7, runner.draft_ctx_slots)
        self.assertIn(3, runner._df_ctx_free)


if __name__ == "__main__":
    unittest.main()
