import ast
import contextlib
import io
import importlib.util
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.bench_utils import load_prompts, metric_delta, percentile


ROOT = Path(__file__).parents[1]
OBSERVER_SPEC = importlib.util.spec_from_file_location(
    "flashengine_engine_observer", ROOT / "flashengine" / "engine" / "observer.py")
OBSERVER_MODULE = importlib.util.module_from_spec(OBSERVER_SPEC)
OBSERVER_SPEC.loader.exec_module(OBSERVER_MODULE)
EngineObserver = OBSERVER_MODULE.EngineObserver


class HostContractTests(unittest.TestCase):
    def test_observer_is_optional_and_overridable(self):
        EngineObserver().on_schedule([], False, 0.0)
        EngineObserver().on_commit([], [], 0.0)

        class Recorder(EngineObserver):
            def on_schedule(self, seqs, is_prefill, timestamp):
                self.event = (len(seqs), is_prefill, timestamp)

        observer = Recorder()
        observer.on_schedule([1, 2], True, 3.0)
        self.assertEqual(observer.event, (2, True, 3.0))

    def test_stdlib_benchmark_helpers(self):
        self.assertEqual(percentile([1, 3], 50), 2)
        self.assertEqual(metric_delta({"x": 2}, {"x": 5}, {"out": "x"}), {"out": 3})
        path = ROOT / "tests" / "questions.tmp.jsonl"
        try:
            path.write_text('{"turns":["a"],"category":"x"}\n'
                            '{"turns":["b"]}\n', encoding="utf-8")
            self.assertEqual(load_prompts(path, "x", 2, -1), ["a", "b"])
        finally:
            path.unlink(missing_ok=True)

    def test_spec_graph_capture_failure_discards_partial_graphs(self):
        source = (ROOT / "flashengine" / "engine" / "model_runner.py").read_text("utf-8")
        tree = ast.parse(source)
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_capture_spec_graphs")
        method.decorator_list = []
        namespace = {
            "_graph_buckets": lambda size: [1, size],
            "reset_context": lambda: None,
            "traceback": traceback,
        }
        exec(compile(ast.Module([method], type_ignores=[]), "capture", "exec"), namespace)

        class Graph:
            def replay(self):
                raise RuntimeError("deferred replay failure")

        class Backend:
            def graph_warmup_iters(self): return 0
            def new_graph(self): return Graph()
            def graph_ctx(self, graph, pool=None): return contextlib.nullcontext()
            def graph_pool(self, graph): return object()
            def synchronize(self): pass
            def empty_cache(self): pass

        runner = SimpleNamespace(
            _graph_max_bs=2, spec_graphs=None, spec_graph_pool=None,
            backend=Backend(), _sg={"v_cu_k": [0, 0, 0], "v_cu_q": [0, 1, 2]},
            _sblob=SimpleNamespace(h2d=lambda: None),
            _push_spec_meta=lambda: None,
            spec_graph_stats={"capture_success": 0, "capture_failure": 0},
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            namespace["_capture_spec_graphs"](runner, lambda: None, lambda _: None, "Spec")
        self.assertIsNone(runner.spec_graphs)
        self.assertIsNone(runner.spec_graph_pool)
        self.assertEqual(runner.spec_graph_stats["capture_failure"], 1)

    def test_measurement_reset_preserves_capture_results(self):
        source = (ROOT / "flashengine" / "engine" / "model_runner.py").read_text("utf-8")
        method = next(node for node in ast.walk(ast.parse(source))
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "reset_measurement_stats")
        namespace = {}
        exec(compile(ast.Module([method], type_ignores=[]), "reset", "exec"), namespace)
        runner = SimpleNamespace(
            graph_runner=None,
            decode_graph_stats={"capture_success": 2, "capture_failure": 1,
                                "replay": 9, "fallback": 3, "bucket_hits": {4: 9}},
            spec_graph_stats={"capture_success": 1, "capture_failure": 0,
                              "replay": 7, "eager_rounds": 2,
                              "sampling_fallbacks": 1, "bucket_hits": {2: 7}},
            prefill_cache_stats={"batches": 8},
            spec_accept_stats={"rounds": 4, "committed": 8, "accepted": 4,
                               "positions": [4, 2]},
            df_prefix_snapshot_stats={"saved": 3, "restored": 2, "evicted": 1},
            config=SimpleNamespace(spec_num_tokens=2),
        )
        namespace["reset_measurement_stats"](runner)
        self.assertEqual(runner.decode_graph_stats["capture_success"], 2)
        self.assertEqual(runner.spec_graph_stats["capture_success"], 1)
        self.assertEqual(runner.decode_graph_stats["replay"], 0)
        self.assertEqual(runner.spec_accept_stats["positions"], [0, 0])

    def test_chat_truncation_preserves_template_and_newest_tail(self):
        source = (ROOT / "scripts" / "openai_server.py").read_text("utf-8")
        tree = ast.parse(source)
        names = {"_normalize_messages", "_template_ids", "_truncate_last",
                 "_chat_prompt_ids_fit"}
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]

        class Tokenizer:
            def encode(self, text): return list(text)
            def decode(self, tokens): return "".join(tokens)
            def apply_chat_template(self, messages, **_):
                text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>"
                               for m in messages) + "<assistant>"
                return list(text)

        namespace = {"_tokenizer": Tokenizer()}
        exec(compile(ast.Module(functions, type_ignores=[]), "prompt", "exec"), namespace)
        ids, dropped = namespace["_chat_prompt_ids_fit"](
            [{"role": "user", "content": "old"},
             {"role": "assistant", "content": "reply"},
             {"role": "user", "content": "0123456789"}], None, 32)
        self.assertLessEqual(len(ids), 32)
        self.assertEqual(dropped, 2)
        rendered = "".join(ids)
        self.assertIn("6789</user>", rendered)
        self.assertTrue(rendered.endswith("<assistant>"))


if __name__ == "__main__":
    unittest.main()
