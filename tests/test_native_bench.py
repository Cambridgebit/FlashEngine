"""基准统计和严格对齐的主机测试，不导入设备框架。"""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.native_bench_common import compare_rows, summary, validate_workload


class NativeBenchTests(unittest.TestCase):
    def test_worker_parse_failure_preserves_process_output(self):
        from scripts.run_native_benchmark import run_job
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad.json"
            with patch("scripts.run_native_benchmark.gpu_sample", return_value={}), patch("scripts.run_native_benchmark.subprocess.run", return_value=subprocess.CompletedProcess(["native"], 0, "broken-json", "diagnostic")):
                result, record = run_job(["native"], output, True, 1)
            self.assertIsNone(result)
            self.assertEqual(record["stdout"], "broken-json")
            self.assertEqual(record["stderr"], "diagnostic")
            self.assertIn("error", json.loads(output.with_suffix(".process.json").read_text()))
            timeout_output = Path(directory) / "timeout.json"
            timeout = subprocess.TimeoutExpired(["native"], 1, output=b"partial-output", stderr=b"failure-detail")
            with patch("scripts.run_native_benchmark.gpu_sample", return_value={}), patch("scripts.run_native_benchmark.subprocess.run", side_effect=timeout):
                result, record = run_job(["native"], timeout_output, True, 1)
            self.assertIsNone(result)
            self.assertTrue(record["timed_out"])
            self.assertEqual(record["stdout"], "partial-output")
            self.assertEqual(record["stderr"], "failure-detail")

    def test_completion_chunk_is_not_chat_delta(self):
        from scripts.openai_server import _chunk
        choice = _chunk("r", "m", "completion", "hello")["choices"][0]
        self.assertEqual(choice["text"], "hello")
        self.assertNotIn("delta", choice)
        final = _chunk("r", "m", "completion", "", "length")["choices"][0]
        self.assertEqual(final["text"], "")
        self.assertEqual(final["finish_reason"], "length")
        self.assertEqual(_chunk("r", "m", "chat", "hello")["choices"][0]["delta"], {"content": "hello"})
        self.assertEqual(_chunk("r", "m", "chat", "", role=True)["choices"][0]["delta"], {"role": "assistant"})

    def test_http_missing_latency_and_resolved_config(self):
        from scripts.bench_frozen_http import check_server_config, latency_percentile
        self.assertIsNone(latency_percentile([], 50))
        self.assertEqual(latency_percentile([2, 4], 50), 3)
        info = dict(dtype="torch.float16", num_kvcache_blocks=32, enable_prefix_cache=False, tensor_parallel_size=1)
        cfg = dict(dtype="float16", num_kvcache_blocks=32)
        check_server_config(info, cfg)
        with self.assertRaisesRegex(ValueError, "不一致"):
            check_server_config(info, dict(cfg, num_kvcache_blocks=16))

    def test_graph_gate_excludes_warmup(self):
        from scripts.audit_native_benchmark import graph_progress
        result = dict(implementation="cpp", runtime_before=dict(decode_graph_replays=10), runtime_after=dict(decode_graph_replays=10, graph_buckets=[1, 2]))
        self.assertEqual(graph_progress(result)["replays"], 0)
        result["runtime_after"]["decode_graph_replays"] = 15
        self.assertEqual(graph_progress(result)["replays"], 5)

    def test_http_gate_rejects_missing_ttft_and_warmup_only_graph(self):
        from scripts.audit_native_http_benchmark import audit_level
        row = dict(output_tokens=2, ttft_seconds=.01, wall_seconds=.02, text="hi")
        before = dict(metrics=dict(flashengine_decode_graph_replays_total=10))
        after = dict(metrics=dict(flashengine_decode_graph_replays_total=12, flashengine_kv_blocks_used=0))
        level = dict(concurrency=1, rows=[row], server_before=before, server_after=after, server_counters_match=True, output_tokens=2, errors=[], peak={})
        self.assertTrue(audit_level(level, True)["passed"])
        before["metrics"]["flashengine_decode_graph_replays_total"] = 12
        self.assertFalse(audit_level(level, True)["passed"])
        row["ttft_seconds"] = None
        self.assertFalse(audit_level(level, False)["latency_valid"])
    def test_compare_allows_reordering_not_missing(self):
        rows = [dict(input_id=str(i), repeat=0, token_ids=[1, 2], finish_reason="length", rounds=[]) for i in range(3)]
        self.assertTrue(compare_rows(rows, list(reversed(rows)), True)["passed"])
        self.assertFalse(compare_rows(rows, rows[:2])["passed"])
        with self.assertRaises(ValueError):
            compare_rows(rows, rows + rows[:1])

    def test_trace_is_strict_even_for_uncommitted_verify(self):
        row = dict(input_id="0", repeat=0, token_ids=[1, 2], finish_reason="length", rounds=[dict(verify_ids=[2, 4])])
        changed = copy.deepcopy(row)
        changed["rounds"][0]["verify_ids"][1] = 5
        self.assertTrue(compare_rows([row], [changed])["passed"])
        self.assertFalse(compare_rows([row], [changed], True)["passed"])

    def test_summary_uses_tokens_not_sse_chunks(self):
        rows = [dict(token_ids=[1, 2, 3], ttft_seconds=.01, e2e_seconds=.05, group_wall_seconds=.1)]
        result = summary(rows, .2)
        self.assertEqual(result["output_tps"], 15)
        self.assertEqual(result["ttft_p50_ms"], 10)
        self.assertEqual(result["tpot_p50_ms"], 20)

    def test_input_rejects_overflow_and_bool_tokens(self):
        cfg = dict(dtype="float16", max_model_len=8, max_num_seqs=1, max_num_batched_tokens=8, max_num_partial_prefills=1, kvcache_block_size=256, num_kvcache_blocks=1)
        workload = dict(schema_version=1, config=cfg, vocab_size=100, requests=[dict(id="0", prompt_token_ids=[1], max_tokens=7, ignore_eos=True)])
        validate_workload(workload)
        for change in (dict(prompt_token_ids=[True]), dict(max_tokens=8), dict(ignore_eos=1)):
            invalid = copy.deepcopy(workload)
            invalid["requests"][0].update(change)
            with self.assertRaises(ValueError):
                validate_workload(invalid)


if __name__ == "__main__":
    unittest.main()
