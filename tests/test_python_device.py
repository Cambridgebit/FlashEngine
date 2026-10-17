"""显式设置 FLASHENGINE_TEST_MODEL 后运行的真实设备适配器回归。"""

import json
import os
import time
import unittest
from unittest.mock import patch

from test_triton_backend import PB, Request, TritonError
from flashengine.triton_backend import TritonPythonModel


@unittest.skipUnless(os.environ.get("FLASHENGINE_TEST_MODEL"), "未指定真实设备模型目录")
class PythonDeviceTests(unittest.TestCase):
    def args(self, decoupled):
        options = dict(backend=os.environ.get("FLASHENGINE_TEST_BACKEND", "cuda"),
                       device_ids=[int(os.environ.get("FLASHENGINE_TEST_DEVICE", "0"))],
                       max_num_seqs=2, max_model_len=512, max_num_batched_tokens=128,
                       memory_budget_bytes=int(os.environ.get("FLASHENGINE_TEST_BUDGET_MIB", "4096")) * 2**20,
                       device_memory_reserve_bytes=512 * 2**20)
        if os.environ.get("FLASHENGINE_TEST_SPEC_MODEL"):
            options.update(spec_model=os.environ["FLASHENGINE_TEST_SPEC_MODEL"], spec_num_tokens=2)
        config = {"max_batch_size": 0, "model_transaction_policy": {"decoupled": decoupled},
                  "parameters": {"model_path": {"string_value": os.environ["FLASHENGINE_TEST_MODEL"]},
                                 "engine_config": {"string_value": json.dumps(options)}}}
        return {"model_config": json.dumps(config), "model_instance_kind": "CPU"}

    def test_budget_failure_cleans_partial_engine_and_restores_defaults(self):
        import torch
        import torch.distributed as dist
        from flashengine import LLM
        from flashengine.engine.memory import MemoryBudgetError
        config = json.loads(self.args(False)["model_config"])
        options = json.loads(config["parameters"]["engine_config"]["string_value"])
        options.update(enforce_eager=True, num_kvcache_blocks=1000000)
        old_device, old_dtype = torch.get_default_device(), torch.get_default_dtype()
        with self.assertRaises(MemoryBudgetError):
            LLM(os.environ["FLASHENGINE_TEST_MODEL"], **options)
        self.assertFalse(dist.is_initialized())
        self.assertEqual(torch.get_default_device(), old_device)
        self.assertEqual(torch.get_default_dtype(), old_dtype)

    def test_real_engine_stream_cancel_reload_and_nonstream_order(self):
        # 替身仅替代服务器传输对象；所有初始化/step/取消/close 均运行真实引擎。
        with patch.dict("sys.modules", {"triton_python_backend_utils": PB}):
            for decoupled in (True, False):
                adapter = TritonPythonModel()
                try:
                    adapter.initialize(self.args(decoupled))
                    from flashengine import LLM
                    from flashengine.engine.sequence import Sequence
                    block_size = Sequence.block_size
                    with self.assertRaises(RuntimeError):
                        LLM(os.environ["FLASHENGINE_TEST_MODEL"], kvcache_block_size=512)
                    self.assertEqual(Sequence.block_size, block_size)
                    if decoupled:
                        stream = Request(parameters={"max_tokens": 100, "ignore_eos": True})
                        short = Request(parameters={"max_tokens": 2, "ignore_eos": True})
                        adapter.execute([stream, short])
                        deadline = time.monotonic() + 10
                        while not stream.sender.responses and time.monotonic() < deadline:
                            time.sleep(.005)
                        self.assertTrue(stream.sender.responses, "未出现真实流式 token")
                        stream.sender.cancelled = True
                        self.assertTrue(stream.sender.finished.wait(10))
                        self.assertEqual(stream.sender.responses[-1][0].error.code, TritonError.CANCELLED)
                        self.assertTrue(short.sender.finished.wait(10))
                        self.assertIsNone(short.sender.responses[-1][0].error)
                        self.assertEqual(sum(flags != 0 for _, flags in short.sender.responses), 1)
                    else:
                        good = Request(parameters={"max_tokens": 2, "ignore_eos": True})
                        responses = adapter.execute([good, Request(inputs={}), good])
                        self.assertEqual(len(responses), 3)
                        self.assertEqual(responses[1].error.code, TritonError.INVALID_ARG)
                        for response in (responses[0], responses[2]):
                            self.assertIsNone(response.error)
                            self.assertEqual(len(response.outputs["TOKEN_IDS"]), 2)
                finally:
                    adapter.finalize()
                    adapter.finalize()


if __name__ == "__main__":
    unittest.main()
