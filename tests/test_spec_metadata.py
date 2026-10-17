"""累计 KV 长度元数据的主机回归，不捕获图、不加载模型。"""

import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec("transformers"), "需要引擎依赖")
class SpecMetadataTests(unittest.TestCase):
    def test_chain_cumulative_lengths_with_unequal_and_padding_rows(self):
        import torch
        from flashengine.engine.model_runner import ModelRunner
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(spec_num_tokens=2)
        runner._pin_M = 3
        lengths = torch.tensor([[3, 7, 0], [4, 8, 1]], dtype=torch.int64)
        cumulative = torch.full((2, 4), -1, dtype=torch.int64)
        synced = []
        runner._sblob = SimpleNamespace(
            view_i64=lambda name, shape: lengths if name == "chain_ctxlen" else cumulative,
            h2d=lambda: synced.append(True))
        runner._push_spec_meta()
        self.assertEqual(cumulative.tolist(), [[0, 3, 10, 10], [0, 4, 12, 13]])
        self.assertEqual(synced, [True])

    def test_k_one_does_not_index_nonexistent_second_draft_row(self):
        import torch
        from flashengine.engine.model_runner import ModelRunner
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(spec_num_tokens=1)
        runner._sg = {"carry_seed": torch.zeros(2, 4)}
        runner._sg_draft_chain(2, True)


if __name__ == "__main__":
    unittest.main()
