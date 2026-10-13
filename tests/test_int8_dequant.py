"""Quark W8A8 int8 权重加载期反量化的纯 CPU 测试（无需 GPU/NPU）。"""

import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from flashengine.utils.loader import dequantize_int8_weight, iter_safetensors_dequant


class Int8DequantTests(unittest.TestCase):

    def test_dequantize_per_channel(self):
        torch.manual_seed(0)
        q = torch.randint(-127, 128, (4, 6), dtype=torch.int8)
        scale = torch.rand(4, dtype=torch.float32) + 0.01
        zp = torch.zeros(4, dtype=torch.int8)
        out = dequantize_int8_weight(q, scale, zp, torch.float32)
        ref = q.to(torch.float32) * scale.reshape(-1, 1)
        self.assertTrue(torch.allclose(out, ref))

    def test_dequantize_subtracts_zero_point(self):
        torch.manual_seed(1)
        q = torch.randint(-127, 128, (4, 6), dtype=torch.int8)
        scale = torch.rand(4, dtype=torch.float32) + 0.01
        zp = torch.tensor([1, -1, 0, 2], dtype=torch.int8)
        out = dequantize_int8_weight(q, scale, zp, torch.bfloat16)
        ref = ((q.to(torch.float32) - zp.to(torch.float32).reshape(-1, 1))
               * scale.reshape(-1, 1)).to(torch.bfloat16)
        self.assertTrue(torch.equal(out, ref))

    def test_iterator_absorbs_scale_and_zero_point(self):
        with tempfile.TemporaryDirectory() as d:
            q = torch.randint(-127, 128, (4, 6), dtype=torch.int8)
            scale = torch.rand(4, dtype=torch.float32) + 0.01
            zp = torch.zeros(4, dtype=torch.int8)
            plain = torch.ones(6, dtype=torch.bfloat16)
            save_file({
                "layers.0.w.weight": q,
                "layers.0.w.weight_scale": scale,
                "layers.0.w.weight_zero_point": zp,
                "norm.weight": plain,
            }, os.path.join(d, "model.safetensors"))

            got = dict(iter_safetensors_dequant(d, torch.float32))
            self.assertNotIn("layers.0.w.weight_scale", got)
            self.assertNotIn("layers.0.w.weight_zero_point", got)
            self.assertTrue(torch.allclose(
                got["layers.0.w.weight"],
                dequantize_int8_weight(q, scale, zp, torch.float32)))
            # 非量化权重原样透传（不因缺少 scale 而被改写，也不做 dtype 转换）。
            self.assertTrue(torch.equal(got["norm.weight"], plain))

    def test_iterator_leaves_bf16_weights_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            w = torch.randn(3, 5, dtype=torch.bfloat16)
            save_file({"norm.weight": w}, os.path.join(d, "model.safetensors"))
            got = dict(iter_safetensors_dequant(d, torch.bfloat16))
            self.assertTrue(torch.equal(got["norm.weight"], w))


if __name__ == "__main__":
    unittest.main()
