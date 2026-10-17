"""共享 RoPE 缓存的寿命与设备/类型隔离回归。"""

import gc
import unittest
from unittest.mock import patch
import weakref

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "需要 PyTorch；不需要 GPU")
class RopeLifetimeTests(unittest.TestCase):
    def test_rope_reused_while_alive_and_released_with_model(self):
        from flashengine.layers.rotary_embedding import get_rope, get_partial_rope
        with patch("flashengine.layers.rotary_embedding.get_backend", return_value=object()):
            for factory, dim in ((get_rope, 8), (get_partial_rope, 4)):
                first = factory(8, dim, 16, 10000)
                second = factory(8, dim, 16, 10000)
                self.assertIs(first, second)
                reference = weakref.ref(first.cos_sin_cache)
                del first, second
                gc.collect()
                self.assertIsNone(reference())

    def test_default_device_and_dtype_are_part_of_cache_key(self):
        from flashengine.layers.rotary_embedding import get_rope
        dtype = torch.get_default_dtype()
        with patch("flashengine.layers.rotary_embedding.get_backend", return_value=object()):
            first = get_rope(8, 8, 16, 10000)
            try:
                torch.set_default_dtype(torch.float64 if dtype != torch.float64 else torch.float32)
                other_dtype = get_rope(8, 8, 16, 10000)
                self.assertIsNot(first, other_dtype)
            finally:
                torch.set_default_dtype(dtype)
            with torch.device("meta"):
                other_device = get_rope(8, 8, 16, 10000)
                self.assertEqual(other_device.cos_sin_cache.device.type, "meta")
            self.assertEqual(first.cos_sin_cache.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
