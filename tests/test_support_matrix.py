import unittest

from flashengine.support_matrix import evaluate, format_matrix, supported_spec_methods


class SupportMatrixTests(unittest.TestCase):
    def test_target_spec_method_tables(self):
        self.assertEqual(
            supported_spec_methods("qwen3"),
            frozenset({"none", "eagle3", "dflash", "dspark"}),
        )
        self.assertEqual(
            supported_spec_methods("qwen3_moe"),
            frozenset({"none", "dflash"}),
        )

    def test_qwen3_dense_all_backends_and_spec(self):
        for backend in ("cuda", "corex", "ascend310p"):
            for spec in ("none", "eagle3", "dflash", "dspark"):
                self.assertTrue(
                    evaluate("qwen3", backend, "none", spec).supported,
                    (backend, spec),
                )

    def test_moe_on_all_backends_and_dflash(self):
        for backend in ("cuda", "corex", "ascend310p"):
            self.assertTrue(
                evaluate("qwen3_moe", backend, "none", "none").supported, backend)
            self.assertTrue(
                evaluate("qwen3_moe", backend, "none", "dflash").supported, backend)
        self.assertFalse(evaluate("qwen3_moe", "ascend310p", "none", "eagle3").supported)
        self.assertFalse(evaluate("qwen3_moe", "ascend310p", "none", "dspark").supported)

    def test_w8a8_only_on_310p(self):
        self.assertTrue(evaluate("qwen3", "ascend310p", "w8a8", "none").supported)
        self.assertTrue(evaluate("qwen3", "ascend310p", "w8a8", "dflash").supported)
        self.assertTrue(evaluate("qwen3_moe", "ascend310p", "w8a8", "dflash").supported)
        self.assertFalse(evaluate("qwen3", "cuda", "w8a8", "none").supported)
        self.assertFalse(evaluate("qwen3_moe", "cuda", "w8a8", "none").supported)
        self.assertFalse(evaluate("qwen3", "corex", "w8a8", "none").supported)

    def test_quark_w8a8_on_cuda_and_corex_only(self):
        for backend in ("cuda", "corex"):
            self.assertTrue(
                evaluate("qwen3_moe", backend, "quark_w8a8", "none").supported, backend)
            self.assertTrue(
                evaluate("qwen3_moe", backend, "quark_w8a8", "dflash").supported, backend)
        self.assertFalse(evaluate("qwen3_moe", "ascend310p", "quark_w8a8", "none").supported)
        self.assertFalse(evaluate("qwen3", "ascend310p", "quark_w8a8", "none").supported)

    def test_unknown_combination_rejected(self):
        self.assertFalse(evaluate("qwen3", "cuda", "int4", "none").supported)
        self.assertFalse(evaluate("qwen3", "cuda", "none", "mtp").supported)
        self.assertFalse(evaluate("unknown", "cuda", "none", "none").supported)

    def test_format_matrix_covers_all_combinations(self):
        text = format_matrix()
        rows = [line for line in text.splitlines() if line.startswith("| qwen3")]
        # 2 targets × 3 backends × 3 quantizations × 4 spec methods
        self.assertEqual(len(rows), 2 * 3 * 3 * 4)


if __name__ == "__main__":
    unittest.main()
