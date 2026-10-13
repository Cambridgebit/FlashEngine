import unittest

from scripts.check_cuda_attention import compare_tokens


def sample(tokens, logits):
    return [{"label": "prefix", "token_ids": [tokens],
             "logits_trace": [[{"ids": [10, 20], "logits": logits}]]}]


class AttentionComparisonTests(unittest.TestCase):
    def test_single_ulp_requires_both_candidate_scores(self):
        base = sample([10], [24.375, 24.25])
        other = sample([20], [24.25, 24.25])
        check = compare_tokens(base, other, other)[0]
        self.assertFalse(check["exact"])
        self.assertTrue(check["bf16_tie"])
        self.assertEqual(check["bf16_ulp"], 0.125)

    def test_larger_gap_is_not_exempted(self):
        base = sample([10], [24.5, 24.25])
        other = sample([20], [24.25, 24.25])
        self.assertFalse(compare_tokens(base, other, other)[0]["bf16_tie"])

    def test_shortened_output_is_not_exempted(self):
        base = sample([10, 20], [24.25, 24.25])
        other = sample([10], [24.25, 24.25])
        self.assertFalse(compare_tokens(base, other, other)[0]["bf16_tie"])

    def test_topk_tie_order_does_not_hide_a_candidate(self):
        base = sample([10], [24.25, 24.25])
        other = [{"label": "prefix", "token_ids": [[20]],
                  "logits_trace": [[{"ids": [20, 30, 10], "logits": [24.25, 24.125, 24.125]}]]}]
        self.assertTrue(compare_tokens(base, other, other)[0]["bf16_tie"])
