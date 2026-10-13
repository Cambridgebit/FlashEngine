import unittest

import torch

from flashengine.layers.sampler import Sampler


class SamplerTests(unittest.TestCase):
    def test_greedy_uses_raw_argmax_in_both_paths(self):
        logits = torch.tensor([[1.0, 3.0, 3.0], [-2.0, 0.0, 4.0]])
        for fast in (True, False):
            sampler = Sampler()
            sampler.fast_top_p = fast
            output = sampler(logits, torch.zeros(2), torch.tensor([0.1, 1.0]), torch.tensor([1, 3]))
            self.assertEqual(output.tolist(), [1, 2])

    def test_top_k_is_applied_before_top_p(self):
        # 原概率 [0.4, 0.3, 0.2, 0.1]；top-k=2 后首位为 4/7 > top-p=0.55。
        logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log()
        sampler = Sampler()
        filtered, indices = sampler._filter_logits(
            logits, torch.ones(1), torch.tensor([0.55]), torch.tensor([2]))
        self.assertEqual(indices[torch.isfinite(filtered)].tolist(), [0])

    def test_nucleus_threshold_keeps_only_required_candidates(self):
        logits = torch.tensor([[0.5, 0.25, 0.125, 0.125]]).log()
        filtered, indices = Sampler()._filter_logits(
            logits, torch.ones(1), torch.tensor([0.5]), torch.tensor([4]))
        self.assertEqual(indices[torch.isfinite(filtered)].tolist(), [0])

    def test_sampling_distribution_and_constraints(self):
        count = 8192
        probabilities = torch.tensor([0.6, 0.3, 0.1])
        logits = probabilities.log().expand(count, -1)
        for fast in (True, False):
            torch.manual_seed(123)
            sampler = Sampler()
            sampler.fast_top_p = fast
            result = sampler(logits, torch.ones(count), torch.ones(count), torch.full((count,), 3))
            observed = torch.bincount(result, minlength=3).float() / count
            torch.testing.assert_close(observed, probabilities, atol=0.025, rtol=0)
            for p, k in ((0.2, 3), (1.0, 1)):
                result = sampler(logits, torch.ones(count), torch.full((count,), p), torch.full((count,), k))
                self.assertTrue(bool((result == 0).all()))

    def test_cpu_rng_fallback_is_still_available(self):
        sampler = Sampler(device_rng=False)
        result = sampler(torch.tensor([[1.0, 5.0]]), torch.zeros(1), torch.ones(1), torch.tensor([2]))
        self.assertEqual(result.item(), 1)


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA GPU")
class SamplerGraphTests(unittest.TestCase):
    def test_replay_advances_rng_and_reads_new_sampling_parameters(self):
        for fast in (True, False):
            with self.subTest(fast=fast):
                torch.cuda.manual_seed_all(123)
                sampler = Sampler()
                sampler.fast_top_p = fast
                logits = torch.zeros(512, 4, device="cuda")
                temperatures = torch.ones(512, device="cuda")
                top_ps = torch.ones(512, device="cuda")
                top_ks = torch.full((512,), 4, device="cuda", dtype=torch.int64)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        sampler(logits, temperatures, top_ps, top_ks)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = sampler(logits, temperatures, top_ps, top_ks)
                samples = []
                for _ in range(4):
                    graph.replay()
                    samples.append(output.cpu())
                self.assertTrue(all(not torch.equal(a, b) for a, b in zip(samples, samples[1:])))
                self.assertEqual(torch.cat(samples).unique().numel(), 4)

                logits.copy_(torch.tensor([0.4, 0.3, 0.2, 0.1], device="cuda").log().expand_as(logits))
                top_ps.fill_(0.1)
                graph.replay()
                self.assertTrue(bool((output == 0).all()))
                top_ps.fill_(1.0)
                top_ks.fill_(1)
                graph.replay()
                self.assertTrue(bool((output == 0).all()))
                temperatures.zero_()
                top_ks.fill_(4)
                logits[:, 3].fill_(10)
                graph.replay()
                self.assertTrue(bool((output == 3).all()))


if __name__ == "__main__":
    unittest.main()
