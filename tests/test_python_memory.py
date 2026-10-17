"""显存规划的纯主机边界回归，不加载模型。"""

import random
import unittest

from flashengine.engine.memory import MemoryBudgetError, plan_kv_memory, validate_memory_options


class MemoryTests(unittest.TestCase):
    def plan(self, **overrides):
        options = dict(total=10000, free=8000, resident=1000, workspace=500,
                       block_bytes=100, utilization=0.9, graph_bytes=500)
        return plan_kv_memory(**(options | overrides))

    def test_external_users_and_future_reserve(self):
        self.assertEqual(self.plan().num_blocks, 60)
        self.assertEqual(self.plan(reserve_bytes=3000).num_blocks, 40)
        self.assertEqual(self.plan(free=4000).num_blocks, 20)

    def test_engine_budget_includes_states_buffers_and_workspace(self):
        plan = self.plan(budget_bytes=4000)
        self.assertEqual(plan.num_blocks, 20)
        self.assertEqual(plan.snapshot()["kv_bytes"], 2000)
        self.assertEqual(self.plan(budget_bytes=4000, resident=2000).num_blocks, 10)

    def test_manual_blocks_cannot_bypass_budget(self):
        self.assertEqual(self.plan(requested_blocks=60).num_blocks, 60)
        with self.assertRaises(MemoryBudgetError):
            self.plan(requested_blocks=61)
        with self.assertRaises(MemoryBudgetError):
            self.plan(budget_bytes=2000, requested_blocks=1)

    def test_dummy_block_is_not_usable_capacity(self):
        with self.assertRaises(MemoryBudgetError):
            self.plan(budget_bytes=2100, minimum_blocks=2)
        self.assertEqual(self.plan(budget_bytes=2200, minimum_blocks=2).num_blocks, 2)

    def test_invalid_inputs(self):
        for utilization in (0, -1, 1.01, float("nan"), float("inf")):
            with self.subTest(utilization=utilization), self.assertRaises(ValueError):
                self.plan(utilization=utilization)
        for key in ("budget_bytes", "reserve_bytes", "graph_bytes", "workspace"):
            for value in (-1, True, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.plan(**{key: value})
        for blocks in (0, -2, True, 1.5):
            with self.assertRaises(ValueError):
                validate_memory_options(.9, 0, 0, 0, 0, blocks)

    def test_random_plans_obey_all_three_limits(self):
        rng = random.Random(10217)
        for _ in range(250):
            free = rng.randrange(2000, 10000)
            reserve = rng.randrange(0, 1000)
            budget = rng.randrange(2500, 9000)
            try:
                plan = self.plan(free=free, reserve_bytes=reserve, budget_bytes=budget)
            except MemoryBudgetError:
                continue
            additional = plan.num_blocks * 100 + 1000
            self.assertLessEqual(additional, free - reserve)
            self.assertLessEqual(additional + 1000, budget)
            self.assertLessEqual(10000 - free + additional, 9000)


if __name__ == "__main__":
    unittest.main()
