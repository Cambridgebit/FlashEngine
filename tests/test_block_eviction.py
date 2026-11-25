"""CPU regression for lease-aware / depth-weighted block eviction.

覆盖 2026-11-24 引入的三条驱逐规则（BS=1 Agent 负载）：

1. 会话主干块在租约期内不被回收，即使它是最久未命中的块；
2. 只有整池都被租约占满（找不到未受保护的空闲块）时才打破租约，分配不能失败；
3. 同一父节点下的旧兄弟分支（多半是报错重试的试错路径）最先被回收；
4. 同空闲时长下，链更深的块先被回收（深度权重）。

这些用例只依赖 xxhash / numpy，不需要 torch / NPU。
"""

import time
import unittest

try:
    import numpy  # noqa: F401
    import xxhash  # noqa: F401
except ImportError:  # pragma: no cover
    numpy = None

if numpy is not None:
    from flashengine.engine.block_manager import BlockManager
    from flashengine.engine.sequence import Sequence


@unittest.skipIf(numpy is None, "requires numpy and xxhash")
class BlockEvictionTest(unittest.TestCase):
    def make_manager(self, num_blocks=16, **kwargs):
        # 引擎在 llm_engine.py 里把块大小写到 Sequence 的类属性上，这里照做。
        Sequence.block_size = 4
        return BlockManager(num_blocks, 4, **kwargs)

    def age(self, bm, offsets):
        """给空闲块设置「上次访问时间」，offsets 越大越久远。"""
        now = time.monotonic()
        for block_id, offset in offsets.items():
            bm.blocks[block_id].last_touch = now - offset

    def test_lease_protects_trunk_block(self):
        bm = self.make_manager()
        now = time.monotonic()
        # 0 号块最久未命中，但持有租约；其余块刚被访问过。
        self.age(bm, {i: 60 for i in range(16)})
        bm.blocks[0].pin_until = now + 300
        victim = bm._pick_victim()
        self.assertNotEqual(bm.free_block_ids[victim], 0)
        self.assertGreaterEqual(bm.evict_stats["lease_protected"], 1)
        self.assertEqual(bm.evict_stats["lease_broken"], 0)

    def test_lease_broken_when_all_pinned(self):
        bm = self.make_manager(num_blocks=8)
        now = time.monotonic()
        bm.free_block_ids = type(bm.free_block_ids)([3])
        bm.blocks[3].pin_until = now + 300
        block_id = bm._allocate_block()
        self.assertEqual(block_id, 3)
        self.assertGreaterEqual(bm.evict_stats["lease_broken"], 1)

    def test_demoted_sibling_evicted_first(self):
        bm = self.make_manager()
        self.age(bm, {i: 60 for i in range(16)})
        # 7 号刚被访问，但它是被放弃的兄弟分支 → 应当优先回收。
        bm.blocks[7].last_touch = time.monotonic()
        bm.blocks[7].demoted = True
        victim = bm._pick_victim()
        self.assertEqual(bm.free_block_ids[victim], 7)

    def test_depth_weighting_prefers_deep_blocks(self):
        bm = self.make_manager()
        same_age = time.monotonic() - 10
        for i in range(16):
            bm.blocks[i].depth = 0
            bm.blocks[i].last_touch = same_age
        bm.blocks[9].depth = 60  # 深块：同样的空闲时长，得分更高
        victim = bm._pick_victim()
        self.assertEqual(bm.free_block_ids[victim], 9)

    def test_fifo_when_lease_disabled(self):
        bm = self.make_manager(lease_ttl=0)
        self.assertEqual(bm.lease_ttl, 0)
        self.assertEqual(bm._allocate_block(), 0)  # 老行为：从左取

    def test_trunk_lease_refreshed_on_hit(self):
        Sequence.block_size = 4
        bm = BlockManager(16, 4, lease_ttl=300, lease_trunk_blocks=2)
        seq = Sequence(list(range(12)))
        bm.allocate(seq, 0)
        seq.num_scheduled_tokens = seq.num_tokens
        bm.hash_blocks(seq)
        for i in range(2):
            self.assertGreater(bm.blocks[seq.block_table[i]].pin_until, time.monotonic())
        # 第三个块在主干之外，不续租。
        self.assertEqual(bm.blocks[seq.block_table[2]].pin_until, 0.0)


if __name__ == "__main__":
    unittest.main()
