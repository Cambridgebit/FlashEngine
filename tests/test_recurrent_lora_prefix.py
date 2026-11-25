"""CPU regression for LoRA-aware KV and recurrent prefix snapshots."""

import unittest
from types import SimpleNamespace

try:
    import torch
    import xxhash  # noqa: F401
    import numpy  # noqa: F401
except ImportError:
    torch = None

if torch is not None:
    from flashengine.engine.block_manager import BlockManager
    from flashengine.engine.recurrent_state import RecurrentStateRuntime
    from flashengine.engine.sequence import Sequence


@unittest.skipIf(torch is None, "requires torch, numpy and xxhash")
class RecurrentLoraPrefixTest(unittest.TestCase):
    def setUp(self):
        layers = [SimpleNamespace(), SimpleNamespace()]
        core = SimpleNamespace(
            gdn_layers=layers,
            get_gdn_state_spec=lambda: {
                "conv_shape": (2,), "conv_dtype": torch.float32,
                "ssm_shape": (2,), "ssm_dtype": torch.float32,
            },
        )
        config = SimpleNamespace(
            backend_impl=SimpleNamespace(device_name="cpu"), max_num_seqs=8,
            spec_num_tokens=0, spec_model=None, recurrent_prefix_cache_bytes=4096,
            enable_prefix_cache=True, kvcache_block_size=4,
        )
        self.runtime = RecurrentStateRuntime(SimpleNamespace(model=core), config)
        self.blocks = BlockManager(24, 4)
        self.blocks.prefix_state_validator = self.runtime.has_prefix

    def sequence(self, lora):
        seq = Sequence(list(range(9)), lora_id=lora)
        seq.block_size = 4
        return seq

    def produce(self, lora):
        seq = self.sequence(lora)
        self.assertEqual(self.blocks.can_allocate(seq), 0)
        self.blocks.allocate(seq, 0)
        for chunk in range(2):
            seq.num_cached_tokens = chunk * 4
            seq.num_scheduled_tokens = 4
            context = SimpleNamespace()
            self.runtime.bind_prefill(context, [seq])
            self.assertEqual(context.gdn_prefix_checkpoints, ((4,),))
            for layer in range(2):
                value = lora * 100 + chunk * 10 + layer
                self.runtime.capture_prefix_layer(
                    layer, 0, 4, torch.full((2,), float(value)),
                    torch.full((2,), float(value + 1000)))
            self.runtime.finish_prefill()
            self.blocks.hash_blocks(seq)
            block = self.blocks.blocks[seq.block_table[chunk]]
            self.assertTrue(self.runtime.has_prefix(block.block_id, block.hash))
        return seq

    def test_lora_isolation_and_restoration_across_chunks(self):
        producers = {}
        for lora in (0, 1, 2):
            producers[lora] = self.produce(lora)

        for chunk in range(2):
            hashes = [self.blocks.blocks[producers[lora].block_table[chunk]].hash
                      for lora in producers]
            self.assertEqual(len(set(hashes)), 3)

        for lora in producers:
            with self.subTest(lora=lora):
                seq = self.sequence(lora)
                hits = self.blocks.can_allocate(seq)
                self.assertEqual(hits, 2)
                self.blocks.allocate(seq, hits)
                self.assertEqual(seq.block_table[:2], producers[lora].block_table[:2])
                seq.num_scheduled_tokens = 1
                context = SimpleNamespace()
                self.runtime.bind_prefill(context, [seq])
                self.assertEqual(context.gdn_host_initial, (True,))
                slot = context.gdn_host_slots[0]
                expected = torch.tensor([[lora * 100 + 10, lora * 100 + 10],
                                         [lora * 100 + 11, lora * 100 + 11]],
                                        dtype=torch.float32)
                torch.testing.assert_close(self.runtime.conv_pool[:, slot], expected)
                torch.testing.assert_close(self.runtime.ssm_pool[:, slot], expected + 1000)
                self.runtime.drop(seq.seq_id)
                self.blocks.deallocate(seq)
        self.assertEqual(self.runtime.prefix_stats["restored"], 3)

    def test_foreign_snapshot_cannot_validate_or_restore(self):
        producer = self.produce(1)
        foreign = self.sequence(2)
        self.assertEqual(self.blocks.can_allocate(foreign), 0)
        foreign.block_table = list(producer.block_table)
        foreign.num_cached_tokens = 8
        with self.assertRaisesRegex(RuntimeError, "recurrent"):
            self.runtime.restore_prefix(foreign)
        self.assertNotIn(foreign.seq_id, self.runtime.slots)


if __name__ == "__main__":
    unittest.main()
