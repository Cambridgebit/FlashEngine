from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True,
                 keep_last_block_unhashed: bool = False):
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        # Paged block-draft models write query K/V at the prompt anchor and
        # beyond. The final prompt block must never become a reusable prefix.
        self.keep_last_block_unhashed = keep_last_block_unhashed
        # Optional draft-state guard: a target KV hit is valid only if the
        # corresponding auxiliary draft state can be restored as well.
        self.cache_hit_validator = None
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.stats = {
            "requests": 0,
            "prefix_hits": 0,
            "cached_blocks": 0,
            "new_blocks": 0,
        }

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        # 关闭前缀缓存：不查哈希表，全部按新块分配（-1 = 空闲块不足）。
        if not self.enable_prefix_cache:
            return 0 if len(self.free_block_ids) >= seq.num_blocks else -1
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            if self.cache_hit_validator is not None and not self.cache_hit_validator(block_id, h):
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

        self.stats["requests"] += 1
        self.stats["cached_blocks"] += num_cached_blocks
        self.stats["new_blocks"] += seq.num_blocks - num_cached_blocks
        if num_cached_blocks > 0:
            self.stats["prefix_hits"] += 1

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def ensure_capacity(self, seq: Sequence, target_num_tokens: int) -> bool:
        """保证 block_table 能容纳 target_num_tokens 个 token（投机解码轮预分配）。

        预分配的容量在拒绝回滚后不回收：空槽中的旧 KV 会被后续写入覆盖，
        注意力永远只读 context_len 范围内的槽位，因此残留值无害。
        返回 False 表示空闲块不足（调用方应抢占该序列）。
        """
        while len(seq.block_table) * self.block_size < target_num_tokens:
            if not self.free_block_ids:
                return False
            seq.block_table.append(self._allocate_block())
        return True

    def hash_blocks(self, seq: Sequence):
        # 关闭前缀缓存：跳过 hash 计算与哈希表登记（含每步 decode 的开销），
        # 块保持 hash=-1，can_allocate 也不会命中任何条目。
        if not self.enable_prefix_cache:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if self.keep_last_block_unhashed:
            end = min(end, max(seq.num_blocks - 1, 0))
        if start >= end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def get_prefix_stats(self):
        stats = dict(self.stats)
        stats["hit_rate"] = (
            stats["prefix_hits"] / stats["requests"]
            if stats["requests"] else 0.0
        )
        return stats

    def reset_prefix_stats(self):
        for key in self.stats:
            self.stats[key] = 0
