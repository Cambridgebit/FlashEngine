import os
import time
from collections import deque
import xxhash
import numpy as np

from flashengine.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        # 链深（0 = 会话首块）。Agent 负载里浅块是 system+tools 公共主干，
        # 深块是某一轮工具输出，被复用的概率差一个数量级——驱逐时按深度加权。
        self.depth = -1
        self.last_touch = 0.0
        # 租约到期时间（monotonic）。> now 表示 trunk 租约保护中。
        self.pin_until = 0.0
        # 同一父块下被新分支取代的旧兄弟（多半是报错/放弃的试错分支），优先回收。
        self.demoted = False
        # 子块哈希集合：父块用它检测「同一父节点下分叉」。
        self.children = set()

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.depth = -1
        self.last_touch = 0.0
        self.pin_until = 0.0
        self.demoted = False
        self.children = set()


def _dbg(msg: str) -> None:
    """前缀缓存链断点排查（FLASHENGINE_DEBUG_CACHE=1 时输出）。"""
    if os.environ.get("FLASHENGINE_DEBUG_CACHE") == "1":
        print(f"[cache] {msg}", flush=True)


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True,
                 keep_last_block_unhashed: bool = False,
                 lease_ttl: float | None = None,
                 lease_trunk_blocks: int | None = None,
                 depth_alpha: float | None = None):
        self.block_size = block_size
        # ---- 会话租约 + 拓扑深度加权驱逐（BS=1 Agent 负载默认开启）----
        # lease_ttl：主干块最近一次命中后保留多久（秒），期间即使 ref_count==0 也不回收。
        # lease_trunk_blocks：前多少个块算「会话主干」（system prompt + 工具定义 + 核心源码）。
        # depth_alpha：深度权重系数，EvictScore = idle_time × (1 + alpha·depth/max_depth)。
        # 三者都可以用环境变量覆盖；置 lease_ttl=0 即完全回到「先进先出」的老行为。
        self.lease_ttl = (float(os.environ.get("FLASHENGINE_LEASE_TTL_S", "300"))
                          if lease_ttl is None else lease_ttl)
        self.lease_trunk_blocks = (int(os.environ.get("FLASHENGINE_LEASE_TRUNK_BLOCKS", "64"))
                                   if lease_trunk_blocks is None else lease_trunk_blocks)
        self.depth_alpha = (float(os.environ.get("FLASHENGINE_EVICT_DEPTH_ALPHA", "1.0"))
                            if depth_alpha is None else depth_alpha)
        # 每次分配最多扫描多少个空闲块来挑受害者（O(n) 全扫在 16k 块上不划算）。
        self.victim_scan = 512
        self.evict_stats = {"lease_protected": 0, "lease_broken": 0,
                            "demoted": 0, "depth_weighted": 0}
        # 可选回调 (block_id, block_hash, expiry)：把同一个租约同步给别的缓存
        # （Qwen3.5 的 GDN 状态快照），否则快照先被淘汰、KV 块保住也没用。
        self.lease_listener = None
        self.enable_prefix_cache = enable_prefix_cache
        # beyond. The final prompt block must never become a reusable prefix.
        self.keep_last_block_unhashed = keep_last_block_unhashed
        self.cache_hit_validator = None
        # recurrent 只需恢复命中末块的状态；中间状态可被独立淘汰。
        self.prefix_state_validator = None
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.stats = {
            "requests": 0,
            "prefix_hits": 0,
            "prefix_hit_blocks": 0,
            "cached_blocks": 0,
            "new_blocks": 0,
        }

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1, lora_id: int = 0):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        # LoRA：同一段 token 在基座与不同适配器下的隐层完全不同，KV 不可互用。
        # 把 lora_id 混进哈希（链首块也混），不同适配器的块必然落在不同的键上。
        if lora_id:
            h.update(b"lora")
            h.update(int(lora_id).to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _scan_victim(self, now: float, max_depth: float, ignore_pins: bool):
        """在空闲块窗口内挑受害者；返回 (候选下标, 被租约保护而跳过的块数)。"""
        best_idx, best_score, skipped = -1, -1.0, 0
        limit = min(len(self.free_block_ids), self.victim_scan)
        for idx in range(limit):
            block = self.blocks[self.free_block_ids[idx]]
            if not ignore_pins and block.pin_until > now:
                skipped += 1
                continue
            if block.demoted:
                return idx, skipped
            idle = now - block.last_touch if block.last_touch else float("inf")
            depth = block.depth if block.depth >= 0 else 0
            score = idle * (1.0 + self.depth_alpha * min(depth, max_depth) / max_depth)
            if score > best_score:
                best_idx, best_score = idx, score
        return best_idx, skipped

    def _pick_victim(self) -> int:
        """在空闲块里挑驱逐受害者：先死胡同分支，再按「空闲时间 × 深度权重」。

        两遍扫描：第一遍跳过租约未到期的主干块；只有窗口内**找不到**任何未
        受保护的块（说明整池都被租约占满），才打破租约，并仍然优先牺牲链最
        深、最久未用的那块——保留浅层 system/tools 主干。
        """
        now = time.monotonic()
        max_depth = max(1.0, float(self.lease_trunk_blocks))
        idx, skipped = self._scan_victim(now, max_depth, ignore_pins=False)
        self.evict_stats["lease_protected"] += skipped
        if idx >= 0:
            return idx
        self.evict_stats["lease_broken"] += 1
        idx, _ = self._scan_victim(now, max_depth, ignore_pins=True)
        return idx if idx >= 0 else 0

    def _allocate_block(self) -> int:
        idx = self._pick_victim() if self.enable_prefix_cache else 0
        block_id = self.free_block_ids[idx]
        if idx:
            del self.free_block_ids[idx]
        else:
            self.free_block_ids.popleft()
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
        if not self.enable_prefix_cache:
            return 0 if len(self.free_block_ids) >= seq.num_blocks else -1
        if getattr(seq, "skip_prefix_cache", False):
            # 多模态请求：占位 token 相同但特征不同，复用 KV 会读到别的音频的隐层。
            return 0 if len(self.free_block_ids) >= seq.num_blocks else -1
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        candidates = []
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h, getattr(seq, "lora_id", 0))
            block_id = self.hash_to_block_id.get(h, -1)
            if i == 0:
                _dbg(
                    f"lookup seq={seq.seq_id} blk0={h & 0xffffffff:08x} "
                    f"found={block_id != -1} "
                    f"blk_hash={(self.blocks[block_id].hash & 0xffffffff) if block_id != -1 else 0:08x}"
                )
            # 哈希表条目必须仍然对应该块当前的哈希：块被重新分配/重写后
            # 旧映射可能残留（hash_blocks 只更新块上的 hash，不清理旧键），
            # 仅比对 token_ids 会把「同内容、不同前缀」的块误判为命中，
            # 读到用别的上下文算出的 KV。stale 映射一律按未命中处理。
            if (block_id == -1
                    or self.blocks[block_id].hash != h
                    or self.blocks[block_id].token_ids != token_ids):
                break
            if self.cache_hit_validator is not None and not self.cache_hit_validator(block_id, h):
                break
            num_cached_blocks += 1
            candidates.append((block_id, h))
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if self.prefix_state_validator is not None:
            while candidates and not self.prefix_state_validator(*candidates[-1]):
                block_id, _ = candidates.pop()
                num_cached_blocks -= 1
                if block_id in self.used_block_ids:
                    num_new_blocks += 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        seq.prefix_cache_hit_blocks += num_cached_blocks
        seq._prefix_cache_hit_pending = num_cached_blocks
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h, getattr(seq, "lora_id", 0))
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            # 命中即刷新活跃时间；主干块（前 lease_trunk_blocks 个）顺带续租。
            block.last_touch = time.monotonic()
            if self.lease_ttl > 0 and i < self.lease_trunk_blocks:
                block.pin_until = block.last_touch + self.lease_ttl
                if self.lease_listener is not None:
                    self.lease_listener(block_id, h, block.pin_until)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

        self.stats["requests"] += 1
        self.stats["prefix_hit_blocks"] += num_cached_blocks
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
        while len(seq.block_table) * self.block_size < target_num_tokens:
            if not self.free_block_ids:
                return False
            seq.block_table.append(self._allocate_block())
        return True

    def hash_blocks(self, seq: Sequence):
        if not self.enable_prefix_cache:
            return
        if getattr(seq, "skip_prefix_cache", False):
            # 多模态请求的 token 序列不足以确定隐层，登记后会污染后续纯文本请求。
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if self.keep_last_block_unhashed:
            end = min(end, max(seq.num_blocks - 1, 0))
        if start >= end: return
        if start > 0:
            parent = self.blocks[seq.block_table[start - 1]].hash
            # 前缀链断了（父块未登记 / 被 clear_prefix_cache 清掉）时不能
            # 把 start 号块当成「链首」登记：它的 token 内容可能与别的
            # 前缀首块相同（重复文本很常见），后续请求会命中一段用错误
            # 上下文算出的 KV。宁可少缓存，也不登记不完整的链。
            if parent == -1:
                return
            h = parent
        else:
            h = -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h, getattr(seq, "lora_id", 0))
            old = block.hash
            if old != -1 and self.hash_to_block_id.get(old) == block.block_id:
                del self.hash_to_block_id[old]
            block.update(h, token_ids)
            block.depth = i
            block.last_touch = time.monotonic()
            # 刚写入的主干块立刻起租：prefill 结束到下一次请求之间是 Agent 的
            # 工具执行空窗期，此时 ref_count 归零，没有租约就会被优先回收。
            if self.lease_ttl > 0 and i < self.lease_trunk_blocks:
                block.pin_until = block.last_touch + self.lease_ttl
                if self.lease_listener is not None:
                    self.lease_listener(block.block_id, h, block.pin_until)
            if i > 0:
                parent = self.blocks[seq.block_table[i - 1]]
                if parent.children and h not in parent.children:
                    # 同一父节点下长出第二条分支：旧兄弟大概率是被放弃的试错路径，
                    # 直接降权，下次需要内存时先回收（只标记一层，叶子仍按深度排序）。
                    for sibling_hash in list(parent.children):
                        sibling = self.hash_to_block_id.get(sibling_hash)
                        if sibling is not None:
                            self.blocks[sibling].demoted = True
                            self.evict_stats["demoted"] += 1
                parent.children.add(h)
            self.hash_to_block_id[h] = block.block_id
        if start == 0:
            _dbg(f"register seq={seq.seq_id} blocks={end - start} "
                 f"first={(self.blocks[seq.block_table[0]].hash & 0xffffffff):08x}")

    def get_evict_stats(self):
        return dict(self.evict_stats)

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
