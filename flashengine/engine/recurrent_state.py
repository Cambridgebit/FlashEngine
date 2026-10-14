"""注册模型的 recurrent 状态生命周期：主槽、验证行槽与接受后回滚。"""

import torch
from collections import OrderedDict

from flashengine.engine.block_manager import BlockManager


class RecurrentStateRuntime:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.backend_impl.device_name
        self.backend = config.backend_impl
        self.device_commit = getattr(self.backend, "supports_device_state_commit", False)
        self.capacity = config.max_num_seqs
        self.rows = config.spec_num_tokens + 1 if config.spec_model else 0
        self.slot_count = self.capacity * (1 + self.rows) + 1
        self.dummy = self.slot_count - 1
        self.slots = {}
        self.free = set(range(self.capacity))
        spec = model.model.get_gdn_state_spec()
        self.layers = model.model.gdn_layers
        self.conv_pool = torch.zeros((len(self.layers), self.slot_count, *spec["conv_shape"]),
                                     dtype=spec["conv_dtype"], device=self.device)
        self.ssm_pool = torch.zeros((len(self.layers), self.slot_count, *spec["ssm_shape"]),
                                    dtype=spec["ssm_dtype"], device=self.device)
        for index, layer in enumerate(self.layers):
            layer.conv_cache = self.conv_pool[index]
            layer.ssm_cache = self.ssm_pool[index]
            layer.prefix_layer_index = index
        self.bytes = sum(t.numel() * t.element_size() for t in (self.conv_pool, self.ssm_pool))
        self.decode_slots = torch.full((self.capacity,), self.dummy, dtype=torch.int32, device=self.device)
        self.decode_initial = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.verify_carry = torch.full((self.capacity,), self.dummy, dtype=torch.int32, device=self.device)
        self.prefix_limit = config.recurrent_prefix_cache_bytes
        self.prefix_bytes = 0
        self.prefix_snapshots = OrderedDict()
        self.prefix_pending = {}
        self.prefix_stats = {"saved": 0, "restored": 0, "evicted": 0}
        self.async_prefix = getattr(self.backend, "supports_async_prefix_snapshots", False)
        self.prefix_entry_bytes = self.bytes // self.slot_count
        if config.enable_prefix_cache and self.prefix_entry_bytes > self.prefix_limit:
            print("[FlashEngine] Recurrent 前缀预算不足以保存一个状态，关闭 prefix cache。", flush=True)
            config.enable_prefix_cache = False

    def get_slot(self, seq):
        slot = self.slots.get(seq.seq_id)
        if slot is None:
            if not self.free:
                raise RuntimeError("Recurrent 主槽耗尽，活跃请求超过 max_num_seqs")
            slot = self.free.pop()
            self.slots[seq.seq_id] = slot
        return slot

    def _tensor(self, values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device="cpu").to(self.device)

    def bind_prefill(self, context, seqs):
        # 新请求只恢复命中末块；分块 prefill 的存活主槽已经是正确初态。
        for seq in seqs:
            if seq.seq_id not in self.slots and seq.num_cached_tokens:
                self.restore_prefix(seq)
        context.gdn_host_slots = tuple(self.get_slot(seq) for seq in seqs)
        context.gdn_host_initial = tuple(seq.num_cached_tokens > 0 for seq in seqs)
        context.gdn_state_slots = self._tensor(context.gdn_host_slots)
        context.gdn_has_initial = self._tensor(context.gdn_host_initial, torch.bool)
        self.prefix_pending = {}
        if not self.config.enable_prefix_cache:
            return
        candidates = []
        block_size = self.config.kvcache_block_size
        for row, seq in enumerate(seqs):
            if not seq.block_table:
                continue
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            block_hash = -1
            for index in range(end // block_size):
                block_hash = BlockManager.compute_hash(seq.block(index), block_hash)
                position = (index + 1) * block_size
                block_id = seq.block_table[index]
                if position > start and not self.has_prefix(block_id, block_hash):
                    candidates.append((row, position - start, block_id, block_hash))
        # 限制本轮暂存峰值；优先保留较长前缀的检查点。
        count = self.prefix_limit // self.prefix_entry_bytes
        for row, offset, block_id, block_hash in candidates[-count:]:
            conv = torch.empty((len(self.layers), *self.conv_pool.shape[2:]),
                               dtype=self.conv_pool.dtype, device="cpu", pin_memory=self.async_prefix)
            ssm = torch.empty((len(self.layers), *self.ssm_pool.shape[2:]),
                              dtype=self.ssm_pool.dtype, device="cpu", pin_memory=self.async_prefix)
            self.prefix_pending[row, offset] = (block_id, block_hash, conv, ssm, set())
        context.gdn_prefix_checkpoints = tuple(
            tuple(offset for r, offset in self.prefix_pending if r == row)
            for row in range(len(seqs)))
        context.gdn_prefix_capture = self.capture_prefix_layer

    def has_prefix(self, block_id, block_hash):
        saved = self.prefix_snapshots.get(block_id)
        return saved is not None and saved[0] == block_hash

    def restore_prefix(self, seq):
        block_hash = -1
        count = seq.num_cached_tokens // self.config.kvcache_block_size
        for index in range(count):
            block_hash = BlockManager.compute_hash(seq.block(index), block_hash)
        block_id = seq.block_table[count - 1]
        if not self.has_prefix(block_id, block_hash):
            raise RuntimeError("前缀已获准复用，但 recurrent 末块状态缺失")
        _, conv, ssm = self.prefix_snapshots[block_id]
        slot = self.get_slot(seq)
        self.conv_pool[:, slot].copy_(conv, non_blocking=self.async_prefix)
        self.ssm_pool[:, slot].copy_(ssm, non_blocking=self.async_prefix)
        self.prefix_snapshots.move_to_end(block_id)
        self.prefix_stats["restored"] += 1

    def capture_prefix_layer(self, layer, row, offset, conv, ssm):
        _, _, conv_out, ssm_out, ready = self.prefix_pending[row, offset]
        conv_out[layer].copy_(conv, non_blocking=self.async_prefix)
        ssm_out[layer].copy_(ssm, non_blocking=self.async_prefix)
        ready.add(layer)

    def finish_prefill(self):
        # 全部层成功计算后才发布；异常前向留下的半成品绝不参与命中。
        if self.prefix_pending and self.async_prefix:
            # CUDA 各层 D2H 排在同一流，只在发布前统一等待一次。
            self.backend.synchronize()
        for block_id, block_hash, conv, ssm, ready in self.prefix_pending.values():
            if len(ready) != len(self.layers):
                continue
            old = self.prefix_snapshots.pop(block_id, None)
            if old is not None:
                self.prefix_bytes -= self.prefix_entry_bytes
            while self.prefix_bytes + self.prefix_entry_bytes > self.prefix_limit:
                self.prefix_snapshots.popitem(last=False)
                self.prefix_bytes -= self.prefix_entry_bytes
                self.prefix_stats["evicted"] += 1
            self.prefix_snapshots[block_id] = (block_hash, conv, ssm)
            self.prefix_bytes += self.prefix_entry_bytes
            self.prefix_stats["saved"] += 1
        self.prefix_pending = {}

    def prepare_decode(self, seqs):
        padding = self.capacity - len(seqs)
        self.decode_slots.copy_(self._tensor([self.get_slot(seq) for seq in seqs] + [self.dummy] * padding))
        self.decode_initial.copy_(self._tensor([True] * len(seqs) + [False] * padding, torch.bool))

    def prepare_capture(self):
        self.decode_slots.fill_(self.dummy)
        self.decode_initial.zero_()
        self.verify_carry.fill_(self.dummy)

    def bind_decode(self, context, batch_size):
        context.gdn_state_slots = self.decode_slots[:batch_size]
        context.gdn_has_initial = self.decode_initial[:batch_size]

    def prepare_verify(self, seqs):
        padding = self.capacity - len(seqs)
        self.verify_carry.copy_(self._tensor([self.get_slot(seq) for seq in seqs] + [self.dummy] * padding))

    def bind_verify(self, context, batch_size):
        context.gdn_state_mode = "row"
        context.gdn_carry_slots = self.verify_carry[:batch_size]
        context.gdn_row_bases = tuple(self.capacity + row * self.rows for row in range(batch_size))
        context.gdn_row_seg = self.rows

    def commit_verify(self, seqs, counts):
        # 行 count-1 已消费 anchor 和接受的提案；bonus 留给下一轮消费。
        source = self._tensor([self.capacity + row * self.rows + count - 1
                               for row, count in enumerate(counts)], torch.int64)
        target = self._tensor([self.get_slot(seq) for seq in seqs], torch.int64)
        for cache in (self.conv_pool, self.ssm_pool):
            cache.index_copy_(1, target, cache.index_select(1, source))

    def commit_verify_device(self, commit_buf, batch_size):
        """静态图尾部状态回写；接受数留在设备，padding 不写 dummy 主槽。"""
        if self.device_commit:
            for cache in (self.conv_pool, self.ssm_pool):
                self.backend.gdn_commit(cache, commit_buf[:batch_size],
                                        self.verify_carry[:batch_size], self.rows, self.capacity)

    def drop(self, seq_id):
        slot = self.slots.pop(seq_id, None)
        if slot is not None:
            self.free.add(slot)

    def clear(self):
        self.slots.clear()
        self.free = set(range(self.capacity))
        self.prepare_capture()
        self.prefix_pending = {}

    def snapshot(self):
        return {"used_slots": len(self.slots), "capacity": self.capacity,
                "pool_slots": self.slot_count, "bytes": self.bytes,
                "prefix_cache": {**self.prefix_stats, "bytes": self.prefix_bytes,
                                 "limit_bytes": self.prefix_limit,
                                 "entries": len(self.prefix_snapshots)}}
