"""注册模型的 recurrent 状态生命周期：主槽、验证行槽与接受后回滚。"""

import torch


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
        self.bytes = sum(t.numel() * t.element_size() for t in (self.conv_pool, self.ssm_pool))
        self.decode_slots = torch.full((self.capacity,), self.dummy, dtype=torch.int32, device=self.device)
        self.decode_initial = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.verify_carry = torch.full((self.capacity,), self.dummy, dtype=torch.int32, device=self.device)

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
        context.gdn_host_slots = tuple(self.get_slot(seq) for seq in seqs)
        context.gdn_host_initial = tuple(seq.num_cached_tokens > 0 for seq in seqs)
        context.gdn_state_slots = self._tensor(context.gdn_host_slots)
        context.gdn_has_initial = self._tensor(context.gdn_host_initial, torch.bool)

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

    def snapshot(self):
        return {"used_slots": len(self.slots), "capacity": self.capacity,
                "pool_slots": self.slot_count, "bytes": self.bytes}
