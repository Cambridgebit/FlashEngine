"""适配器生命周期与模块挂载。

``LoRAManager`` 在基座权重装载完成之后才 attach：此时 Linear 的 weight 形状
（含 TP 切分与 310P 的 NZ 布局）都已确定，只需要按输出分片切适配器的 B 行宽。
"""

import torch

from flashengine.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from flashengine.lora.config import LoRAConfig
from flashengine.lora.layers import LoRABinding, make_forward_hook, set_trace_enabled
from flashengine.utils.pinned import PIN_MEMORY
from flashengine.lora.loader import (
    FUSED_SUBMODULES,
    SPLIT_SUBMODULES,
    load_adapter_tensors,
    read_adapter_config,
)


class LoRAManager:
    def __init__(self, model, lora_config: LoRAConfig, dtype: torch.dtype,
                 device: torch.device):
        self.model = model
        self.config = lora_config
        self.dtype = dtype
        self.device = device
        self.slots: dict[str, int] = {}
        self.bindings: dict[tuple[int, int], LoRABinding] = {}
        self.module_bindings: dict[object, list[LoRABinding]] = {}
        self.modules: dict[str, LinearBase] = {
            path: module for path, module in model.named_modules()
            if isinstance(module, LinearBase)
        }
        self._hooks = []
        self.static = False
        self.unresolved: list[str] = []
        # 路由掩码的常驻缓冲：主机侧 pinned + 设备侧，按行数复用。
        # 每步新建 torch.zeros 再 .to(device) 是页锁定外的同步拷贝（实测每步
        # 白等 167ms），改成「填常驻 pinned 缓冲 + non_blocking 拷贝」后设备侧
        # 只多一次异步 H2D。
        self._mask_host: dict[int, torch.Tensor] = {}
        self._mask_dev: dict[int, torch.Tensor] = {}
        # 上一次下发的内容键：批组成不变时无需重新拷贝。稳态解码每步的 ids 完全
        # 相同，跳过拷贝即可省掉一次主机阻塞的 H2D（实测每步 165-390ms）。
        self._mask_key = None

    # ---------------- 注册 ----------------
    def register(self, name: str, path: str) -> int:
        """把适配器装进一个槽位（1-based），返回槽号。"""
        if name in self.slots:
            return self.slots[name]
        if len(self.slots) >= self.config.max_loras:
            raise ValueError(
                f"已注册 {len(self.slots)} 个适配器，达到 max_loras={self.config.max_loras}")
        slot = len(self.slots) + 1
        cfg = read_adapter_config(path)
        if cfg["rank"] > self.config.max_lora_rank:
            raise ValueError(
                f"适配器 {name} 的 rank={cfg['rank']} 超过 max_lora_rank="
                f"{self.config.max_lora_rank}")
        tensors = load_adapter_tensors(path)
        if not tensors:
            raise ValueError(f"适配器 {name} 没有可用的 LoRA 权重: {path}")
        self._install(slot, cfg, tensors)
        self.slots[name] = slot
        return slot

    def _install(self, slot: int, cfg: dict, tensors: dict) -> None:
        scaling = cfg["scaling"]
        for engine_path, (A, B) in tensors.items():
            resolved = self._resolve(engine_path)
            if resolved is None:
                self.unresolved.append(engine_path)
                continue
            module, offset, size = resolved
            key = (id(module), offset)
            binding = self.bindings.get(key)
            if binding is None:
                binding = LoRABinding(
                    module, offset, size, A.shape[1],
                    self.config.max_loras, self.config.max_lora_rank,
                    self.dtype, self.device, name=engine_path)
                self.bindings[key] = binding
                self.module_bindings.setdefault(module, []).append(binding)
            # 适配器里的 B 只覆盖它自己那段输出；融合模块的分片在适配器侧
            # 也是从第 0 行开始（q_proj 的 B 就是 [q, r]），所以直接取前 size 行。
            binding.set_adapter(slot, A, B[:size], scaling)

    # ---------------- 名字解析 ----------------
    def _resolve(self, engine_path: str):
        """引擎模块路径 → (module, 该分片的输出起点, 宽度)。"""
        parts = engine_path.split(".")
        leaf = parts[-1]
        if leaf in SPLIT_SUBMODULES:
            fused, shard_ids = SPLIT_SUBMODULES[leaf]
            module = self.modules.get(".".join(parts[:-1] + [fused]))
            if module is None:
                return None
            sizes = shard_sizes(module)
            start = sum(sizes[: shard_ids[0]])
            width = sum(sizes[i] for i in shard_ids)
            return module, start, width
        if leaf in FUSED_SUBMODULES:
            fused, shard = FUSED_SUBMODULES[leaf]
            module = self.modules.get(".".join(parts[:-1] + [fused]))
            if module is None:
                return None
            sizes = shard_sizes(module)
            return module, sum(sizes[:shard]), sizes[shard]
        module = self.modules.get(engine_path)
        if module is None:
            return None
        return module, 0, module.weight.shape[0]

    # ---------------- 挂载 ----------------
    def attach(self) -> int:
        if self._hooks:
            return len(self._hooks)
        for module, bindings in self.module_bindings.items():
            bindings.sort(key=lambda item: item.start)
            self._hooks.append(module.register_forward_hook(make_forward_hook(bindings)))
        return len(self._hooks)

    def flush(self) -> int:
        """把所有 binding 的 CPU 镜像整块同步到设备（注册完适配器后调一次）。"""
        flushed = 0
        for binding in self.bindings.values():
            binding.flush()
            flushed += 1
        return flushed

    def set_static(self, static: bool) -> None:
        """切换图内静态路径（掩码定长 GEMM）。"""
        self.static = static
        for binding in self.bindings.values():
            binding.static = static
        if static:
            # 捕获期必须静音 trace：打印里的 .tolist()/.item() 是主机同步，
            # 会让 NPUGraph 捕获直接失败（aclrtMemcpy 107030）。
            set_trace_enabled(False)

    def detach(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    # ---------------- 查询 ----------------
    def slot_of(self, name: str) -> int:
        slot = self.slots.get(name)
        if slot is None:
            raise KeyError(
                f"未注册的 LoRA 适配器 {name!r}；已注册: {sorted(self.slots)}")
        return slot

    @property
    def cols(self) -> int:
        """路由掩码宽度 = max_loras * max_lora_rank（所有 binding 一致）。"""
        return self.config.max_loras * self.config.max_lora_rank

    def build_mask(self, ids) -> torch.Tensor:
        """主机侧构造 0/1 路由掩码 ``[len(ids), cols]``（CPU，模型 dtype）。

        ``ids[r]`` 是该行 token 的适配器下标（0 = 基座，1..n = 槽位）；掩码只
        在槽位自己的 rank 列段上为 1。整块 contiguous 拷进设备，图内只剩一次
        ``mul``——不要改成设备侧整数比较（见 layers.py 的说明）。
        """
        rank = self.config.max_lora_rank
        cols = self.cols
        dtype = self.dtype if self.dtype in (torch.float16, torch.bfloat16) else torch.float16
        mask = torch.zeros(len(ids), cols, dtype=dtype, device="cpu")
        for row, slot in enumerate(ids):
            slot = int(slot)
            if slot > 0:
                lo = (slot - 1) * rank
                mask[row, lo:lo + rank] = 1.0
        return mask

    @property
    def mask_dtype(self) -> torch.dtype:
        return self.dtype if self.dtype in (torch.float16, torch.bfloat16) else torch.float16

    def _mask_pair(self, rows: int):
        if rows not in self._mask_dev:
            dtype = self.mask_dtype
            try:
                host = torch.zeros(rows, self.cols, dtype=dtype, device="cpu",
                                   pin_memory=PIN_MEMORY)
            except RuntimeError:
                # 锁页内存不可用时退回普通主机缓冲（纯性能损失，不影响正确性）。
                host = torch.zeros(rows, self.cols, dtype=dtype, device="cpu")
            self._mask_host[rows] = host
            self._mask_dev[rows] = torch.zeros(rows, self.cols, dtype=dtype,
                                               device=self.device)
        return self._mask_host[rows], self._mask_dev[rows]

    def mask_into(self, host: torch.Tensor, ids) -> None:
        """把 ``ids`` 的 one-hot 掩码写进既有的主机缓冲（行数取缓冲大小）。"""
        host.zero_()
        rank = self.config.max_lora_rank
        rows = min(len(ids), host.shape[0])
        for row in range(rows):
            slot = int(ids[row])
            if slot > 0:
                lo = (slot - 1) * rank
                host[row, lo:lo + rank] = 1.0

    def fill_mask(self, ids) -> torch.Tensor:
        """写掩码缓冲并下发，返回可直接绑到前向上下文的设备缓冲。

        同一批组成连续多步复用时直接返回既有缓冲：稳态解码的 ids 每步相同，
        重复拷贝只会白白阻塞主机（310P 上页锁定关闭时这一步是同步拷贝）。
        """
        key = (len(ids), tuple(ids))
        dev = self._mask_dev.get(len(ids))
        if dev is not None and self._mask_key == key:
            return dev
        host, dev = self._mask_pair(len(ids))
        self.mask_into(host, ids)
        dev.copy_(host, non_blocking=True)
        self._mask_key = key
        return dev

    def host_mask(self, rows: int) -> torch.Tensor:
        """取 ``rows`` 行对应的常驻 pinned 主机缓冲（分桶静态掩码用）。"""
        return self._mask_pair(rows)[0]

    def stats(self) -> dict:
        return {
            "max_loras": self.config.max_loras,
            "max_lora_rank": self.config.max_lora_rank,
            "registered": dict(self.slots),
            "bindings": len(self.bindings),
            "modules": len(self.module_bindings),
            "unresolved": list(self.unresolved),
            "static": self.static,
            "stack_bytes": sum(
                b.AT.numel() * b.AT.element_size() + b.BT.numel() * b.BT.element_size()
                for b in self.bindings.values()),
        }


def shard_sizes(module: LinearBase) -> list[int]:
    """融合模块各分片的输出宽度（已按 TP 切分）。"""
    if isinstance(module, MergedColumnParallelLinear):
        return [size // module.tp_size for size in module.output_sizes]
    if isinstance(module, QKVParallelLinear):
        q = module.num_heads * module.head_size
        kv = module.num_kv_heads * module.head_size
        return [q, kv, kv]
    return [module.weight.shape[0]]
