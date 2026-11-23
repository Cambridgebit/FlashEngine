"""LoRA 增量层：秩拼接权重 + 前向 hook。

每个 binding 对应「某个 Linear 模块的一段输出列」（融合模块被切成多个分片，
例如 qkv_proj 的 q/k/v、in_proj_qkvz 的 q/k/v/z）。同一个 binding 上叠放
``max_loras`` 份适配器权重，前向时按 token 路由。

**存储与计算口径（2026-11 重写）**：适配器权重按秩**拼接**存放，而不是逐槽
独立 GEMM。即

* ``AT``：``[in, max_loras*rank]``，第 s 个槽占列 ``[s*rank, (s+1)*rank)``；
* ``BT``：``[max_loras*rank, out]``，第 s 个槽占行 ``[s*rank, (s+1)*rank)``。

一次前向只需两个稠密 GEMM：

    h = x @ AT                 # [T, max_loras*rank]
    h = h * M                  # 只保留每个 token 自己槽位的 rank 列
    y = h @ BT                 # [T, out]

``M`` 是 ``[T, max_loras*rank]`` 的 0/1 掩码，由 ``LoRAManager.build_mask``
在**主机侧**按适配器下标算好（每步一次 copy_ 进设备），一次前向内所有
binding 共享。相比旧实现「逐槽位 8×2 次小 GEMM + 8 次掩码」，算子数从
~4000/回放降到 ~750/回放，且完全没有 ``torch.nonzero``（旧 eager 路径每个
binding 每槽一次 nonzero = 每次预填充近 2000 次设备同步，是 LoRA 版 TTFT
从 1.4s 涨到 4.0s 的主因）。

**掩码必须是预构建的 fp16 buffer，不能在图内用整数比较生成**（2026-11 实测）：
``idx.unsqueeze(1) == slot_of_col`` 在图内回放时结果与捕获期语义不一致，
投机接受长度会直接塌到 1（选中正确性无从自查）；换成静态 fp16 掩码后图内
只剩一次 ``mul``，捕获与回放算子完全同构。
"""

import contextlib
import os

import torch

from flashengine.lora.context import get_active_mask

_TRACE = os.environ.get("FLASHENGINE_LORA_TRACE", "") not in ("", "0")
_TRACED: list[tuple] = []


def set_trace_enabled(flag: bool) -> None:
    """图捕获期必须关掉调试打印（.tolist()/.item() 会让 capture 失败）。"""
    global _TRACE
    _TRACE = bool(flag)


@contextlib.contextmanager
def no_trace():
    """捕获/回放期临时关掉 trace。"""
    global _TRACE
    old = _TRACE
    _TRACE = False
    try:
        yield
    finally:
        _TRACE = old


class LoRABinding:
    """一个 Linear 输出分片上的适配器权重栈（秩拼接布局）。"""

    def __init__(self, module, start: int, size: int, input_size: int,
                 max_loras: int, max_rank: int, dtype: torch.dtype,
                 device: torch.device, name: str = ""):
        self.module = module
        self.start = start
        self.end = start + size
        self.out_size = size
        self.input_size = input_size
        self.max_loras = max_loras
        self.max_rank = max_rank
        self.cols = max_loras * max_rank
        self.name = name
        # 缩放已折进 BT；短 rank 适配器的尾部列/行留零。
        self.AT = torch.zeros(input_size, self.cols, dtype=dtype, device=device)
        self.BT = torch.zeros(self.cols, size, dtype=dtype, device=device)
        # 主机侧镜像：310P 上对设备张量的分片（跨步）写入会触发 AICPU 异常
        # （`copy_between_host_and_device_opapi`，507018），所以先在 CPU 拼接，
        # 再整块 contiguous 拷进设备缓冲——只在 flush() 时做一次。
        # 注意：引擎在 310P 上会 torch.set_default_device('npu')，所以这里必须
        # 显式 device="cpu"，否则镜像会落在设备上，跨步写又会踩 AICPU。
        self._AT_cpu = torch.zeros(input_size, self.cols, dtype=dtype, device="cpu")
        self._BT_cpu = torch.zeros(self.cols, size, dtype=dtype, device="cpu")
        self._dirty = False
        self.loaded = [False] * max_loras
        # 图捕获期间置 True；掩码路径图内外同构，保留标记供调用方查询。
        self.static = False

    def set_adapter(self, slot: int, A: torch.Tensor, B: torch.Tensor, scaling: float):
        """slot 为 1-based；A/B 为 fp32 CPU 张量（B 未折缩放）。"""
        if slot < 1 or slot > self.max_loras:
            raise ValueError(f"适配器槽位 {slot} 超出 max_loras={self.max_loras}")
        rank = A.shape[0]
        if rank > self.max_rank:
            raise ValueError(
                f"适配器 rank={rank} 超过 max_lora_rank={self.max_rank}（{self.name}）")
        if A.shape[1] != self.input_size or B.shape[0] != self.out_size:
            raise ValueError(
                f"LoRA 形状与模块不符 {self.name}: A{A.shape} B{B.shape} "
                f"期望 in={self.input_size} out={self.out_size}")
        lo = (slot - 1) * self.max_rank
        # 先清零本槽整块，避免上一次更短 rank 的残留列参与计算。
        self._AT_cpu[:, lo:lo + self.max_rank].zero_()
        self._BT_cpu[lo:lo + self.max_rank, :].zero_()
        # 转置放到 CPU 上做：310P 的跨步/转置拷贝（copy_between_host_and_device_opapi）
        # 会触发 AICPU 异常，设备侧只接受 contiguous 整块传输。
        a_cpu = A.detach().to("cpu", self.AT.dtype).t().contiguous()
        # BT 是 [rank, out]（列存），适配器的 B 是 [out, rank]，需要转置。
        b_cpu = (B.detach().to("cpu", torch.float32) * scaling).to(self.BT.dtype).t().contiguous()
        self._AT_cpu[:, lo:lo + rank].copy_(a_cpu)
        self._BT_cpu[lo:lo + rank, :].copy_(b_cpu)
        self.loaded[slot - 1] = True
        self._dirty = True

    def flush(self) -> None:
        """把 CPU 镜像整块拷进设备缓冲（contiguous，避开跨步写）。"""
        if not self._dirty:
            return
        self.AT.copy_(self._AT_cpu.to(self.AT.device))
        self.BT.copy_(self._BT_cpu.to(self.BT.device))
        self._dirty = False

    def delta(self, x: torch.Tensor, mask: torch.Tensor):
        """返回 [T, out_size] 的 LoRA 增量（两次 GEMM + 一次乘掩码）。"""
        return (x @ self.AT * mask) @ self.BT


def _fit_mask(mask: torch.Tensor, rows: int, dtype: torch.dtype) -> torch.Tensor:
    """把路由掩码对齐到本次前向的行数/dtype（图外兜底，图内本来就相等）。"""
    if mask.shape[0] == rows and mask.dtype == dtype:
        return mask
    if mask.shape[0] > rows:
        mask = mask[:rows]
    elif mask.shape[0] < rows:
        pad = torch.zeros(rows - mask.shape[0], mask.shape[1],
                          dtype=mask.dtype, device=mask.device)
        mask = torch.cat([mask, pad])
    return mask if mask.dtype == dtype else mask.to(dtype)


def make_forward_hook(bindings: list[LoRABinding]):
    """给一个 Linear 模块挂的 hook：把各分片的增量加回输出。"""

    def hook(module, args, output):
        mask = get_active_mask()
        if mask is None:
            return output
        x = args[0]
        # 图内外同构：只有一次 mul 的定长掩码路径。窄 rank 路径（动态切片 +
        # contiguous）已在 310P 上实测为纯负担：它的关键 K 维从 64 降到 8 并不
        # 减少耗时（小 GEMM 成本是 launch/transdata 常数项），却会引入
        # Transpose 动态算子——曾触发 aicore timeout 并挂死整机。
        mask = _fit_mask(mask, x.shape[0], x.dtype)
        for binding in bindings:
            delta = binding.delta(x, mask)
            # 调试打印必须整体包在开关里：任何 .tolist()/.item() 都是主机同步，
            # 一旦在图捕获路径上无条件执行就会让 capture 直接失败
            # （aclrtMemcpy 107030 / stream is captured）。
            if _TRACE and len(_TRACED) < 400:
                sig = (binding.name, tuple(mask[:4, :1].tolist()), x.shape[0])
                if sig not in _TRACED:
                    _TRACED.append(sig)
                    print(f"[LORA-TRACE] {binding.name} x={tuple(x.shape)} "
                          f"mask_rows={tuple(mask.shape)} "
                          f"mask_sum={mask.sum().item():.0f} "
                          f"|delta|max={delta.abs().max().item():.4g}", flush=True)
            segment = output[:, binding.start:binding.end]
            segment += delta
        return output

    return hook
