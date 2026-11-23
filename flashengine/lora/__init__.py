"""多 LoRA 适配（1 基座 + N 适配器同时在线，按请求路由）。

设计要点（与 vLLM 的 punica/SGMV 思路等价，但实现更薄）：

- **权重栈**：每个被命中的 Linear 子分片持有 `A[max_loras, r, in]` 与
  `B[max_loras, out, r]`，缩放系数（alpha/r，rslora 用 alpha/sqrt(r)）在装载期
  折进 B，因此前向里只做两次小 GEMM，不需要逐 token 乘缩放。
- **路由**：`context.set_active_mask(tensor)` 在每步前设置
  `[num_tokens, max_loras*rank]` 的 0/1 掩码（主机侧算好再 copy 进设备）。
  层内 forward hook 读取该张量，不修改任何模型 forward 签名，也不影响非
  LoRA 路径；eager 与 NPUGraph 图内走同一条路径（两次定长 GEMM + 一次乘）。
- **前缀缓存**：`BlockManager.compute_hash` 把 lora_id 混进哈希，不同适配器的
  KV 永不互相命中（同一段文本在基座与适配器下隐层完全不同）。
"""

from flashengine.lora.config import LoRAConfig, LoRARequest
from flashengine.lora.context import (
    clear_active_mask,
    get_active_mask,
    set_active_mask,
)
from flashengine.lora.manager import LoRAManager

__all__ = [
    "LoRAConfig",
    "LoRARequest",
    "LoRAManager",
    "set_active_mask",
    "get_active_mask",
    "clear_active_mask",
]
