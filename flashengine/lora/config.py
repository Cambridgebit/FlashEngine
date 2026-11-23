"""LoRA 的配置与请求句柄。"""

from dataclasses import dataclass


@dataclass(slots=True)
class LoRAConfig:
    """多 LoRA 运行期配置。

    max_loras 是同时可在线的适配器数（也是权重栈的第 0 维），必须 >= 请求里
    实际用到的适配器数；超出时 register 会直接报错而不是静默丢请求。
    """

    enabled: bool = False
    max_loras: int = 8
    max_lora_rank: int = 16
    # ((name, path), ...)：启动即注册，slot 按顺序分配（1..max_loras）。
    modules: tuple[tuple[str, str], ...] = ()
    # 后端的 Linear 输出是否可直接原地相加（图内静态缓冲为 True）。
    inplace_add: bool = True


@dataclass(frozen=True)
class LoRARequest:
    """单请求的适配器选择。lora_name 必须在启动注册的 modules 里。"""

    lora_name: str
    lora_path: str = ""

    def __post_init__(self):
        if not isinstance(self.lora_name, str) or not self.lora_name:
            raise ValueError("LoRARequest.lora_name 必须是非空字符串")
