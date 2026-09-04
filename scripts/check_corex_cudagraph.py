"""CoreX CUDA Graph runtime probe（CUDA Graph 适配指导 §21）。

先于一切 Attention Graph 测试运行：验证 CoreX 的 torch.cuda.CUDAGraph
基础能力（capture / replay / pool）可用。

通过标准：打印 "CoreX CUDA Graph PASS"
"""
import torch


x = torch.zeros(
    1024,
    device="cuda",
)

y = torch.zeros_like(x)

graph = torch.cuda.CUDAGraph()


# Warmup

y.copy_(x * 2)

torch.cuda.synchronize()


# Capture

with torch.cuda.graph(graph):
    y.copy_(x * 2)


for i in range(20):

    src = torch.randn_like(x)

    x.copy_(src)

    graph.replay()

    torch.testing.assert_close(
        y,
        src * 2,
    )


print(
    "CoreX CUDA Graph PASS"
)
