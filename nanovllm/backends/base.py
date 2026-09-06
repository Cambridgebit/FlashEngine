from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
import torch.nn.functional as F

# RMSNorm 等编译单元按 (rank, batch) 累积编译版本，默认 recompile_limit=8 很快用尽，
# 之后的新 shape 回退 eager——浮点运算顺序与已编译 kernel 不同，会造成跨引擎 token
# 级结果分歧。调高规避（原 layernorm.py 顶部常量随融合点方法迁移至此）。
torch._dynamo.config.recompile_limit = 256


def _apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class BackendBase(ABC):

    name = "base"
    # TP 初始化的分布式后端名（CoreX 走厂商 torch 的 NCCL 兼容路径，V1.3 实证）。
    dist_name = "nccl"
    # 是否尝试 spec 单图整轮 CUDA Graph 捕获；失败由 capture 的 try/except 自动
    # 降级 eager spec 轮（多硬件方法论 = 直接尝试 + 优雅降级，见 V1.3）。
    # spec 图默认门 = supports_spec_graph 且 graph_sample_default（spec 图无条件录
    # Gumbel 采样，RNG 图安全性未验证的后端不默认尝试）；NANOVLLM_SPEC_GRAPH
    # 显式设置时优先后端默认。
    # 降级真值表（三后端 × 两图种；成败均由捕获现场裁决，此表描述「是否尝试」）：
    #   CUDA       : decode 图 ✅实证 / spec 整轮图 ✅实证（graph_sample 默认 True）
    #   CoreX      : decode 图 ✅实证（V1.3：厂商 torch 的 CUDA Graph 兼容层可捕获）
    #                / spec 图 尝试+降级
    #   Ascend310P : decode 图 尝试+降级（NPUGraph）/ spec 全路径 ❌ 引擎层拒绝
    #                （supports_spec=False：verify 恒传 block_tables → attention 层
    #                走整池 4D 分支，TND 不消费——paged-TND 适配后再开；
    #                graph_sample 默认 False——NPU RNG 图安全性未验证）
    #   env 覆盖：NANOVLLM_SPEC_GRAPH=0 → spec 图一律不尝试（全后端）；
    #   NANOVLLM_GRAPH_SAMPLE=0/1 → 覆盖 graph_sample_default；TP>1 守卫不受 env
    #   影响（lm_head gather 不可捕获，恒禁图内采样）。
    supports_spec_graph = False
    # spec 全路径（eager 轮 + 图轮）可用性；False = 引擎在出现 spec_model 配置时
    # 直接干净拒绝（守卫见 model_runner，能力位裁决、不做 name 字符串判断）。
    # 310P 因 TND 不消费 paged KV 而为 False。
    supports_spec = True
    # 设备语义串（"cuda"/"npu"）：set_default_device 与张量 .to(device) 的目标。
    device_name = "cuda"
    # 后端强制 dtype（None = 跟随 hf_config；310P 因 ATB 算子不支持 bf16 而强制 fp16）。
    default_dtype = None
    # decode 图内采样（lm_head+Sampler 入图）的后端默认值；env NANOVLLM_GRAPH_SAMPLE
    # 显式设置时优先后端默认。Ascend 起步保守关闭（NPU RNG 图安全性未验证），
    # 关闭时 logits/采样走图外 gv["outputs"] 分支。
    graph_sample_default = True

    @abstractmethod
    def store_kvcache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        raise NotImplementedError

    @abstractmethod
    def prefill_attention(
        self,
        q,
        k,
        v,
        context,
        scale,
    ):
        raise NotImplementedError

    @abstractmethod
    def decode_attention(
        self,
        q,
        k_cache,
        v_cache,
        context,
        scale,
    ):
        raise NotImplementedError

    # ---- 设备代理（Tier-2 收口）：厂商差异调用统一走 backend；默认 = CUDA 语义，
    # CoreX 靠厂商 torch 兼容层同享，Ascend 覆盖为 torch.npu.* ----
    def set_device(self, rank):
        torch.cuda.set_device(rank)

    def synchronize(self):
        torch.cuda.synchronize()

    def empty_cache(self):
        torch.cuda.empty_cache()

    def mem_get_info(self):
        return torch.cuda.mem_get_info()

    def memory_stats(self):
        return torch.cuda.memory_stats()

    def reset_peak_memory_stats(self):
        torch.cuda.reset_peak_memory_stats()

    # ---- 图 API 工厂：CUDA Graph 的厂商镜像收口（Ascend=NPUGraph，见 ascend.py）。
    # 返回「实例」而非类——调用方语义是"给我一个新图"。旧形态（返回类）在调用点
    # 需要多写一层 ()，漏写即把类对象传进捕获、以 unbound 方式调 capture_begin，
    # 已被 T3 位等门禁实证抓出，故收口为实例语义。
    def new_graph(self):
        return torch.cuda.CUDAGraph()

    @contextmanager
    def graph_ctx(self, graph, pool=None):
        # 纯透传：厂商 torch 若不支持 pool kwarg，TypeError 由此抛出、由
        # capture 层的既有 try/except 整体降级 eager。禁止在此吞 TypeError
        # 重试——重试会二次执行捕获体（kernel 双跑），且掩盖真实异常源。
        with torch.cuda.graph(graph, pool=pool):
            yield

    # ---- 算子融合点：默认实现 = 现行为逐字迁移（CUDA/CoreX 位等保真，含编译装饰）；
    # Ascend 覆盖为 npu 融合算子（npu_rms_norm/npu_add_rms_norm/...，见 ascend.py）----
    @staticmethod
    @torch.compile
    def rms_norm(x, weight, eps):
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + eps))
        x = x.to(orig_dtype).mul_(weight)
        return x

    @staticmethod
    @torch.compile
    def add_rms_norm(x, residual, weight, eps):
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + eps))
        x = x.to(orig_dtype).mul_(weight)
        return x, residual

    @staticmethod
    @torch.compile
    def silu_and_mul(x):
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    @staticmethod
    def rope(positions, q, k, cos_sin_cache):
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        return q, k

    def get_kv_cache_shape(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        # 默认 CUDA 布局：
        # [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        return (
            2,
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )

    def graph_slot_mapping_dtype(self):
        # CUDA Graph 的 static slot_mapping buffer dtype。
        # 默认 int32（CUDA triton store kernel 的约定 dtype，-1 padding 跳过）；
        # 若 vendor kernel 要求 int64，由 backend override（见 CoreXBackend）。
        return torch.int32

    def all_reduce(self, tensor):
        import torch.distributed as dist

        dist.all_reduce(tensor)
        return tensor
