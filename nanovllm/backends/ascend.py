# Ascend 310P（Atlas 300I Duo / 200I Pro）后端。
#
# 硬件约束（vllm-ascend 310p 硬件节）：仅 fp16（ATB 算子不支持 bf16，default_dtype
# 强制）、不支持 triton（本路径不依赖）、prefill causal mask 按 max_model_len 全量构建。
# 算子调用形态对齐 nano-vllm-ascend 参考实现（prefill TND / decode BNSD）。
#
# torch_npu 懒加载：模块顶部只 import torch——CUDA 机器 import nanovllm 零 NPU 依赖；
# torch_npu 的导入与承重算子校验收口在 _torch_npu() 首用引导。

import os
from contextlib import contextmanager

import torch

from .base import BackendBase

_npu = None    # torch_npu 模块缓存（_torch_npu 首用引导后填充）

# 承重算子清单：任一缺失即无法运行（老版本 torch_npu/CANN）
_REQUIRED_OPS = (
    "_npu_reshape_and_cache",
    "npu_fused_infer_attention_score_v2",
    "npu_add_rms_norm",
    # npu_rms_norm 与 add_rms_norm 的清单不对称：rms_norm()（首层 input_layernorm
    # residual=None、q_norm/k_norm 热路径）在用却未入门——老 torch_npu 会穿过启动
    # 门，拖到首次 forward 深处才 AttributeError，故补入提前校验。
    "npu_rms_norm",
    "npu_swiglu",
)


def _torch_npu():
    """torch_npu 首用引导：懒导入 + 承重算子存在性校验（报错带版本指引）。"""
    global _npu
    if _npu is None:
        try:
            import torch_npu
        except ImportError as e:
            raise RuntimeError(
                "未找到 torch_npu——ascend310p 后端需要 NPU 环境的 torch_npu/CANN"
                "（参考 vllm-ascend：torch-npu==2.10.0.post4 / CANN 9.1.0）"
            ) from e
        missing = [op for op in _REQUIRED_OPS if not hasattr(torch_npu, op)]
        if missing:
            raise RuntimeError(
                f"当前 torch_npu 缺少 ascend310p 后端必需的算子: {', '.join(missing)}"
                "——请升级 torch_npu/CANN（参考 vllm-ascend："
                "torch-npu==2.10.0.post4 / CANN 9.1.0）"
            )
        _npu = torch_npu
    return _npu


def setup_env():
    """收口 torch_npu 的环境耦合：必须在 torch_npu 首次 import 前调用。

    CANN 自带原生 protobuf 运行时，torch_npu 捆绑的 TorchAir Python protobuf
    绑定会再加载一个与之冲突的 C++ protobuf 运行时；切到纯 Python 实现规避。
    参考实现板端实证同款做法（nano_vllm_torchair_bootstrap.py:8-12）。
    setdefault 幂等：外部已显式设置时不覆盖。
    """
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


class Ascend310PBackend(BackendBase):

    name = "ascend310p"
    # TP 分布式走 HCCL（init_process_group 端口用 config.hccl_port）。
    dist_name = "hccl"
    # spec 全路径不可用：verify 恒传 block_tables → attention 层走整池 4D 分支，
    # TND 算子不消费该形态（首个 spec 轮即形状报错硬崩）——引擎层按 supports_spec
    # 在出现 spec_model 配置时干净拒绝；paged-TND 适配后再开（decode 图仍尝试）。
    supports_spec = False
    supports_spec_graph = False
    # 设备语义串：torch.set_default_device 与张量 .to(device) 的目标。
    device_name = "npu"
    # 310P ATB 算子不支持 bf16 → 强制 fp16（加载权重时隐式转换）。
    default_dtype = torch.float16
    # NPU RNG 图安全性未验证，decode 图内采样起步保守关闭。
    graph_sample_default = False
    # prefix cache 命中形态（prefill_attention 收到整池 4D k/v 切片）不支持，
    # 引擎据此强制关闭 prefix cache。两条依据：
    # ① 本后端 TND prefill（npu_fused_infer_attention_score_v2）只直读当前批
    #    k/v，不消费整池 4D 切片（登记见 prefill_attention 内注释）；
    # ② flash_incre 档要求 q/kv 等长，prefix-hit 的 kv 长度 > q 长度直接违反
    #    该约束（计划 Scope 第 3 条）。
    supports_prefix_cache = False

    def __init__(self):
        # 环境收口必须先于 _torch_npu() 的任何调用：torch_npu 是懒加载 import，
        # __init__ 首行时尚未 import，在此调 setup_env() 正好赶在其前。
        setup_env()
        # 选型即校验：torch_npu 缺包/缺算子在 get_backend() 处干净报错，
        # 不拖到首次 forward 才在模型栈深处炸出难读 traceback。
        _torch_npu()
        # causal 掩码缓存（bool，上三角 True = 遮蔽位）：懒构建、只增不减；
        # _mask_fp16 供 op 拒收 bool 掩码时的重试。
        self._mask = None
        self._mask_fp16 = None
        self._mask_device = None
        # bool 掩码被 CANN 拒收后的偏好锁存（见 prefill_attention）。
        self._mask_wants_fp16 = False

    # ---- 设备代理（Tier-2 收口）：base 默认 torch.cuda.*，NPU 机器必须镜像 torch.npu.* ----

    def set_device(self, rank):
        torch.npu.set_device(rank)

    def synchronize(self):
        torch.npu.synchronize()

    def empty_cache(self):
        torch.npu.empty_cache()

    def mem_get_info(self):
        return torch.npu.mem_get_info()

    def memory_stats(self):
        return torch.npu.memory_stats()

    def reset_peak_memory_stats(self):
        torch.npu.reset_peak_memory_stats()

    # ---- 图 API 工厂（npu 分支）----

    def new_graph(self):
        return torch.npu.NPUGraph()

    @contextmanager
    def graph_ctx(self, graph, pool=None):
        # 纯透传（同 base 约定）：厂商 torch 不支持 pool 时的 TypeError 由此抛出、
        # 由 capture 层整体降级 eager。禁止在此吞 TypeError 重试——重试会二次
        # 执行捕获体（kernel 双跑），且掩盖真实异常源。
        with torch.npu.graph(graph, pool=pool):
            yield

    # ---- KV 写入与 attention ----

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping):
        tn = _torch_npu()
        # 引擎传入的 k/v_cache 是 KV 池 [2, L, B, S, H, D] 的层切片（连续），
        # 即 BSHD [num_blocks, block_size, kv_heads, head_dim]；
        # slot_mapping 为全局扁平槽位（block_idx*block_size+offset），CUDA 同族布局；
        # 图路径 padding 槽位为 -1，算子跳负槽（与 CUDA triton 同语义——vllm-ascend
        # 310P 生产路径同款依赖），勿改出无守卫的变体。
        blocks, block_size, kv_heads, head_dim = k_cache.shape
        tn._npu_reshape_and_cache(
            key,
            value,
            k_cache.view(blocks, block_size, kv_heads, head_dim),
            v_cache.view(blocks, block_size, kv_heads, head_dim),
            # 算子约定 int32：图路径静态 buffer 已是 int32（.int() 为 no-op），
            # eager 路径就地转型。
            slot_mapping.int(),
        )

    def prefill_attention(self, q, k, v, context, scale):
        tn = _torch_npu()
        # 图兼容双模：capture/verify 路径以 int32 张量传 cu_seqlens（tolist 的
        # D2H 同步不可入图）；eager prefill 用 list（参考实现便捷口径）。
        if context.use_cuda_graph:
            actual_seq_qlen = context.cu_seqlens_q[1:]
            actual_seq_kvlen = context.cu_seqlens_k[1:]
        else:
            actual_seq_qlen = context.cu_seqlens_q[1:].tolist()
            actual_seq_kvlen = context.cu_seqlens_k[1:].tolist()

        def _call(mask):
            # TND 直读当前批 k/v；prefix cache 命中路径（k/v=整池 4D 切片）
            # 不在本算子消费范围（对齐参考实现，310P 上需回避该负载形态）。
            return tn.npu_fused_infer_attention_score_v2(
                q, k, v,
                num_query_heads=q.shape[-2],
                num_key_value_heads=k.shape[-2],
                input_layout="TND",
                softmax_scale=scale,
                atten_mask=mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                sparse_mode=3,      # 因果：左上对齐下三角
                inner_precise=1,
            )

        if not self._mask_wants_fp16:
            try:
                return _call(self._causal_mask(q.device, context))[0]
            except (RuntimeError, TypeError):
                # 部分 CANN 版本拒收 bool 掩码 → fp16（1=遮蔽，0=保留）；
                # 锁存偏好：拒收型 CANN 上不再每轮先抛一次异常（热路径开销 +
                # 混淆真实形状错误的报错栈）。
                self._mask_wants_fp16 = True
        return _call(self._causal_mask_fp16(q.device, context))[0]

    def decode_attention(self, q, k_cache, v_cache, context, scale):
        tn = _torch_npu()
        bs, num_heads, head_dim = q.shape
        # q [bs, H, D] → BNSD [bs, H, 1, D]；paged KV 读取走 block_table + block_size
        q_input = q.view(bs, 1, num_heads, head_dim).transpose(1, 2).contiguous()
        out = tn.npu_fused_infer_attention_score_v2(
            q_input, k_cache, v_cache,
            num_query_heads=num_heads,
            num_key_value_heads=k_cache.shape[-2],
            input_layout="BNSD",
            softmax_scale=scale,
            block_table=context.block_tables,
            block_size=context.block_size,
            actual_seq_kvlen=context.context_lens,
            sparse_mode=0,
            inner_precise=1,
        )[0]
        # [bs, H, 1, D] → [bs, H*D]
        return out.transpose(1, 2).reshape(bs, -1)

    # ---- causal 掩码（prefill/verify 共用，类持有缓存）----

    def _causal_mask(self, device, context):
        # 尺寸取 max_context_len（engine 全局上界）；prepare_prefill / spec verify
        # 均不传该字段（None）→ 退 max_seqlen_k（verify 路径即 max_model_len）。
        size = context.max_context_len or context.max_seqlen_k
        if self._mask is None or self._mask_device != device or self._mask.shape[0] < size:
            # 只增不减：capture 前的 warmup 轮已把容量抬到 max_model_len，
            # 图内不再触发重建（重建算子若被录进图会逐 replay 重放）。
            self._mask = ~torch.tril(torch.ones((size, size), dtype=torch.bool, device=device))
            self._mask_fp16 = None
            self._mask_device = device
        return self._mask

    def _causal_mask_fp16(self, device, context):
        mask = self._causal_mask(device, context)
        if self._mask_fp16 is None:
            self._mask_fp16 = mask.to(torch.float16)
        return self._mask_fp16

    # ---- 算子融合点覆盖 ----
    # （rope 纯 torch、get_kv_cache_shape=CUDA 族布局、graph_slot_mapping_dtype=int32
    #   三者继承 base 默认，不在此覆盖。）

    @staticmethod
    def rms_norm(x, weight, eps):
        # npu_rms_norm 返回 (x, rstd)，只要归一化输出
        x, _ = _torch_npu().npu_rms_norm(x, weight, eps)
        return x

    @staticmethod
    def add_rms_norm(x, residual, weight, eps):
        # op 返回 (x, rstd, residual_out)
        x, _, residual = _torch_npu().npu_add_rms_norm(x, residual, weight, eps)
        return x, residual

    @staticmethod
    def silu_and_mul(x):
        # 默认 dim=-1，与 chunk(2, -1) 的 SwiGLU 语义一致；npu_swiglu 已入承重
        # 清单（老 torch_npu 在 _torch_npu 处干净报错，不回退 torch.compile 版
        # ——inductor-NPU 可用性未验证）。
        return _torch_npu().npu_swiglu(x)
