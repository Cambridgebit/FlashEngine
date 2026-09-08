# Ascend 310P（Atlas 300I Duo / 200I Pro）后端。
#
# 硬件约束（vllm-ascend 310p 硬件节）：仅 fp16（ATB 算子不支持 bf16，default_dtype
# 强制）、不支持 triton（本路径不依赖）、prefill causal mask 按 max_model_len 全量构建。
# 算子调用形态对齐 nano-vllm-ascend 参考实现（prefill TND / decode BNSD）。
#
# torch_npu 懒加载：模块顶部只 import torch——CUDA 机器 import nanovllm 零 NPU 依赖；
# torch_npu 的导入与承重算子校验收口在 _torch_npu() 首用引导。

import json
import os
import subprocess
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


# _has_op 算子探测结果缓存（算子名 → bool）
_OP_CACHE = {}
# indexing fallback 的一次性告警锁存（print 热路径每 token 都过，只告警一次）
_INDEXING_WARNED = False
# auto 档 probe 裁决回落的一次性告警锁存（resolve_auto_attention_impl 用，
# 与 _INDEXING_WARNED 同款模式；构造期正常只解析一次，测试直调防刷屏）。
_AUTO_FALLBACK_WARNED = False
# graph_ctx pool kwarg TypeError 兜底的一次性告警锁存（pool 不被支持时每张
# 图捕获都会触发同一 TypeError，锁存后只告警一次；与上两处同款模式）。
_POOL_WARNED = False


def _has_op(name):
    """探测 torch_npu 是否提供指定算子（hasattr 结果按算子名缓存）。

    与 _torch_npu() 的启动门语义不同：这里是能力探测（store_kvcache 的
    fused/indexing 分派用），import 失败静默返回 False，不抛 RuntimeError。
    仅在后端实例方法内调用——构造函数已确保 setup_env() 先于 torch_npu
    import，此处不破坏懒加载边界。
    """
    if name not in _OP_CACHE:
        try:
            import torch_npu
        except ImportError:
            _OP_CACHE[name] = False
        else:
            _OP_CACHE[name] = hasattr(torch_npu, name)
    return _OP_CACHE[name]


# ACL_FORMAT_FRACTAL_NZ：CANN 分形 NZ 排布的 format 编号（npu_format_cast 的
# 目标 format；参考实现 probe_310p_native_attention.py 与 probe_ascend310p.py
# 的 _check_flash_nz 同款常量）。
ACL_FORMAT_FRACTAL_NZ = 29


def _nd_to_nz_rearrange(mask_2d):
    """nd → NZ 分形重排的纯 torch 前半段（CPU 可跑、零 torch_npu 依赖）。

    写法逐行对照参考实现 probe_310p_native_attention.py:33-38（即其注释
    "Equivalent to vllm_ascend.utils.nd_to_nz_2d for aligned dimensions" 的
    等价展开）：[S, S] → reshape(1, S, S//16, 16) → transpose(1, 2) →
    contiguous()，得 [1, S//16, S, 16] 连续张量——npu_format_cast(
    ACL_FORMAT_FRACTAL_NZ) 的输入形态。后半段 npu_format_cast 为 NPU 专属
    调用（本机无 torch_npu 不可执行），由 _nz_causal_mask 在设备上收口，
    板端数值契约由 probe 的 _check_flash_nz 裁决；本函数即微测对照的引擎侧
    纯函数形态。要求边长 16 对齐（调用方守卫负责）。
    """
    s0, s1 = mask_2d.shape
    return mask_2d.reshape(1, s0, s1 // 16, 16).transpose(1, 2).contiguous()


def _gather_kv_history(cache, block_table_row, ctx_len, block_size):
    """按单序列 block_table 行从 KV 池层切片 gather 历史并截断（纯 torch，CPU 可跑）。

    flash_incre 档 decode 主路径的历史收集步（每序列每层每步一次）。写法对照
    参考实现 compat._decode（nano_vllm_310p_compat.py:254-276）：块 id 行上取
    整对齐 block_size → index_select 取整块 → reshape 展平 → 截断 [:ctx_len]。

    cache: [num_blocks, block_size, H, D]（KV 池 [2,L,B,S,H,D] 的层切片，BSHD）；
    block_table_row: 该序列块 id 行（设备 int32 padded，可长于所需——padded 尾
    部块 id 由上取整切片排除）；ctx_len: 该序列当前上下文长度（Python int，
    来自 context_lens .item() 的 host sync 产物）；block_size: KV 块大小。

    gather 上界 = ceil(ctx_len/block_size) 个整块（展平后 ceil*block_size 行），
    再截断 [:ctx_len] 丢弃尾块残留行；行序 = 块表顺序拼接（块 b 的行紧接块表
    下一块的行）。返回 [ctx_len, H, D]。
    """
    block_count = (ctx_len + block_size - 1) // block_size    # 上取整对齐块整
    block_ids = block_table_row[:block_count].to(torch.long)
    hist = cache.index_select(0, block_ids).reshape(-1, cache.shape[-2], cache.shape[-1])
    return hist[:ctx_len]


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
    # attention 路径档位：auto→按 probe 报告裁决（后续任务注入实际值）；
    # fia_v2→npu_fused_infer_attention_score_v2 现路径；flash_incre→新路径。
    # store_kvcache 分派对 auto 暂按"探测到 fused 算子即走 fused"处理（与
    # 最终语义一致，probe 裁决接管后不影响分派结果）。
    attention_impl = "auto"
    # decode paged 候选优化开关：默认 False（逐序列主路径兜底）。置 True 条件
    # = probe 报告 paged_attention 项 PASS 且显式选 flash_incre+paged 档（任务
    # 9/10 依 probe 报告决定是否置 True）；auto / 无 probe 报告一律走逐序列
    # 主路径（_decode_flash 内分派）。
    paged_candidate_enabled = False

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
        # NZ 分形掩码缓存（flash_incre 档 prefill 专用，fp16 -inf 形态 + NZ 排布）：
        # 与上面 bool 掩码缓存并列实现（形态不同，不共用），同"只增不减"模式。
        self._nz_mask = None
        self._nz_mask_device = None

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
        # 构造期 vs 体执行期严格区分（关键安全语义，演进自旧版整条禁令
        # "禁止在此吞 TypeError 重试——重试会二次执行捕获体"）：
        # - 旧禁令针对的是【体执行期】异常：__enter__ 之后捕获体已在执行，
        #   吞掉重试 = 捕获体二次执行（kernel 双跑、KV 双写），必须原样穿透
        #   到 capture 层整体 try/except 降级 eager——本任务不变该语义；
        # - 而 torch.npu.graph(graph, pool=pool) 的 TypeError 发生在【上下文
        #   管理器构造期】（pool kwarg 不被该版本 torch_npu 接受；构造与
        #   __enter__ 分离的语义与 torch.cuda.graph 同族），此刻捕获体一行
        #   都没执行——构造无共享池版本重试是安全的，且把"整图降级 eager"
        #   改善为"无共享池捕获成功"（逐图独立内存，显存代价见 AGENTS.md）。
        try:
            ctx = torch.npu.graph(graph, pool=pool)
        except TypeError:
            # pool 不支持时每张图捕获都会抛同一 TypeError，锁存一次性告警
            # （_POOL_WARNED，与 _INDEXING_WARNED/_AUTO_FALLBACK_WARNED 同款）。
            global _POOL_WARNED
            if not _POOL_WARNED:
                _POOL_WARNED = True
                print("[ascend310p] torch.npu.graph 不支持 pool kwarg，已改用无共享池捕获"
                      "（逐图独立内存，显存代价见 AGENTS.md）")
            ctx = torch.npu.graph(graph)
        # with 在 try/except 之外：ctx.__enter__ 与捕获体执行期的任何异常
        # （含 __enter__ 之后的）不在此捕获，原样穿透到 capture 层降级链
        # （model_runner.capture_cudagraph 的告警 → graphs=None → eager）。
        with ctx:
            yield

    # ---- attention auto 档裁决（model_runner 构造期调用一次，非热路径）----

    def resolve_auto_attention_impl(self, configured: str) -> str:
        """auto 档最终裁决：读 probe 报告（scripts/probe_ascend310p.py 产出，
        T8 契约含 toolchain/repo_hash/recommended_attention_impl/checks）的
        recommended_attention_impl，翻译成具体档位返回。

        纯函数式：只返回档位值，不自行写 self.attention_impl（由调用方
        model_runner 构造期赋值，热路径 store/prefill/decode 只读该值分派）。

        采信规则（计划 Verification strategy 定死）：
        - 显式档位（fia_v2/flash_incre）→ 原样返回，尊重用户选择；
        - 报告 repo_hash 与当前 git HEAD 不一致 → 报告不可信，告警回落
          flash_incre；git 调用失败（非仓库/无 git 环境）视同不一致处理；
        - 报告文件缺失 / 解析失败 / recommended_attention_impl 缺失或不在
          值域 {"fia_v2", "flash_incre"} → 同样告警回落 flash_incre。
        所有回落路径走模块级一次性中文告警（_AUTO_FALLBACK_WARNED 锁存）。
        """
        if configured != "auto":
            return configured

        def _fallback(reason):
            global _AUTO_FALLBACK_WARNED
            if not _AUTO_FALLBACK_WARNED:
                _AUTO_FALLBACK_WARNED = True
                print(f"[ascend310p] attention auto 档回落 flash_incre（原因：{reason}）；"
                      "板端跑 scripts.probe_ascend310p --save-out 可修复裁决")
            return "flash_incre"

        path = os.environ.get("NANOVLLM_PROBE_REPORT") or ".omo/evidence/ascend310p-backend/probe_report.json"
        try:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        except FileNotFoundError:
            return _fallback(f"probe 报告不存在：{path}")
        except (OSError, ValueError) as e:
            # ValueError 覆盖 json.JSONDecodeError（其父类）；OSError 覆盖
            # 路径是目录/权限类错误。
            return _fallback(f"probe 报告解析失败：{e!r}")
        # repo_hash 对比：报告的 repo_hash 来自写入时的 git HEAD。git 调用失败
        # （非仓库环境）视同不一致——无法核验的报告不采信，不抛出。
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            head = proc.stdout.strip() if proc.returncode == 0 else ""
        except Exception:
            head = ""
        if not head or head != report.get("repo_hash"):
            return _fallback(f"报告 repo_hash={report.get('repo_hash')!r} 与当前 HEAD={head!r} 不一致")
        impl = report.get("recommended_attention_impl")
        if impl not in ("fia_v2", "flash_incre"):
            return _fallback(f"recommended_attention_impl 缺失或不在值域：{impl!r}")
        return impl

    # ---- KV 写入与 attention ----

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping):
        # 分派入口（对外签名不变，layers/attention.py 唯一调用点）：
        # - 显式 fia_v2 档，或探测到 fused 算子（auto 档默认语义）→ 算子写入；
        # - 算子缺失 → 纯 torch 高级索引写入 + 一次性告警。
        # 显式 fia_v2 但环境不满足时由 _store_fused 内 _torch_npu() 抛清晰错误
        # （启动门语义），不静默回落。
        if self.attention_impl == "fia_v2" or _has_op("_npu_reshape_and_cache"):
            self._store_fused(key, value, k_cache, v_cache, slot_mapping)
        else:
            global _INDEXING_WARNED
            if not _INDEXING_WARNED:
                _INDEXING_WARNED = True
                print("[ascend310p] 未检测到 _npu_reshape_and_cache 算子，"
                      "KV cache 写入已回落纯 torch 高级索引路径（仅限 eager，性能次优）")
            self._store_indexing(key, value, k_cache, v_cache, slot_mapping)

    @staticmethod
    def _store_fused(key, value, k_cache, v_cache, slot_mapping):
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

    @staticmethod
    def _store_indexing(key, value, k_cache, v_cache, slot_mapping):
        # 纯 torch 高级索引写入（fused 算子缺失时的 fallback）：KV 池按槽位展平
        # 成 [B*S, H, D] 后按扁平槽位（block_idx*block_size+offset）直接赋值。
        # 本 fallback 仅限 eager 路径——eager 无 -1 padding，下述 >=0 掩码是
        # 防御层；不得进入图捕获（动态布尔索引不可捕获，图路径必须走 fused）。
        # 掩码是硬要求：图路径 padding 槽位为 -1（model_runner 静态 slot buffer
        # fill_(-1)），高级索引负槽会按 Python 负索引写池尾 = 静默 KV 污染
        # （fused 算子有负槽跳过语义，纯 torch 没有该语义）。
        blocks, block_size, kv_heads, head_dim = k_cache.shape
        valid = slot_mapping >= 0
        idx = slot_mapping.to(torch.long)[valid]
        k_cache.view(blocks * block_size, kv_heads, head_dim)[idx] = key.to(k_cache.dtype)[valid]
        v_cache.view(blocks * block_size, kv_heads, head_dim)[idx] = value.to(v_cache.dtype)[valid]

    def prefill_attention(self, q, k, v, context, scale):
        # 档位分派（attention_impl 选型语义见类属性注释）：
        # - flash_incre → _prefill_flash_nz（310P 板端唯一实证形态，保底档，
        #   eager-only）：图兼容双模（use_cuda_graph 张量/list）仅 fia_v2 档
        #   承担——flash_incre 档下引擎图捕获会失败降级 eager（可接受，capture
        #   的 try/except 已兜底）；
        # - fia_v2 / auto → 下方 TND 现路径原样。auto 暂按 fia_v2 处理（与 T5
        #   store 的 auto 语义对齐：先给可用路径，auto 的最终裁决归 probe 报告，
        #   任务 9 接管后注入实际档位值）。
        if self.attention_impl == "flash_incre":
            return self._prefill_flash_nz(q, k, v, context, scale)
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
        # 档位分派（attention_impl 选型语义见类属性注释）：
        # - flash_incre → _decode_flash（npu_incre_flash_attention 逐序列主路径，
        #   eager-only：host sync 不可入图，本档下引擎 decode 图捕获失败自动
        #   降级 eager，capture 的 try/except 已兜底）；
        # - fia_v2 / auto → 下方 BNSD paged 现路径原样。auto 暂按 fia_v2 处理
        #   （与 store/prefill 的 auto 语义对齐：先给可用路径，auto 的最终裁决
        #   归 probe 报告，任务 9 接管后注入实际档位值）。
        if self.attention_impl == "flash_incre":
            return self._decode_flash(q, k_cache, v_cache, context, scale)
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

    # ---- flash_incre 档 decode（npu_incre_flash_attention 逐序列主路径）----
    # 板端实证形态（参考实现 compat._decode、本仓 probe _check_incre 同源，两处
    # 须保持同构）：逐序列按 block_table 从整池 gather K/V 历史 → BNSD 单序列
    # incre_flash 前向 → cat 拼回。逐序列循环 + .item() 的 host sync ×每序列×
    # 每层×每步是吞吐瓶颈点（唯一被板端证实的形态，后续性能轮优化对象）。
    # 批量 paged 形态为 UNVERIFIED 候选优化，默认关闭（_decode_paged_candidate）。

    def _decode_flash(self, q, k_cache, v_cache, context, scale):
        """flash_incre decode 主路径：逐序列 gather 历史 + incre_flash（板端实证）。

        参数组逐项对照 probe _check_incre 与参考实现 compat._decode
        （nano_vllm_310p_compat.py:245-296）：每序列 q_part [1,H_q,1,D]、历史
        k/v [1,H_kv,ctx,D]（BNSD）、actual_seq_lengths=[ctx]（list）、
        inner_precise=1；输出 squeeze/transpose 后 cat 拼 [bs,H_q,D]。

        eager-only：.item() host sync 不可入图——本档下引擎 decode 图捕获失败
        自动降级 eager（capture try/except 已兜底，同 prefill flash_incre 档）。
        """
        tn = _torch_npu()
        if self.paged_candidate_enabled and _has_op("_npu_paged_attention"):
            # 候选优化分派：paged_candidate_enabled 由任务 9/10 依 probe 报告
            # 显式置 True（probe paged_attention 项 PASS + 显式 flash_incre+
            # paged 才启用）；auto / 无报告一律走下方逐序列主路径。
            return self._decode_paged_candidate(q, k_cache, v_cache, context, scale)
        block_size = int(context.block_size)
        outputs = []
        # 逐序列循环：每序列每层每步一次 Python 迭代 + 单序列算子 launch——
        # host 开销点之一（吞吐瓶颈点，后续性能轮优化对象）。
        for i in range(q.shape[0]):
            # host sync：context_lens D2H（.item()）——逐序列每层每步一次，
            # 吞吐瓶颈点（后续性能轮优化对象）。
            ctx_len = int(context.context_lens[i].item())
            k_hist = _gather_kv_history(k_cache, context.block_tables[i], ctx_len, block_size)
            v_hist = _gather_kv_history(v_cache, context.block_tables[i], ctx_len, block_size)
            q_part = q[i:i + 1].transpose(0, 1).unsqueeze(0).contiguous()   # [1,H_q,1,D]
            k_part = k_hist.transpose(0, 1).unsqueeze(0).contiguous()       # [1,H_kv,ctx,D]
            v_part = v_hist.transpose(0, 1).unsqueeze(0).contiguous()
            out = tn.npu_incre_flash_attention(
                q_part, k_part, v_part,
                actual_seq_lengths=[ctx_len],        # compat/probe 同款 list
                num_heads=q.shape[-2],
                num_key_value_heads=k_cache.shape[-2],
                input_layout="BNSD",
                scale_value=scale,
                inner_precise=1,
            )
            # [1,H_q,1,D] → [1,H_q,D]（compat 同款 squeeze/transpose 整形）
            outputs.append(out.squeeze(0).transpose(0, 1).contiguous())
        # [bs,H_q,D] → [bs,H*D]（attention 层消费口径，与 fia_v2 decode 返回形态一致）
        return torch.cat(outputs, dim=0).reshape(q.shape[0], -1)

    def _decode_paged_candidate(self, q, k_cache, v_cache, context, scale):
        """UNVERIFIED 候选：_npu_paged_attention 批量 paged 形态（板端未实证）。

        参数组对照 vllm-ascend 310P 官方形态（vllm_ascend/_310p/attention/
        attention_v1.py forward_paged_attention）与本仓 probe _check_paged 的
        kwarg 组——两者均 UNVERIFIED：官方 cache 为 5D 对齐布局 (2,B,(H*D)/16,
        S,16)，本引擎为 4D 池层切片 [B,S,H_kv,D]；q 形态取 probe 两连试的 A 案
        [bs,1,H,D]（BNSD）。形态差异是否可跑/数值正确以板端 probe 裁决为准，
        默认不启用（启用条件见 _decode_flash 分派注释）。
        """
        tn = _torch_npu()
        bs, num_heads, head_dim = q.shape
        # q [bs, H, D] → BNSD [bs, H, 1, D]（probe A 案同款整形）
        q_input = q.view(bs, 1, num_heads, head_dim).transpose(1, 2).contiguous()
        out = torch.empty_like(q_input)
        tn._npu_paged_attention(
            query=q_input,
            key_cache=k_cache,
            value_cache=v_cache,
            num_kv_heads=k_cache.shape[-2],
            num_heads=num_heads,
            scale_value=scale,
            block_table=context.block_tables,
            context_lens=context.context_lens,
            out=out,
        )
        # [bs,1,H,D] → [bs,H*D]（probe 同款 reshape 口径，attention 层消费形态）
        return out.reshape(bs, -1)

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

    # ---- flash_incre 档 prefill（_npu_flash_attention + NZ 分形掩码）----
    # 310P 板端唯一实证过的 attention 形态（vllm-ascend 310P prefill 同款私有
    # 算子）：_npu_flash_attention + NZ 因果掩码 + q/kv 等长。算子契约与参考
    # 实现 probe_310p_native_attention.py、compat 的 _prefill 及本仓 probe
    # scripts/probe_ascend310p.py 的 _check_flash_nz 同源（引擎侧与探针侧同一
    # 形态，两处须保持同构）。与 fia_v2（TND，性能候选）并存 = 计划决策 D1。

    def _prefill_flash_nz(self, q, k, v, context, scale):
        """flash_incre prefill：q/k/v 等长 varlen [total_tokens, H, D] 单次前向。

        eager-only：NZ 掩码构建与 seq_len 的 CPU 构造均含 host 交互，不可入图
        ——图捕获仅 fia_v2 档承担（见 prefill_attention 分派注释）。
        """
        tn = _torch_npu()
        # 防御层：prefix-hit 形态（k/v=整池 4D 切片、kv 长于 q）不该到达本档
        # ——supports_prefix_cache=False 已在引擎层全局排除，此处兜底报错而非
        # 静默错算。
        if context.block_tables is not None:
            raise RuntimeError(
                "flash_incre 档不消费 prefix-hit 形态（block_tables 非空）——"
                "310P 后端 supports_prefix_cache=False 已全局排除该负载，"
                "出现此错误说明引擎能力位被绕过"
            )
        # 逐序列长度（CPU int32，host sync——本档 eager-only 可接受；算子口径
        # 同 compat._prefill 的 seq_len=逐序列长度，非 cumsum）。
        q_off = context.cu_seqlens_q.to(device="cpu", dtype=torch.int32)
        k_off = context.cu_seqlens_k.to(device="cpu", dtype=torch.int32)
        q_lens = q_off[1:] - q_off[:-1]
        k_lens = k_off[1:] - k_off[:-1]
        if not torch.equal(q_lens, k_lens):
            raise RuntimeError(
                "310P flash_incre 档要求 prefill 逐序列 q/kv 等长——当前批 "
                "cu_seqlens_q/k 不一致（多为 prefix cache 命中形态，该形态应由 "
                "supports_prefix_cache=False 全局排除）"
            )
        out = torch.empty_like(q)
        # 参数组逐项对照 probe _check_flash_nz / compat._prefill：q/k/v varlen
        # 压缩 [total_tokens, H, D]（B=1 隐含）、mask=NZ 分形掩码、seq_len=CPU
        # int32 逐序列长度、out 预分配同形。
        tn._npu_flash_attention(
            query=q.contiguous(),
            key=k.contiguous(),
            value=v.contiguous(),
            mask=self._nz_causal_mask(q.device, context),
            seq_len=q_lens.contiguous(),
            scale_value=scale,
            num_heads=q.shape[-2],
            num_kv_heads=k.shape[-2],
            out=out,
        )
        # [total_tokens, H, D] → [total_tokens, H*D]（attention 层消费口径，
        # 与 decode_attention 的返回形态一致）。
        return out.reshape(q.shape[0], -1)

    def _nz_causal_mask(self, device, context):
        """NZ 分形因果掩码（fp16 -inf 上三角，npu_format_cast 转分形 NZ）缓存。

        尺寸口径同 _causal_mask：max_context_len or max_seqlen_k。max_model_len
        守卫用该目标尺寸作代理判定（context 无 max_model_len 字段）：仅在新
        构建时检查（掩码可大于序列——probe 512 掩码配 seq=2 同理；缓存已覆盖
        时不重建、不再拦）。16 对齐是 NZ 分形排布的硬约束，≤2048 为掩码内存
        上界；该约束仅 flash_incre 档有，fia_v2 档无。
        """
        size = context.max_context_len or context.max_seqlen_k
        if self._nz_mask is None or self._nz_mask_device != device or self._nz_mask.shape[0] < size:
            if size % 16 != 0 or size > 2048:
                raise RuntimeError(
                    f"flash_incre 档 NZ 掩码目标尺寸 {size} 不满足约束"
                    "（须 16 对齐且 ≤2048）——本档以掩码尺寸代理 max_model_len "
                    "校验，请将 max_model_len 配置为 16 的倍数且不超过 2048"
                    "（fia_v2 档无此约束）"
                )
            # 构建写法逐行对照 probe _check_flash_nz：CPU 构建 fp16 -inf 因果
            # 掩码 → nd_to_nz 前半段重排（纯 torch，模块级 _nd_to_nz_rearrange）
            # → 设备上 npu_format_cast 转分形 NZ（NPU 专属，板端数值契约由
            # probe 裁决）。
            mask = torch.zeros((size, size), dtype=torch.float16)
            mask.masked_fill_(
                torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1),
                float("-inf"),
            )
            rearranged = _nd_to_nz_rearrange(mask).to(device)
            self._nz_mask = _torch_npu().npu_format_cast(rearranged, ACL_FORMAT_FRACTAL_NZ)
            self._nz_mask_device = device
        return self._nz_mask

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
