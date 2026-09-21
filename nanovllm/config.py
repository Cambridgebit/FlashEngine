import os
from dataclasses import dataclass
from transformers import AutoConfig

from nanovllm.backends import detect_backend


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    # torch.compile 编译模型（主要加速 eager prefill；CoreX 上因 Inductor 融合 bug，
    # Qwen3Attention 整块保持 eager，实测仅约 8% 提速）。
    compile: bool = False
    backend: str = "auto"
    # TP 初始化端口（init_method 的 tcp 端口；Ascend hccl 与 CUDA nccl 共用）。
    hccl_port: int = 28000
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # 前缀缓存开关（block 级 hash 复用，仅 prefill 阶段命中）。
    # 关闭时跳过 hash 计算与命中查找（含每步 decode 的 hash_blocks 开销），
    # 用于 A/B 对照测量其对 TTFT / 吞吐的影响。
    enable_prefix_cache: bool = True
    # EAGLE3 投机解码：draft 模型目录（官方 EAGLE-3 权重，含 midlayer/fc/d2t）。
    # 空字符串 = 关闭。chain 模式：draft 链式起草，target 单次前向验证。
    spec_model: str = ""
    # 每轮起草的 token 数（draft depth）。本轮收益 = 接受数 + 1（bonus token）。
    spec_num_tokens: int = 4
    # 目标模型 aux 隐层捕获索引（层输入序号，即第 i-1 层完整输出），
    # 拼接后经 draft fc(3*hidden -> hidden)。必须与 draft 训练时的捕获层一致；
    # 默认 (2, n//2, n-3) 与官方 EAGLE 仓库 / vLLM 的 Qwen3 捕获点一致。
    spec_aux_layers: tuple[int, ...] = ()
    # spec 方法派生（__post_init__ 按 draft checkpoint 填充）："eagle3" / "dflash"。
    # 判据：draft config 含 dflash_config 键或 architectures 含 DFlash → dflash，
    # 否则 eagle3（官方 EAGLE3 checkpoint 不含这两者，判定不受影响）。
    spec_method: str = ""
    # draft 的 AutoConfig（spec_model 设置时统一在此加载一次，model_runner 复用，
    # 避免二次 IO）。qwen3_5 的 dflash_config.target_layer_ids 亦从此读取。
    draft_config: AutoConfig | None = None
    # ---- hybrid（Qwen3.5）派生标记（__post_init__ 填充；qwen3 恒 False/None）----
    # Qwen3.5 = 18×GDN 线性注意力 + 6×全注意力的混合架构（model_type=qwen3_5，
    # text 形状在嵌套 text_config；纯文本 checkpoint 顶层可能直接是 qwen3_5_text，
    # 一并识别）。引擎所有 hybrid 分支（状态池/图/spec 门控）以此为准。
    is_hybrid: bool = False
    # text 侧形状配置便捷访问：qwen3_5 取嵌套 text_config，qwen3 即 hf_config 本身
    # （引擎统一经 config.text_config 取形状，qwen3 路径取值与原 hf_config 直取逐字节相同）。
    text_config: AutoConfig | None = None

    def __post_init__(self):
        if self.backend != "auto":
            os.environ["NANOVLLM_BACKEND"] = self.backend
        assert os.path.isdir(self.model)
        # block_size 越小，prefix cache 粒度越细（每步最多重 prefill 一个 block）。
        # 限制为 2 的幂；CoreX/IXFormer paged kernel 支持的最小值需实测确认。
        assert self.kvcache_block_size >= 16
        assert self.kvcache_block_size & (self.kvcache_block_size - 1) == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # hybrid 识别 + text_config 便捷访问（qwen3 的 text_config 即 hf_config）。
        model_type = getattr(self.hf_config, "model_type", "")
        self.is_hybrid = model_type in ("qwen3_5", "qwen3_5_text")
        self.text_config = self.hf_config.text_config if self.is_hybrid else self.hf_config
        # max_model_len clamp 经 text_config（qwen3 路径行为不变）。
        self.max_model_len = min(self.max_model_len, self.text_config.max_position_embeddings)
        if self.is_hybrid:
            # ---- hybrid 门控（阶段1 范围）----
            # 1) 强制关闭前缀缓存：GDN 状态随序列连续推进，无法按 block hash 复用，
            #    保留开关只会白付每步 hash 开销。
            if self.enable_prefix_cache:
                print(
                    "[nano-vllm] Prefix caching is disabled for Qwen3.5 hybrid "
                    "models because GDN states cannot be reused by block hash.",
                    flush=True,
                )
                self.enable_prefix_cache = False
            # 2) 禁 chunked prefill：GDN 状态只能整段一次推进（分段 prefill 需要
            #    中间态落池/回填，阶段1 不做）。做法：prefill 批预算封顶到
            #    max_model_len——单步 prefill 至多一条满长序列（短 prompt 仍可
            #    多条同批）。这同时把 warmup 批自动压到 1×seq_len，使「warmup
            #    预留的峰值显存」与「运行时最坏 GDN 前向瞬时」严格一致：8GB 卡上
            #    按 8×2048 批预留会把 KV 池预算直接挤到 0 块（实测 warmup 峰值
            #    5029 MiB vs current 4131 MiB，碎片 ~1GB）。调度器「仅首序列可
            #    切分」分支在 mnbt >= mml 下不可达（核实注释见 scheduler.schedule）。
            if self.max_num_batched_tokens > self.max_model_len:
                print(
                    "[nano-vllm] Qwen3.5 hybrid prefill budget adjusted: "
                    f"max_num_batched_tokens {self.max_num_batched_tokens} -> "
                    f"{self.max_model_len}. Chunked prefill is disabled for GDN state safety.",
                    flush=True,
                )
                self.max_num_batched_tokens = self.max_model_len
            assert self.max_num_batched_tokens >= self.max_model_len, (
                "Qwen3.5 hybrid models disable chunked prefill: "
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be >= "
                f"max_model_len ({self.max_model_len})."
            )
            # 3) 仅 CUDA 后端：GDN 走纯 torch 实现，但状态池/图/spec 语义只在
            #    CUDA 验证。经 detect_backend() 取名（与 backend.name 同源；config
            #    阶段不实例化后端单例，避免 fork/spawn 前的隐式设备初始化）。
            backend_name = detect_backend()
            if backend_name != "cuda":
                raise RuntimeError(
                    "Qwen3.5 hybrid models currently support the CUDA backend only; "
                    f"selected backend: {backend_name}."
                )
            # 4) 阶段1 仅 TP=1：GDN 状态槽按 seq_id 在各 rank 独立分配，而 Sequence
            #    的跨进程 pickle 不携带 seq_id，槽一致性不在本阶段范围（GDN 的 TP
            #    分片本身也待模型侧验证）。
            assert self.tensor_parallel_size == 1, \
                "Qwen3.5 hybrid models currently require tensor_parallel_size == 1."
        if self.spec_model:
            # spec 解码路径依赖 rank0 上的 draft 状态（h_seed），仅支持 TP=1。
            assert self.tensor_parallel_size == 1, "speculative decoding requires tensor_parallel_size == 1"
            assert os.path.isdir(self.spec_model)
            assert self.spec_num_tokens >= 1
            # draft config 统一在此加载一次（model_runner / dflash 模型复用）。
            self.draft_config = AutoConfig.from_pretrained(self.spec_model)
            dc = self.draft_config
            dfc = getattr(dc, "dflash_config", None) or {}
            archs = getattr(dc, "architectures", None) or []
            # spec_method 派生（计划 §4-2b）：dflash_config 键或 architectures 含
            # DFlash → dflash；官方 EAGLE3 checkpoint 两者皆无 → eagle3（原判路）。
            self.spec_method = "dflash" if (dfc or any("DFlash" in str(a) for a in archs)) else "eagle3"
            if self.spec_method == "dflash":
                # k = block_size - 1 自动派生（z-lab 语义：block = 1 anchor + k 个
                # mask 查询行）。spec_num_tokens 无法在 Config 层区分「用户显式传」
                # 与「默认值」：值 ∈ {4（本 dataclass 默认）, 3（benchmark/
                # openai_server 的 CLI 默认）} 一律视为未显式指定而派生；其余值
                # 尊重用户（注：显式传 3/4 也会被派生覆盖，block=16 的 DFlash 用
                # k=3/4 属误用，help 注明）。
                block = int(dfc.get("block_size", getattr(dc, "block_size", 16)))
                if self.spec_num_tokens in (3, 4):
                    self.spec_num_tokens = block - 1
                # aux 捕获层 = dflash_config.target_layer_ids 经目标口径换算：
                # - qwen3_5（hybrid）aux 钩子 = 0-based 层输出口径，直配；
                # - qwen3 dense aux 钩子（qwen3.py）= 层输入口径，而层 l 输出 ≡
                #   层 l+1 输入（同一残差流值），故 +1 换算——qwen3.py 零改动，
                #   与 vllm qwen3 dense 的 dflash aux 转换同构（Qwen3-4B 的
                #   [1,9,17,25,33] → (2,10,18,26,34)）。
                if not self.spec_aux_layers:
                    tli = tuple(dfc["target_layer_ids"])
                    if self.is_hybrid:
                        self.spec_aux_layers = tli
                    else:
                        self.spec_aux_layers = tuple(l + 1 for l in tli)
                assert len(self.spec_aux_layers) == len(dfc["target_layer_ids"]), (
                    "DFlash projection width does not match the number of auxiliary layers: "
                    f"spec_aux_layers={self.spec_aux_layers} vs "
                    f"target_layer_ids={dfc['target_layer_ids']}")
                assert max(self.spec_aux_layers) < self.text_config.num_hidden_layers, (
                    f"DFlash auxiliary layer index is out of range: "
                    f"spec_aux_layers={self.spec_aux_layers}, "
                    f"num_hidden_layers={self.text_config.num_hidden_layers}"
                )
            elif not self.is_hybrid:
                if not self.spec_aux_layers:
                    n = self.hf_config.num_hidden_layers
                    self.spec_aux_layers = (2, n // 2, n - 3)
                assert len(self.spec_aux_layers) == 3, (
                    "EAGLE3 draft projection expects three concatenated auxiliary hidden states."
                )
            # hybrid + spec_model 的方法匹配守卫在 model_runner（原 NotImplementedError
            # 位置替换为 spec_method 分派检查；eagle3+hybrid / dflash+qwen3 均干净拒绝）。
        # 310P：paged attention 只支持 block_size 64/128。prefix-hit 由后端
        # SplitFuse 读取 NZ paged KV；若运行时缺少 SplitFuse，ModelRunner 在后端
        # 实例化后关闭 prefix cache（Config 阶段不 import torch_npu，避免提前初始化）。
        if detect_backend() == 'ascend310p':
            if self.kvcache_block_size not in (64, 128):
                print(
                    f"[nano-vllm] Ascend 310P KV-cache block size adjusted: "
                    f"{self.kvcache_block_size} -> 128; supported values are 64 and 128.",
                    flush=True,
                )
                self.kvcache_block_size = 128
            head_dim = getattr(self.text_config, 'head_dim', None)
            if head_dim is None:
                head_dim = self.text_config.hidden_size // self.text_config.num_attention_heads
            assert self.kvcache_block_size * head_dim <= 128 * 128, (
                "Ascend 310P paged attention requires "
                "block_size * head_dim <= 128*128; "
                f"got {self.kvcache_block_size} * {head_dim}."
            )
