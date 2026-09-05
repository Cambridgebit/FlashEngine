import os
from dataclasses import dataclass
from transformers import AutoConfig


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
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # EAGLE3 投机解码：draft 模型目录（官方 EAGLE-3 权重，含 midlayer/fc/d2t）。
    # 空字符串 = 关闭。chain 模式：draft 链式起草，target 单次前向验证。
    spec_model: str = ""
    # 每轮起草的 token 数（draft depth）。本轮收益 = 接受数 + 1（bonus token）。
    spec_num_tokens: int = 4
    # 目标模型 aux 隐层捕获索引（层输入序号，即第 i-1 层完整输出），
    # 拼接后经 draft fc(3*hidden -> hidden)。必须与 draft 训练时的捕获层一致；
    # 默认 (2, n//2, n-3) 与官方 EAGLE 仓库 / vLLM 的 Qwen3 捕获点一致。
    spec_aux_layers: tuple[int, ...] = ()

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
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.spec_model:
            # spec 解码路径依赖 rank0 上的 draft 状态（h_seed），仅支持 TP=1。
            assert self.tensor_parallel_size == 1, "speculative decoding requires tensor_parallel_size == 1"
            assert os.path.isdir(self.spec_model)
            assert self.spec_num_tokens >= 1
            if not self.spec_aux_layers:
                n = self.hf_config.num_hidden_layers
                self.spec_aux_layers = (2, n // 2, n - 3)
            assert len(self.spec_aux_layers) == 3, "EAGLE3 draft fc 输入为 3 个 aux 隐层拼接"
