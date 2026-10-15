"""Qwen3.5 MoE 主机契约验收：专家布局、路由、共享专家与 310P 调度接线。"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


def make_draft(target, output, window, block_size):
    """生成匹配 tiny MoE 的未训练草稿；只作功能验收，绝不用于加速比结论。"""
    import json
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, Qwen3Config, Qwen3ForCausalLM

    config = AutoConfig.from_pretrained(target, local_files_only=True)
    text = getattr(config, "text_config", config)
    if text.model_type != "qwen3_5_moe_text":
        raise ValueError("此生成器只用于 Qwen3.5 MoE 文本检查点")
    if window < 0 or block_size < 2:
        raise ValueError("window 不得为负，block_size 至少为 2")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"拒绝覆盖非空目录：{output}")
    aux = list(range(min(3, text.num_hidden_layers)))
    draft = Qwen3Config(
        vocab_size=1, hidden_size=text.hidden_size,
        intermediate_size=max(64, text.hidden_size * 2),
        num_hidden_layers=2, num_attention_heads=text.num_attention_heads,
        num_key_value_heads=text.num_key_value_heads, head_dim=text.head_dim,
        max_position_embeddings=text.max_position_embeddings,
        layer_types=["sliding_attention", "full_attention"] if window else ["full_attention"] * 2,
        sliding_window=window or None, use_sliding_window=bool(window),
        rms_norm_eps=text.rms_norm_eps, attention_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(35)
        reference = Qwen3ForCausalLM(draft)
        weights = {name.removeprefix("model."): value.contiguous()
                   for name, value in reference.state_dict().items()
                   if name.startswith(("model.layers.", "model.norm."))}
        weights["fc.weight"] = torch.randn(text.hidden_size, len(aux) * text.hidden_size) * 0.02
        weights["hidden_norm.weight"] = torch.ones(text.hidden_size)
    # 不保存 embedding/lm_head，运行时共享目标权重。
    draft.vocab_size = text.vocab_size
    draft.architectures = ["DFlashDraftModel"]
    draft.dflash_config = {"block_size": block_size, "mask_token_id": 0,
                           "target_layer_ids": aux}
    draft.flashengine_synthetic_fixture = True
    output.mkdir(parents=True, exist_ok=True)
    draft.save_pretrained(output)
    save_file(weights, str(output / "model.safetensors"))
    print(json.dumps({"draft": str(output), "synthetic_untrained": True,
                      "bytes": (output / "model.safetensors").stat().st_size,
                      "aux_layers": aux, "sliding_window": window,
                      "block_size": block_size}, ensure_ascii=False))


def cuda_attention_check():
    """直接比较草稿内核输出；目标 verifier 位等本身不能证明草稿窗口正确。"""
    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_varlen_func
    from flashengine.backends.cuda_dflash import attention
    from flashengine.models.dflash import block_attention_mask

    torch.manual_seed(35)
    device = "cuda"
    query = torch.randn(3, 4, 2, 16, device=device, dtype=torch.float16)
    key = torch.randn(3, 4, 1, 16, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    ctx_key = torch.randn(3, 12, 1, 16, device=device, dtype=torch.float16)
    ctx_value = torch.randn_like(ctx_key)
    lengths = torch.tensor([0, 2, 9], device=device, dtype=torch.int32)

    def reference(causal, window):
        k = torch.cat((ctx_key, key), dim=1).transpose(1, 2).repeat_interleave(2, 1)
        v = torch.cat((ctx_value, value), dim=1).transpose(1, 2).repeat_interleave(2, 1)
        mask = block_attention_mask(lengths, 12, 4, causal, window)
        return F.scaled_dot_product_attention(query.transpose(1, 2).float(), k.float(), v.float(),
                                              attn_mask=mask, scale=0.25).transpose(1, 2).reshape(12, 2, 16)

    for causal in (False, True):
        for window in (None, 1, 4, 32):
            lengths.copy_(torch.tensor([0, 2, 9], device=device))
            def run():
                return attention(flash_attn_varlen_func, query, key, value,
                                 ctx_key, ctx_value, lengths, 0.25, causal, window)
            torch.testing.assert_close(run().float(), reference(causal, window), atol=2e-3, rtol=3e-3)
            # 固定图、不同实际上下文长度；不能只验证捕获时的掩码。
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run()
            lengths.copy_(torch.tensor([8, 0, 12], device=device))
            graph.replay()
            torch.testing.assert_close(output.float(), reference(causal, window), atol=2e-3, rtol=3e-3)
    print("CUDA 草稿注意力通过：8 组全局/因果/双向窗口 × eager/更新长度后的 Graph replay。")


def host_check():
    """CPU 对照 Transformers；NPU 算子用数学替身，仅验证接线，不代表板卡验收。"""
    import torch
    import torch.nn.functional as F
    import flashengine.backends as backends
    from flashengine.backends.base import BackendBase
    from flashengine.backends.ascend import Ascend310PBackend
    from flashengine.models.qwen3_5 import Qwen3_5ForCausalLM
    from flashengine.models.registry import resolve_target_model
    from flashengine.support_matrix import require_supported
    from flashengine.models.dflash import DFlashDraftModel, block_attention_mask
    from flashengine.speculative.registry import block_draft_attention
    from flashengine.config import Config
    from transformers import Qwen3Config, Qwen3_5MoeConfig, Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    class HostBackend(BackendBase):
        def store_kvcache(self, *args):
            raise NotImplementedError

        prefill_attention = decode_attention = store_kvcache

        @staticmethod
        def silu_and_mul(x):
            gate, up = x.chunk(2, dim=-1)
            return F.silu(gate) * up

    cfg = Qwen3_5MoeTextConfig(
        hidden_size=32, vocab_size=128, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .25},
    )
    cfg._experts_implementation = "eager"
    cfg.norm_topk_prob = False  # Qwen3.5 即使带此历史字段，仍须按自身架构归一化。
    torch.manual_seed(123)
    reference = Qwen3_5MoeForCausalLM(cfg).eval()
    weights = list(reference.state_dict().items())
    backend = HostBackend()
    with (patch.object(backends, "_backend", backend),
          patch("torch.distributed.get_rank", return_value=0),
          patch("torch.distributed.get_world_size", return_value=1)):
        fused = Qwen3_5ForCausalLM(cfg).eval()
        fused.load_weights(weights)
        split = []
        for name, value in weights:
            if ".experts.gate_up_proj" in name:
                for index in range(cfg.num_experts):
                    for proj, part in zip(("gate_proj", "up_proj"), value[index].chunk(2, 0)):
                        split.append((name.replace("gate_up_proj", f"{index}.{proj}.weight"), part))
            elif ".experts.down_proj" in name:
                split.extend((name.replace("down_proj", f"{index}.down_proj.weight"), value[index])
                             for index in range(cfg.num_experts))
            else:
                split.append((name, value))
        # 同时覆盖多模态壳前缀；视觉/MTP 权重应跳过。
        wrapped = [("model.language_model." + name[6:] if name.startswith("model.") else name, value)
                   for name, value in split]
        wrapped.extend([("model.visual.unused", torch.zeros(1)), ("mtp.unused", torch.zeros(1))])
        per_expert = Qwen3_5ForCausalLM(cfg).eval()
        per_expert.load_weights(wrapped)
        import copy
        tied_cfg = copy.deepcopy(cfg)
        tied_cfg.tie_word_embeddings = True
        tied = Qwen3_5ForCausalLM(tied_cfg)
        tied.load_weights((name, value) for name, value in weights if name != "lm_head.weight")
        assert tied.lm_head.weight.data_ptr() == tied.model.embed_tokens.weight.data_ptr()
        for (name, value), (other, expected) in zip(fused.named_parameters(), per_expert.named_parameters()):
            assert name == other
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        for bad in (
            [row for row in split if row[0] != "model.layers.0.mlp.experts.1.up_proj.weight"],
            [(name, value[:1] if name == "model.layers.0.mlp.experts.gate_up_proj" else value)
             for name, value in weights],
        ):
            try:
                Qwen3_5ForCausalLM(cfg).load_weights(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("缺失专家/可广播的错误形状必须拒绝")
        for config in (cfg, Qwen3_5MoeConfig(text_config=cfg.to_dict())):
            adapter = resolve_target_model(config)
            assert adapter.name == "qwen3_5_moe" and adapter.is_moe and adapter.recurrent_state
        for name in ("cuda", "corex", "ascend310p"):
            require_supported("qwen3_5_moe", name)
            require_supported("qwen3_5_moe", name, spec_method="dflash")
        # 模拟后端能力而非硬件：窗口草稿不能误分配 NZ/paged 草稿池。
        with TemporaryDirectory(prefix="flashengine-moe-contract-") as directory:
            root = Path(directory)
            cfg.save_pretrained(root / "target")
            for name, paged in (("cuda", False), ("corex", False), ("ascend310p", True)):
                backend.name, backend.supports_moe = name, True
                backend.supports_dflash_paged = paged
                for window in (None, 4):
                    draft_cfg = Qwen3Config(
                        hidden_size=32, intermediate_size=64, vocab_size=128,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=16, max_position_embeddings=512,
                        layer_types=["sliding_attention", "full_attention"] if window else ["full_attention"] * 2,
                        sliding_window=window, use_sliding_window=bool(window))
                    draft_cfg.architectures = ["DFlashDraftModel"]
                    draft_cfg.dflash_config = {"block_size": 4, "mask_token_id": 0,
                                               "target_layer_ids": [0, 1, 2]}
                    draft_cfg.save_pretrained(root / "draft")
                    options = Config(str(root / "target"), spec_model=str(root / "draft"),
                                     max_model_len=512, max_num_seqs=4)
                    draft_model = DFlashDraftModel(options.draft_config, fused.model.embed_tokens,
                                                  fused.lm_head.weight, max_position=512)
                    assert options.spec_aux_layers == (0, 1, 2)
                    assert options.use_paged_draft == draft_model.use_paged_kv == (paged and window is None)
                    assert options.keep_last_block_unhashed == options.use_paged_draft
                backend.supports_dflash_paged = False
        block = fused.model.layers[0].mlp
        for count in (1, 7, 129):
            x = torch.randn(count, cfg.hidden_size)
            with torch.inference_mode():
                actual = block(x)
                expected = reference.model.layers[0].mlp(x[None])[0]
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
        # 310P：替身采用独立的按专家排序、分组 GEMM 与逆置换，检查真实调用链。
        def route(logits, k):
            scores, ids = torch.softmax(logits.float(), -1).topk(k, -1)
            return scores.half(), ids.int(), None

        def dispatch(x, ids, **kwargs):
            assert kwargs["quant_mode"] == -1 and kwargs["row_idx_type"] == 0
            assert ids.dtype == torch.int32
            flat = ids.flatten().long()
            order = flat.argsort(stable=True)
            groups = flat.bincount(minlength=kwargs["expert_num"]).cumsum(0)
            return x.repeat_interleave(ids.shape[1], 0)[order], order, groups, None

        def grouped(x, weight, group_list, **kwargs):
            assert kwargs == {"split_item": 2, "group_list_type": 0, "group_type": 0}
            result, begin = [], 0
            for index, end in enumerate(group_list.tolist()):
                result.append(x[0][begin:end] @ weight[0][index])
                begin = end
            return [torch.cat(result)]

        def unpermute(permuted_tokens, sorted_indices, probs):
            flat = torch.empty_like(permuted_tokens)
            flat[sorted_indices] = permuted_tokens
            return (flat.view(*probs.shape, -1) * probs[..., None]).sum(1)

        npu = SimpleNamespace(
            npu_moe_gating_top_k_softmax=route, npu_moe_init_routing_v2=dispatch,
            npu_grouped_matmul=grouped, npu_moe_token_unpermute=unpermute,
            npu_swiglu=backend.silu_and_mul, npu_format_cast=lambda x, fmt: x.contiguous(),
        )
        ascend = object.__new__(Ascend310PBackend)
        reference.half()
        fused.half()
        with patch("flashengine.backends.ascend._torch_npu", return_value=npu):
            assert ascend.optimize_moe_weights(fused) == 2 * cfg.num_hidden_layers
            block.backend = block.experts.backend = ascend
            for count in (1, 7, 1031):
                # 1031 行额外覆盖路由算子的 1024 行分块边界。
                x = torch.randn(count, cfg.hidden_size).half()
                with torch.inference_mode():
                    actual = block(x)
                    expected = reference.model.layers[0].mlp(x[None])[0]
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-3)
            options = SimpleNamespace(tensor_parallel_size=1, model_config=cfg,
                                      kvcache_block_size=128, is_moe=True)
            ascend.validate_config(options)
            bad_cfg = SimpleNamespace(hidden_size=8, head_dim=16)
            options.model_config = bad_cfg
            try:
                ascend.validate_config(options)
            except ValueError:
                pass
            else:
                raise AssertionError("310P 必须在启动时拒绝未按 16 对齐的 MoE 隐层")
    # 独立标量参考覆盖 padding、零上下文、双向窗口和逐层因果默认值。
    lengths = torch.tensor([0, 2, 9])
    for causal in (False, True):
        for window in (None, 1, 4, 32):
            actual = block_attention_mask(lengths, 10, 5, causal, window)
            expected = torch.zeros(3, 1, 5, 15, dtype=torch.bool)
            for b, length in enumerate(lengths.tolist()):
                for q in range(5):
                    for k in range(15):
                        pos = k if k < 10 else length + k - 10
                        delta = length + q - pos
                        expected[b, 0, q, k] = (
                            (k >= 10 or k < length) and (not causal or delta >= 0)
                            and (window is None or (delta < window and (causal or -delta < window))))
            assert torch.equal(actual, expected)
            q = torch.randn(3, 2, 5, 16)
            key = torch.randn(3, 2, 15, 16)
            value = torch.randn_like(key)
            # 直接执行 310P 的稠密注意力方法（纯数学，不调用 NPU）。
            result = ascend.dense_noncausal_attention(q, key, value, actual, .25)
            reference = F.scaled_dot_product_attention(q, key, value, attn_mask=expected, scale=.25)
            torch.testing.assert_close(result, reference, atol=1e-6, rtol=1e-5)
    for override in (None, False, True):
        c = SimpleNamespace(num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"],
                            sliding_window=4, is_causal=override)
        assert block_draft_attention(c) == ((True if override is None else override, 4),
                                           (False if override is None else override, None))
    c.sliding_window = None
    try:
        block_draft_attention(c)
    except ValueError:
        pass
    else:
        raise AssertionError("滑窗层缺少窗口必须拒绝")
    print("主机契约通过：MoE 三后端接线与权重、310P 调度、DFlash 逐层因果/窗口掩码。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--host-only", action="store_true",
                        help="执行 CPU 契约；端到端请用 scripts.check_qwen35 MODEL")
    mode.add_argument("--make-draft", metavar="TARGET", help="为本地 MoE 生成未训练功能草稿")
    mode.add_argument("--cuda-attention-only", action="store_true", help="CUDA 草稿全局/窗口内核对照")
    parser.add_argument("--output", default="bench_results/qwen35-check/models/tiny-moe-dflash")
    parser.add_argument("--draft-window", type=int, default=8, help="草稿滑窗；0 表示旧版全局非因果")
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    if args.host_only:
        host_check()
    elif args.cuda_attention_only:
        cuda_attention_check()
    else:
        make_draft(args.make_draft, args.output, args.draft_window, args.block_size)


if __name__ == "__main__":
    main()
