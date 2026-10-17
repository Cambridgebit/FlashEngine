"""将 Python 已确认的 Qwen3 行为冻结为无 Python 部署 bundle。"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile

CONTRACT_VERSION = "0.4.0"
BUNDLE_FORMAT = "1.3.0"
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "I64": 8, "I8": 1, "BOOL": 1}


def parse_json(text):
    """解析 JSON，并拒绝重复键及非标准 NaN/Infinity。"""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复 JSON 键：{key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"非法 JSON 常量：{value}")
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def read_json(path):
    """读取严格 UTF-8 JSON。"""
    return parse_json(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """使用稳定顺序与换行保存发布元数据。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    """流式计算文件哈希，不将权重整份读入内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    """记录 Python oracle 的实际版本。"""
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def source_state():
    """未提交实现也必须记录实际源码哈希，不能只以 HEAD 冒充 oracle 版本。"""
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "--", "flashengine", "scripts", "contracts"], text=True).splitlines()
    sources = {name: sha256(root / name) for name in sorted(set(paths)) if (root / name).is_file()}
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    return dict(git_sha=git_sha(), dirty=dirty, source_tree_sha256=digest, sources=sources)


def safetensors_header(path):
    """验证 safetensors 结构；返回不含元数据的张量描述。"""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"截断的 safetensors：{path}")
        count = struct.unpack("<Q", raw)[0]
        if count > 100_000_000 or count > size - 8:
            raise ValueError(f"非法 safetensors header：{path}")
        encoded = handle.read(count)
        if not encoded.startswith(b"{"):
            raise ValueError("safetensors header 必须以对象起始字符开头")
        header = parse_json(encoded)
    if not isinstance(header, dict):
        raise ValueError("safetensors header 必须为对象")
    tensors, spans = {}, []
    for name, item in header.items():
        if name == "__metadata__":
            if not isinstance(item, dict) or any(not isinstance(value, str) for value in item.values()):
                raise ValueError("safetensors metadata 必须为字符串字典")
            continue
        dtype = item["dtype"]
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"不支持的张量类型：{dtype}")
        elements = 1
        for dim in item["shape"]:
            if type(dim) is not int or dim < 0:
                raise ValueError(f"非法 shape：{name}")
            elements *= dim
            if elements > 2**63 - 1:
                raise ValueError(f"shape 溢出：{name}")
        begin, end = item["data_offsets"]
        if type(begin) is not int or type(end) is not int:
            raise ValueError(f"offset 必须为整数：{name}")
        if not (0 <= begin <= end <= size - 8 - count) or end - begin != elements * DTYPE_BYTES[dtype]:
            raise ValueError(f"非法张量边界：{name}")
        spans.append((begin, end))
        tensors[name] = {"dtype": dtype, "shape": item["shape"], "offset": begin + 8 + count, "bytes": end - begin}
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise ValueError(f"数据重叠或空洞：{path}")
        cursor = end
    if cursor != size - 8 - count:
        raise ValueError(f"未索引的权重数据：{path}")
    return tensors


def normalize_model(raw, *, draft=False):
    """归一化原生 Qwen3/Qwen3.5 文本与 DFlash1/EAGLE3 profile。"""
    if not draft and raw.get("model_type", "").startswith("qwen3_5"):
        return normalize_hybrid(raw)
    architectures = raw.get("architectures", [])
    eagle = draft and any("eagle3" in name.lower() for name in architectures)
    if draft:
        if not eagle and not any("DFlash" in name and "DFlash2" not in name for name in architectures):
            raise ValueError("原生只接受 DFlash1/EAGLE3 草稿")
    elif (raw.get("model_type"), tuple(architectures)) not in {
        ("qwen3", ("Qwen3ForCausalLM",)), ("qwen3_moe", ("Qwen3MoeForCausalLM",))
    }:
        raise ValueError("目标模型仅支持 Qwen3 dense/MoE")
    quant = normalize_quantization(raw.get("quantization_config"))
    if draft and quant:
        raise ValueError("草稿量化尚不支持")
    if raw.get("attention_bias", False) or raw.get("hidden_act", "silu") != "silu":
        raise ValueError("当前 profile 要求无 attention bias、silu 激活")
    if raw.get("use_sliding_window") or any(t != "full_attention" for t in raw.get("layer_types", [])):
        raise ValueError("当前 profile 不支持滑窗注意力")
    if draft and not eagle and raw.get("is_causal", False):
        raise ValueError("当前 DFlash1 profile 仅支持非因果全注意力")
    rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("当前 profile 只支持 default RoPE")
    keys = ("hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads")
    result = {key: int(raw[key]) for key in keys}
    if any(v <= 0 for v in result.values()):
        raise ValueError("模型维度必须为正")
    result.update(
        head_dim=int(raw.get("head_dim", result["hidden_size"] // result["num_attention_heads"])),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
        rope_theta=float(rope.get("rope_theta", raw.get("rope_theta", 1000000))),
        max_position_embeddings=int(raw.get("max_position_embeddings", 4096)),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
    )
    if "layer_types" in raw and len(raw["layer_types"]) != result["num_hidden_layers"]:
        raise ValueError("layer_types 数量与层数不一致")
    if result["head_dim"] % 2 or result["num_attention_heads"] % result["num_key_value_heads"]:
        raise ValueError("非法 GQA/RoPE 维度")
    result["model_type"] = "qwen3" if draft else raw["model_type"]
    if quant:
        result["quantization"] = quant
    if eagle:
        if result["num_hidden_layers"] != 1:
            raise ValueError("EAGLE3 profile 要求一个 midlayer")
        result.update(spec_method="eagle3", draft_vocab_size=int(raw.get("draft_vocab_size", result["vocab_size"])))
        if not 0 < result["draft_vocab_size"] <= result["vocab_size"]:
            raise ValueError("EAGLE3 草稿词表大小非法")
    if result["model_type"] == "qwen3_moe":
        result.update(num_experts=int(raw["num_experts"]), num_experts_per_tok=int(raw["num_experts_per_tok"]), moe_intermediate_size=int(raw["moe_intermediate_size"]), decoder_sparse_step=int(raw.get("decoder_sparse_step", 1)), mlp_only_layers=list(raw.get("mlp_only_layers") or []), norm_topk_prob=bool(raw.get("norm_topk_prob", True)))
        if not 1 <= result["num_experts_per_tok"] <= result["num_experts"] or result["moe_intermediate_size"] <= 0 or result["decoder_sparse_step"] <= 0:
            raise ValueError("非法 MoE 路由/专家维度")
        if len(set(result["mlp_only_layers"])) != len(result["mlp_only_layers"]) or any(type(i) is not int or not 0 <= i < result["num_hidden_layers"] for i in result["mlp_only_layers"]):
            raise ValueError("非法 MoE dense 层列表")
        if raw.get("num_shared_experts", 0):
            raise ValueError("Qwen3 MoE profile 不支持 shared expert")
    elif raw.get("num_experts", 0):
        raise ValueError("dense 配置不能包含专家")
    return result


def normalize_quantization(quant):
    """仅承诺 Quark W8A8 的 int8 权重加载期反量化，不承诺 int8 GEMM。"""
    if not quant:
        return None
    if quant.get("quant_method") != "quark" or quant.get("export", {}).get("pack_method") not in (None, "reorder"):
        raise ValueError("原生仅支持 Quark per-channel int8 反量化")
    groups = quant.get("config_groups") or {}
    if not groups:
        raise ValueError("Quark 缺少 config_groups")
    for group in groups.values():
        w = group.get("weights") or {}
        if w.get("num_bits") != 8 or w.get("type") != "int" or w.get("strategy") != "channel" or w.get("group_size") not in (None, -1):
            raise ValueError("Quark 只支持逐输出通道 int8 权重")
    return "quark-int8-dequant-v1"


def normalize_hybrid(raw):
    """Qwen3.5 只打包文本模型；显式保存 GDN 和门控注意力语义。"""
    if raw.get("model_type") not in ("qwen3_5","qwen3_5_text","qwen3_5_moe","qwen3_5_moe_text"):
        raise ValueError("不支持的 Qwen3.5 family")
    text = dict(raw.get("text_config") or raw)
    if raw.get("quantization_config"):
        text.setdefault("quantization_config",raw["quantization_config"])
    moe = text.get("num_experts", 0) > 0
    proxy = dict(text, model_type="qwen3_moe" if moe else "qwen3", architectures=["Qwen3MoeForCausalLM" if moe else "Qwen3ForCausalLM"])
    proxy.pop("layer_types", None)
    proxy.pop("num_shared_experts", None)
    if moe:
        proxy.setdefault("intermediate_size", text["moe_intermediate_size"])
    result = normalize_model(proxy)
    result["model_type"] = "qwen3_5_moe" if moe else "qwen3_5"
    count = result["num_hidden_layers"]
    kinds = text.get("layer_types") or ["full_attention" if (i+1) % int(text.get("full_attention_interval", 4)) == 0 else "linear_attention" for i in range(count)]
    if len(kinds) != count or set(kinds) - {"full_attention", "linear_attention"}:
        raise ValueError("Qwen3.5 layer_types 非法")
    if not text.get("attn_output_gate", True) or text.get("mamba_ssm_dtype", "float32") != "float32":
        raise ValueError("Qwen3.5 要求门控 attention 和 FP32 SSM")
    for key in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim"):
        result[key] = int(text[key])
        if result[key] <= 0:
            raise ValueError("GDN 维度必须为正")
    if result["linear_num_value_heads"] % result["linear_num_key_heads"]:
        raise ValueError("GDN value heads 必须整除 key heads")
    rope = text.get("rope_parameters") or {}
    rotary = int(result["head_dim"] * float(rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1))))
    if rotary <= 0 or rotary % 2 or rotary > result["head_dim"]:
        raise ValueError("非法 partial RoPE")
    gate = text.get("output_gate_type", "silu")
    if gate not in ("silu", "swish", "sigmoid"):
        raise ValueError("不支持 GDN gate 激活")
    result.update(layer_types=kinds, rotary_dim=rotary, output_gate_type="silu" if gate == "swish" else gate, mamba_ssm_dtype="float32")
    if moe:
        result.update(shared_expert_intermediate_size=int(text["shared_expert_intermediate_size"]), norm_topk_prob=True)
        if result["shared_expert_intermediate_size"] <= 0:
            raise ValueError("共享专家宽度非法")
    return result


def is_moe_layer(config, layer):
    """模型 profile 拥有稀疏层选择语义，调度器不判断模型名称。"""
    return config.get("model_type") in ("qwen3_moe", "qwen3_5_moe") and layer not in config["mlp_only_layers"] and (layer + 1) % config["decoder_sparse_step"] == 0


def expected_weights(config, *, draft=False):
    """显式约束权重完整性，避免空权重运行。"""
    h, d, v, i = (config[k] for k in ("hidden_size", "head_dim", "vocab_size", "intermediate_size"))
    q, kv = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
    result = {"norm.weight": [h]}
    if draft and config.get("spec_method") == "eagle3":
        dv = config["draft_vocab_size"]
        result.update({"fc.weight": [h, 3*h], "lm_head.weight": [dv, h], "d2t": [dv]})
        result.update({"midlayer."+name: shape for name, shape in {
            "self_attn.q_proj.weight": [q, 2*h], "self_attn.k_proj.weight": [kv, 2*h],
            "self_attn.v_proj.weight": [kv, 2*h], "self_attn.o_proj.weight": [h, q],
            "hidden_norm.weight": [h], "input_layernorm.weight": [h], "post_attention_layernorm.weight": [h],
            "mlp.gate_proj.weight": [i,h], "mlp.up_proj.weight": [i,h], "mlp.down_proj.weight": [h,i]}.items()})
        return result
    if draft:
        result.update({"fc.weight": [h, len(config["target_layer_ids"]) * h], "hidden_norm.weight": [h]})
    else:
        result["embed_tokens.weight"] = [v, h]
        if not config["tie_word_embeddings"]:
            result["lm_head.weight"] = [v, h]
    for lid in range(config["num_hidden_layers"]):
        prefix = f"layers.{lid}."
        shapes = {
            "self_attn.q_proj.weight": [q, h], "self_attn.k_proj.weight": [kv, h],
            "self_attn.v_proj.weight": [kv, h], "self_attn.o_proj.weight": [h, q],
            "self_attn.q_norm.weight": [d], "self_attn.k_norm.weight": [d],
            "mlp.gate_proj.weight": [i, h], "mlp.up_proj.weight": [i, h],
            "mlp.down_proj.weight": [h, i], "input_layernorm.weight": [h],
            "post_attention_layernorm.weight": [h],
        }
        result.update({prefix + name: shape for name, shape in shapes.items()})
        if config.get("model_type", "").startswith("qwen3_5"):
            if config["layer_types"][lid] == "full_attention":
                result[prefix+"self_attn.q_proj.weight"] = [2*q,h]
            else:
                for name in list(result):
                    if name.startswith(prefix+"self_attn."):
                        del result[name]
                nk,nv,dk,dv = (config[k] for k in ("linear_num_key_heads","linear_num_value_heads","linear_key_head_dim","linear_value_head_dim"))
                cd = 2*nk*dk+nv*dv
                linear = {"in_proj_qkv.weight":[cd,h], "in_proj_z.weight":[nv*dv,h], "in_proj_b.weight":[nv,h], "in_proj_a.weight":[nv,h], "conv1d.weight":[cd,1,config["linear_conv_kernel_dim"]], "A_log":[nv], "dt_bias":[nv], "norm.weight":[dv], "out_proj.weight":[h,nv*dv]}
                result.update({prefix+"linear_attn."+name:shape for name,shape in linear.items()})
        if is_moe_layer(config, lid):
            for name in ("gate_proj", "up_proj", "down_proj"):
                result.pop(prefix + "mlp." + name + ".weight")
            experts, width = config["num_experts"], config["moe_intermediate_size"]
            result[prefix + "mlp.gate.weight"] = [experts, h]
            if config.get("expert_layouts", {}).get(str(lid)) == "fused":
                result[prefix + "mlp.experts.gate_up_proj.weight"] = [experts, 2 * width, h]
                result[prefix + "mlp.experts.down_proj.weight"] = [experts, h, width]
            else:
                for expert in range(experts):
                    for projection, shape in (("gate", [width, h]), ("up", [width, h]), ("down", [h, width])):
                        result[f"{prefix}mlp.experts.{expert}.{projection}_proj.weight"] = shape
            if config.get("model_type") == "qwen3_5_moe":
                shared = config["shared_expert_intermediate_size"]
                result.update({prefix+"mlp.shared_expert.gate_proj.weight":[shared,h],prefix+"mlp.shared_expert.up_proj.weight":[shared,h],prefix+"mlp.shared_expert.down_proj.weight":[h,shared],prefix+"mlp.shared_expert_gate.weight":[1,h]})
    return result


def weight_plan(model_dir, config, *, draft=False):
    """生成与设备布局无关的完整权重装载计划。"""
    found, entries, ignored = set(), [], []
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise ValueError("需要 safetensors 检查点；不加载 pickle/pytorch_model.bin")
    headers = {file: safetensors_header(file) for file in files}
    hybrid = config.get("model_type", "").startswith("qwen3_5")
    def canonical(name):
        if hybrid and name.startswith("model.language_model."):
            return name.removeprefix("model.language_model.")
        if hybrid and name.startswith("language_model."):
            return name.removeprefix("language_model.").removeprefix("model.")
        return name.removeprefix("model.")
    if config.get("model_type") in ("qwen3_moe", "qwen3_5_moe"):
        names = {canonical(name) for header in headers.values() for name in header}
        config["expert_layouts"] = {str(lid): "fused" if f"layers.{lid}.mlp.experts.gate_up_proj.weight" in names else "separate" for lid in range(config["num_hidden_layers"]) if is_moe_layer(config, lid)}
    expected = expected_weights(config, draft=draft)
    for file in files:
        for source, desc in headers[file].items():
            if hybrid and source.startswith(("model.visual.","visual.","model.mtp.","mtp.")):
                ignored.append({"file":file.name,"source":source,"reason":"仅部署 Qwen3.5 文本，跳过视觉/MTP"})
                continue
            name = canonical(source)
            if name in found:
                raise ValueError(f"重复权重：{name}")
            found.add(name)
            if config.get("quantization") and name.endswith((".weight_scale", ".weight_zero_point")):
                base = name.rsplit(".",1)[0]+".weight"
                if base not in expected or len(expected[base]) != 2 or desc["shape"] not in ([expected[base][0]], [expected[base][0],1]) or desc["dtype"] not in ("F16","BF16","F32","I8"):
                    raise ValueError(f"量化伴生参数形状/类型错误：{source}")
                entries.append({"file":file.name,"source":source,"destination":name,**desc})
                continue
            optional = {"embed_tokens.weight": [config["vocab_size"], config["hidden_size"]], "lm_head.weight": [config["vocab_size"], config["hidden_size"]]}
            if name not in expected:
                if name in optional and (draft or config["tie_word_embeddings"]):
                    if desc["shape"] != optional[name]:
                        raise ValueError(f"额外 embedding/head 形状错误：{source}")
                    # 草稿自己的 embedding 会覆盖共享 embedding，必须实际加载。
                    if draft:
                        entries.append({"file": file.name, "source": source, "destination": name, **desc})
                    else:
                        ignored.append({"file": file.name, "source": source, "reason": "使用目标输出头或绑定 embedding"})
                    continue
                if draft and name in ("d2t", "t2d"):
                    if config.get("spec_method") == "eagle3":
                        if name != "t2d" or desc["dtype"] != "BOOL" or desc["shape"] != [config["vocab_size"]]:
                            raise ValueError("非法 EAGLE3 t2d")
                        ignored.append({"file":file.name,"source":source,"reason":"greedy 推理仅用 d2t"})
                        continue
                    if name == "d2t":
                        with file.open("rb") as handle:
                            handle.seek(desc["offset"])
                            if any(handle.read(desc["bytes"])):
                                raise ValueError("首版不支持非恒等草稿词表映射")
                    ignored.append({"file": file.name, "source": source, "reason": "全词表恒等映射，不参与前向"})
                    continue
                raise ValueError(f"无法识别的权重：{source}")
            allowed = ("I64",) if name == "d2t" else ("F16", "BF16", "F32", "I8") if config.get("quantization") and len(expected[name]) == 2 and name.endswith(".weight") else ("F16", "BF16", "F32")
            if desc["shape"] != expected[name] or desc["dtype"] not in allowed:
                raise ValueError(f"权重类型/形状错误：{source}：{desc}")
            entries.append({"file": file.name, "source": source, "destination": name, **desc})
    missing = set(expected) - found
    if missing:
        raise ValueError(f"检查点缺失权重：{sorted(missing)}")
    indexed = {e["destination"]:e for e in entries}
    for name,e in indexed.items():
        if e["dtype"] == "I8" and name.endswith(".weight") and name.removesuffix(".weight")+".weight_scale" not in indexed:
            raise ValueError(f"int8 权重缺少 scale：{name}")
        if name.endswith((".weight_scale", ".weight_zero_point")):
            base = name.rsplit(".",1)[0]+".weight"
            if base not in indexed or indexed[base]["dtype"] != "I8":
                raise ValueError("量化伴生参数缺少 int8 权重")
    return {"entries": entries, "ignored": ignored}, files


def tokenizer_golden(tokenizer):
    """覆盖 Unicode、特殊标记、多轮及 thinking 两种语义。"""
    texts = ["", "Hello, world!", "你好，世界！", "αβ café e\u0301 😀👨‍👩‍👧‍👦", "  a\t\r\n\n123456789", "def f(x):\n    return x + 1", "<|im_start|>user\n测试<|im_end|>"]
    cases = [{"text": text, "token_ids": tokenizer.encode(text, add_special_tokens=False)} for text in texts]
    for case in cases:
        case["decoded"] = tokenizer.decode(case["token_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    chats = []
    for messages in ([{"role": "user", "content": "你好"}], [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"}, {"role": "user", "content": "为什么？😀"}]):
        for thinking in (True, False):
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            chats.append({"messages": messages, "enable_thinking": thinking, "rendered": rendered, "token_ids": tokenizer.encode(rendered, add_special_tokens=False)})
    return {"texts": cases, "chats": chats}


def export_bundle(model, output, *, draft=None, spec_num_tokens=None, hardlink=False, verifier=None):
    """生成新目录；任何失败均保留明确的未完成标识，不覆盖既有 bundle。"""
    from transformers import AutoTokenizer
    model, output = Path(model).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"输出目录已存在，拒绝覆盖：{output}")
    target = normalize_model(read_json(model / "config.json"))
    token_data = read_json(model / "tokenizer.json")
    if token_data.get("model", {}).get("type") != "BPE" or token_data.get("normalizer") not in (None, {"type": "NFC"}):
        raise ValueError("当前 tokenizer profile 仅支持无归一化或 NFC 的 byte-level BPE")
    if token_data["model"].get("dropout") is not None or token_data["model"].get("byte_fallback", False):
        raise ValueError("当前 profile 不支持 BPE dropout/byte_fallback")
    config = {"target": target, "draft": None}
    plans = {}
    sources = [("target", model, target)]
    if draft:
        draft = Path(draft).resolve()
        raw = read_json(draft / "config.json")
        dc = normalize_model(raw, draft=True)
        eagle = dc.get("spec_method") == "eagle3"
        if eagle:
            ids = raw.get("aux_hidden_state_layer_ids") or [2,target["num_hidden_layers"]//2,target["num_hidden_layers"]-3]
            k = 3 if spec_num_tokens is None else spec_num_tokens
            if len(ids) != 3 or len(set(ids)) != 3 or any(type(i) is not int or i < 0 or i >= target["num_hidden_layers"] for i in ids) or not 1 <= k <= 15:
                raise ValueError("EAGLE3 aux 层或 k 非法；小模型需显式指定三层")
            if dc["hidden_size"] != target["hidden_size"] or dc["vocab_size"] != target["vocab_size"] or target["model_type"].startswith("qwen3_5"):
                raise ValueError("EAGLE3 当前只接 Qwen3 dense/MoE 同宽目标")
            dc.update(aux_layers=ids,num_tokens=k)
            config["draft"] = dc
            sources.append(("draft",draft,dc))
        else:
            dc["spec_method"] = "dflash"
            configure_dflash(dc,raw,target,spec_num_tokens)
            config["draft"] = dc
            sources.append(("draft",draft,dc))
    # 在复制大权重前完成所有结构验证。
    for kind, directory, mc in sources:
        plans[kind] = weight_plan(directory, mc, draft=kind == "draft")
    return _write_bundle(model,output,target,config,sources,plans,token_data,AutoTokenizer,hardlink,verifier)


def configure_dflash(dc,raw,target,spec_num_tokens):
        df = raw.get("dflash_config") or {}
        dc.update(target_layer_ids=df.get("target_layer_ids"), mask_token_id=df.get("mask_token_id"), block_size=int(df.get("block_size", raw.get("block_size", 16))))
        if not dc["target_layer_ids"] or not isinstance(dc["mask_token_id"], int):
            raise ValueError("草稿必须显式提供 target_layer_ids/mask_token_id")
        if dc["hidden_size"] != target["hidden_size"] or dc["vocab_size"] != target["vocab_size"]:
            raise ValueError("草稿 hidden_size/vocab_size 与目标不一致")
        hybrid = target["model_type"].startswith("qwen3_5")
        if any(type(lid) is not int or lid < 0 or lid + (0 if hybrid else 1) >= target["num_hidden_layers"] for lid in dc["target_layer_ids"]) or len(set(dc["target_layer_ids"])) != len(dc["target_layer_ids"]):
            raise ValueError("草稿 aux 层号越界或重复")
        if not 0 <= dc["mask_token_id"] < target["vocab_size"]:
            raise ValueError("草稿 mask_token_id 越界")
        if raw.get("draft_vocab_size", target["vocab_size"]) != target["vocab_size"]:
            raise ValueError("首版不支持缩减草稿词表")
        k = dc["block_size"] - 1 if spec_num_tokens is None else spec_num_tokens
        if not 1 <= k < dc["block_size"]:
            raise ValueError("k 必须满足 1 <= k < block_size")
        dc["num_tokens"] = k


def _write_bundle(model,output,target,config,sources,plans,token_data,AutoTokenizer,hardlink,verifier):
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    golden = tokenizer_golden(tokenizer)
    output.mkdir(parents=True)
    write_json(output / "INCOMPLETE.json", {"reason": "尚未完成打包/原生校验"})
    write_json(output / "engine/config.json", config)
    weights = {}
    for kind, directory, _ in sources:
        plan, files = plans[kind]
        for entry in plan["entries"]:
            entry["file"] = f"weights/{kind}/{entry['file']}"
        weights[kind] = plan
        for file in files:
            dest = output / "weights" / kind / file.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if hardlink:
                os.link(file, dest)
            else:
                shutil.copyfile(file, dest)
        write_json(output / "provenance" / f"{kind}-config.json", read_json(directory / "config.json"))
    write_json(output / "engine/weights.plan.json", weights)
    write_json(output / "tokenizer/tokenizer.json", token_data)
    write_json(output / "tokenizer/runtime.json", {"eos_token_id": tokenizer.eos_token_id, "bos_token_id": tokenizer.bos_token_id, "vocab_size": target["vocab_size"], "skip_special_tokens": True, "add_special_tokens": False, "profile": "qwen-byte-bpe-v1"})
    (output / "tokenizer/chat_template.jinja").write_text(tokenizer.chat_template, encoding="utf-8")
    write_json(output / "golden/tokenizer_chat.json", golden)
    write_json(output / "provenance/oracle-source.json", source_state())
    profile = {"qwen3":"qwen3-dense-v1","qwen3_moe":"qwen3-moe-v1","qwen3_5":"qwen35-dense-v1","qwen3_5_moe":"qwen35-moe-v1"}[target["model_type"]]
    draft = config["draft"]
    manifest = {"format_version": BUNDLE_FORMAT, "contract_version": CONTRACT_VERSION, "oracle_git_sha": git_sha(), "profiles": [profile] + (["eagle3-greedy-v1" if draft.get("spec_method")=="eagle3" else "dflash1-greedy-v1"] if draft else []) + ([target["quantization"]] if target.get("quantization") else []), "files": {}}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "INCOMPLETE.json":
            manifest["files"][path.relative_to(output).as_posix()] = {"size": path.stat().st_size, "sha256": sha256(path)}
    write_json(output / "bundle.json", manifest)
    if verifier:
        subprocess.run([str(verifier), "verify", str(output)], check=True)
        write_json(output / "verification.json", {"native_verified": True, "bundle_sha256": sha256(output / "bundle.json")})
    (output / "INCOMPLETE.json").unlink()
    return manifest


def publish_contracts(source, destination):
    """发布字节可重复的契约 tar.gz，拒绝覆盖既有产物。"""
    source, destination = Path(source), Path(destination)
    with destination.open("xb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in sorted(source.rglob("*")):
                    if not path.is_file() or path.is_symlink():
                        continue
                    info = tarfile.TarInfo(path.relative_to(source).as_posix())
                    info.size, info.mode = path.stat().st_size, 0o644
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    return sha256(destination)
