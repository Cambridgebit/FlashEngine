"""PEFT LoRA 适配器离线融合：基座 checkpoint + adapter → 独立模型目录。

产物与基座同构（HF state_dict 命名、safetensors 分片 + index），可被 transformers
与 FlashEngine 直接加载，行为等价于引擎的 ``--lora-module`` 运行时路径：
``W_new = W + scaling * (B @ A)``，scaling = alpha / r（rsLoRA 时为 alpha / sqrt(r)）。

纯 CPU、不占 NPU，可作为模块运行（容器内用 ``/data2/fe-t5-venv/bin/python3``）::

    # rank/alpha/scaling 全部从 adapter_config.json 读取，dtype 从基座 config.json 推断，
    # 写出后自动抽样重算校验——只需给基座、适配器、输出目录
    python -m scripts.merge_lora --base /data/Qwen3.5-9B \\
        --adapter /data/lora/Qwen3.5-9B-lora-642 \\
        --out /data/Qwen3.5-9B-lora642-fp16

    # 310P 部署需要 fp16 时显式指定；--verify-tensors 可为 all / 整数 / off
    python -m scripts.merge_lora --base /data/Qwen3.5-9B \\
        --adapter /data/lora/Qwen3.5-9B-lora-642 \\
        --out /data/Qwen3.5-9B-lora642-fp16 --dtype float16 --verify-tensors all

    # 批量：<adapters-dir> 下每个含 adapter_model.safetensors 的子目录各产出一份
    python -m scripts.merge_lora --base /data/Qwen3.5-9B \\
        --adapters-dir /data/lora/adapters --all --out /data/merged

    # 只审适配器（不需要基座）：报模块模式/缩放/是否含逐专家专家权重
    python -m scripts.merge_lora --adapter /data/lora/Qwen3.5-35B-A3B --audit-adapter

适配器里的模块名是 PEFT 的 HF 路径（``base_model.model.model.language_model.layers.N...``），
剥掉 ``base_model.model.`` 后与基座索引键同名；仅当基座用了不同的多模态外壳前缀时
才按候选前缀回退匹配。``in_proj_qkv`` 在基座里是单个张量，B 的行已覆盖 q/k/v 全序，
因此整张量相加即可，不需要按引擎融合模块（``in_proj_qkvz``/``in_proj_ba``）拆片。

dense（Qwen3.5-9B）与 MoE（Qwen3.5-35B-A3B，``mlp.shared_expert.*`` /
``mlp.shared_expert_gate``）走同一条路径——这两类都是普通 Linear；若适配器还按
``mlp.experts.<i>.<proj>`` 逐专家训练（Qwen 系列检查点就是逐专家存权重），也能按
同名键直接融合；只有基座把专家堆叠成融合张量时才会因形状不一致而明确报错。
"""

import argparse
import json
import os
import re
import shutil
import sys

import torch

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# 与 flashengine/lora/loader.py 同口径：这些模块不属于可融合的 Linear 权重。
SKIP_TOKENS = ("visual.", "mtp.", "conv1d", "lm_head", "embed_tokens")

ADAPTER_WEIGHT_FILE = "adapter_model.safetensors"
ADAPTER_CONFIG_FILE = "adapter_config.json"


class MergeError(RuntimeError):
    """融合过程中的不可恢复错误（输入不满足约定）。"""


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _strip_adapter_prefix(name: str) -> str:
    if name.startswith("base_model.model."):
        return name[len("base_model.model."):]
    return name


def _candidate_prefixes(module: str) -> list[str]:
    """候选基座键前缀（按优先级）：适配器名可能带/不带多模态外壳。"""
    out = [module]
    for shell in ("model.language_model.", "language_model.", "model."):
        if module.startswith(shell):
            remainder = module[len(shell):]
            for alt in ("model.language_model.", "language_model.", "model.", ""):
                cand = alt + remainder
                if cand not in out:
                    out.append(cand)
    return out


def _is_skipped(module: str) -> bool:
    return any(token in module for token in SKIP_TOKENS)


def read_adapter_meta(path: str) -> dict:
    cfg_path = os.path.join(path, ADAPTER_CONFIG_FILE)
    if not os.path.isfile(cfg_path):
        raise MergeError(f"适配器缺少 {ADAPTER_CONFIG_FILE}: {path}")
    cfg = _read_json(cfg_path)
    peft_type = str(cfg.get("peft_type", "LORA")).upper()
    if peft_type != "LORA":
        raise MergeError(f"只支持 peft_type=LORA，实际为 {peft_type}")
    if cfg.get("use_dora"):
        raise MergeError("不支持 DoRA（use_dora=true）：需要合并 magnitude 向量，本脚本不近似")
    bias = str(cfg.get("bias", "none")).lower()
    if bias != "none":
        raise MergeError(f"不支持 bias={bias}：适配器含需合并的 bias 张量")
    if cfg.get("modules_to_save"):
        raise MergeError(
            "不支持 modules_to_save="
            + repr(cfg.get("modules_to_save"))
            + "：这些模块是整权重替换而非低秩增量"
        )
    if cfg.get("target_parameters"):
        raise MergeError(
            "不支持 target_parameters="
            + repr(cfg.get("target_parameters"))
            + "：非 Linear 权重上的 LoRA 需要专门的合并路径"
        )
    if cfg.get("fan_in_fan_out"):
        raise MergeError(
            "不支持 fan_in_fan_out=true：Conv1D 风格权重以 [in, out] 存储，"
            "delta 必须转置后才能相加"
        )
    rank = int(cfg.get("r", 8))
    alpha_value = cfg.get("lora_alpha", rank)
    try:
        alpha = float(alpha_value)
    except (TypeError, ValueError) as exc:
        raise MergeError(f"lora_alpha 不是数值: {alpha_value!r}") from exc
    return {
        "path": path,
        "rank": rank,
        "alpha": alpha,
        "use_rslora": bool(cfg.get("use_rslora", False)),
        "rank_pattern": dict(cfg.get("rank_pattern") or {}),
        "alpha_pattern": dict(cfg.get("alpha_pattern") or {}),
        "target_modules": cfg.get("target_modules"),
        "base_model_name_or_path": cfg.get("base_model_name_or_path"),
    }


def _pattern_value(patterns: dict, module: str, fallback):
    for pattern, value in patterns.items():
        if re.fullmatch(pattern, module) or re.search(pattern, module):
            return value
    return fallback


def detect_base_dtype(base_dir: str) -> tuple[str, str]:
    """按基座 config.json 推断权重 dtype，返回 ``(dtype 名, 来源说明)``。"""
    payload = {}
    config_path = os.path.join(base_dir, "config.json")
    if os.path.isfile(config_path):
        payload = _read_json(config_path)
    scopes = [
        ("config.json", payload),
        ("config.json text_config", payload.get("text_config") or {}),
        ("config.json llm_config", payload.get("llm_config") or {}),
        ("config.json language_config", payload.get("language_config") or {}),
    ]
    for where, scope in scopes:
        for key in ("dtype", "torch_dtype"):
            value = scope.get(key)
            if not isinstance(value, str):
                continue
            name = value.strip().replace("torch.", "")
            if name in DTYPES:
                return name, f"{where} 的 {key}={value}"
    return "bfloat16", "config.json 未声明 dtype，按 bfloat16 兜底"


_PATTERN_NUM = re.compile(r"\.\d+\.")


def module_pattern(module: str) -> str:
    """把 ``layers.12.mlp`` 归一化成 ``layers.N.mlp``，用于按模式去重。"""
    return _PATTERN_NUM.sub(".N.", module)


def _sample_keys(keys: list[str], planned: dict, count: int) -> list[str]:
    """挑校验用张量：先覆盖每种模块模式，再按顺序均匀取样。"""
    ordered = sorted(keys)
    if count >= len(ordered):
        return ordered
    seeds: list[str] = []
    seen = set()
    for key in ordered:
        pattern = module_pattern(planned[key][0])
        if pattern in seen:
            continue
        seen.add(pattern)
        seeds.append(key)
        if len(seeds) >= count:
            return seeds
    picked = set(seeds)
    remaining = [key for key in ordered if key not in picked]
    need = count - len(seeds)
    if need > 0 and remaining:
        step = max(1, len(remaining) // need)
        seeds.extend(remaining[::step][:need])
    return seeds


def resolve_verify_spec(spec, keys: list[str], planned: dict) -> list[str]:
    """把 ``--verify-tensors`` 解析成待校验键列表；默认 auto = 32 个（含每种模式）。"""
    text = str(spec).strip().lower()
    if text in ("0", "off", "no", "none", "false"):
        return []
    if text == "all":
        return sorted(keys)
    if text in ("auto", ""):
        return _sample_keys(keys, planned, min(32, len(keys)))
    try:
        count = int(text)
    except ValueError as exc:
        raise MergeError(
            f"--verify-tensors 取值非法: {spec}（可用 auto / all / off / 正整数）") from exc
    if count <= 0:
        return []
    return _sample_keys(keys, planned, count)


def adapter_scaling(meta: dict, module: str) -> float:
    rank = float(_pattern_value(meta["rank_pattern"], module, meta["rank"]))
    alpha = float(_pattern_value(meta["alpha_pattern"], module, meta["alpha"]))
    if rank <= 0:
        raise MergeError(f"{module}: rank={rank} 非法")
    if meta["use_rslora"]:
        return alpha / (rank ** 0.5)
    return alpha / rank


def load_adapter_tensors(path: str):
    """读适配器，返回 ({模块名: (A[r,in], B[out,r])}, 跳过列表)；张量保持 fp32。"""
    from safetensors import safe_open

    weight_file = os.path.join(path, ADAPTER_WEIGHT_FILE)
    if not os.path.isfile(weight_file):
        raise MergeError(f"适配器缺少 {ADAPTER_WEIGHT_FILE}: {path}")
    a_map: dict[str, torch.Tensor] = {}
    b_map: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    unexpected: list[str] = []
    with safe_open(weight_file, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.endswith(".lora_A.weight"):
                module = _strip_adapter_prefix(key[: -len(".lora_A.weight")])
                if _is_skipped(module):
                    skipped.append(module)
                    continue
                a_map[module] = handle.get_tensor(key).to(torch.float32)
            elif key.endswith(".lora_B.weight"):
                module = _strip_adapter_prefix(key[: -len(".lora_B.weight")])
                if _is_skipped(module):
                    continue
                b_map[module] = handle.get_tensor(key).to(torch.float32)
            else:
                unexpected.append(key)
    if unexpected:
        raise MergeError(
            "适配器含非 lora_A/lora_B 张量（DoRA/偏置/嵌入 LoRA 等），本脚本不处理："
            + ", ".join(sorted(unexpected)[:8])
        )
    missing_b = sorted(set(a_map) - set(b_map))
    if missing_b:
        raise MergeError("以下模块只有 lora_A 没有 lora_B：" + ", ".join(missing_b[:8]))
    missing_a = sorted(set(b_map) - set(a_map))
    if missing_a:
        raise MergeError("以下模块只有 lora_B 没有 lora_A：" + ", ".join(missing_a[:8]))
    if not a_map:
        raise MergeError(f"适配器 {path} 里没有任何可融合的 LoRA 权重")
    return {module: (a_map[module], b_map[module]) for module in sorted(a_map)}, sorted(set(skipped))


class BaseCheckpoint:
    """基座 safetensors 分片视图（支持 index.json 与单文件两种布局）。"""

    def __init__(self, path: str):
        self.path = path
        index_path = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            index = _read_json(index_path)
            weight_map = dict(index["weight_map"])
            shards = sorted(set(weight_map.values()))
            self.total_bytes = int(index.get("metadata", {}).get("total_size", 0) or 0)
        else:
            single = os.path.join(path, "model.safetensors")
            if not os.path.isfile(single):
                raise MergeError(f"基座目录既无 index.json 也无 model.safetensors: {path}")
            from safetensors import safe_open

            with safe_open(single, framework="pt", device="cpu") as handle:
                weight_map = {key: "model.safetensors" for key in handle.keys()}
            shards = ["model.safetensors"]
            self.total_bytes = 0
        self.weight_map = weight_map
        self.shard_files = shards
        self.keys = set(weight_map)

    def key_bytes(self, key: str) -> int:
        return 0  # 仅在无 index 元数据时用于粗估，见 estimate_bytes()

    def estimate_bytes(self) -> int:
        if self.total_bytes:
            return self.total_bytes
        from safetensors import safe_open

        total = 0
        for shard in self.shard_files:
            with safe_open(os.path.join(self.path, shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    count = 1
                    for dim in handle.get_slice(key).get_shape():
                        count *= int(dim)
                    total += count * 2
        self.total_bytes = total
        return total

    def resolve(self, module: str) -> str | None:
        for candidate in _candidate_prefixes(module):
            key = candidate + ".weight"
            if key in self.keys:
                return key
        return None

    def open_shard(self, shard: str):
        from safetensors import safe_open

        return safe_open(os.path.join(self.path, shard), framework="pt", device="cpu")


def plan_merges(base: BaseCheckpoint, tensors: dict, meta: dict):
    """建立 {基座张量名: (模块名, scaling)}，并对每个模块校验形状。"""
    planned: dict[str, tuple[str, float]] = {}
    unresolved: list[str] = []
    for module in tensors:
        key = base.resolve(module)
        if key is None:
            unresolved.append(module)
            continue
        if key in planned:
            raise MergeError(
                f"两个适配器模块映射到同一基座张量 {key}: {planned[key][0]} / {module}"
            )
        planned[key] = (module, adapter_scaling(meta, module))
    if unresolved:
        raise MergeError(
            "以下适配器模块在基座里找不到对应权重（拒绝部分融合）："
            + ", ".join(unresolved[:8])
        )
    shapes: dict[str, tuple[int, ...]] = {}
    for key, (module, _scale) in planned.items():
        a, b = tensors[module]
        shapes[key] = (b.shape[0], a.shape[1])
    return planned, shapes


def _merge_delta(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.mm(b, a) * scale


def _describe(value: float) -> str:
    return f"{value:.6g}"


def merge_checkpoint(base: BaseCheckpoint, tensors: dict, meta: dict, planned: dict,
                     shapes: dict, out_dir: str, dtype_name: str, shard_gb: float,
                     overwrite: bool) -> dict:
    from safetensors.torch import save_file

    dtype = DTYPES[dtype_name]
    shard_bytes = int(shard_gb * (1 << 30))
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not overwrite:
        raise MergeError(f"输出目录非空（加 --overwrite 覆盖）: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # 逐分片流式写出：先写临时名，全部完成后再按最终分片数重命名。
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0
    written: list[tuple[str, list[str], int]] = []
    merged_keys: list[str] = []
    written_keys: set[str] = set()

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        name = f"merge-tmp-{len(written):05d}.safetensors"
        save_file(pending, os.path.join(out_dir, name), metadata={"format": "pt"})
        written.append((name, list(pending), pending_bytes))
        pending = {}
        pending_bytes = 0

    for shard in base.shard_files:
        handle = base.open_shard(shard)
        for key in sorted(k for k in handle.keys() if k not in written_keys):
            weight = handle.get_tensor(key)
            if key in planned:
                module, scale = planned[key]
                a, b = tensors[module]
                expected = shapes[key]
                if tuple(weight.shape) != expected:
                    hint = ""
                    if ".experts." in key:
                        hint = ("；这是 MoE 专家权重，若基座把专家堆叠成融合张量"
                                "（如 [num_experts, out, in]），需要逐专家导出的适配器")
                    raise MergeError(
                        f"{key} 形状 {tuple(weight.shape)} 与 B@A {expected} 不一致"
                        f"（模块 {module}）{hint}"
                    )
                delta = _merge_delta(a, b, scale)
                weight = (weight.to(torch.float32) + delta).to(dtype)
                merged_keys.append(key)
            elif weight.dtype != dtype:
                # 未融合的张量也统一到目标 dtype：产物是单 dtype 模型，避免
                # 「语言层 fp16 + 视觉塔 bf16」的混合产物在设备侧二次转换。
                weight = weight.to(dtype)
            pending[key] = weight
            written_keys.add(key)
            pending_bytes += weight.numel() * weight.element_size()
            if pending_bytes >= shard_bytes:
                flush()
        del handle
    flush()

    missing = sorted(set(planned) - set(merged_keys))
    if missing:
        raise MergeError("以下张量在基座分片里没有出现（索引与文件不一致）：" + ", ".join(missing[:8]))

    total = len(written)
    weight_map: dict[str, str] = {}
    total_bytes = 0
    for index, (temp_name, keys, nbytes) in enumerate(written, start=1):
        final = f"model.safetensors-{index:05d}-of-{total:05d}.safetensors"
        os.replace(os.path.join(out_dir, temp_name), os.path.join(out_dir, final))
        for key in keys:
            weight_map[key] = final
        total_bytes += nbytes
    _write_json(
        os.path.join(out_dir, "model.safetensors.index.json"),
        {"metadata": {"total_size": total_bytes}, "weight_map": dict(sorted(weight_map.items()))},
    )
    _copy_auxiliary_files(base.path, out_dir, dtype_name)
    return {
        "out": out_dir,
        "shards": total,
        "bytes": total_bytes,
        "merged": len(merged_keys),
        "keys": len(weight_map),
    }


def _copy_auxiliary_files(src_dir: str, out_dir: str, dtype_name: str) -> None:
    for name in sorted(os.listdir(src_dir)):
        source = os.path.join(src_dir, name)
        if not os.path.isfile(source):
            continue
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        target = os.path.join(out_dir, name)
        if name.endswith(".json") and name in ("config.json", "generation_config.json"):
            payload = _read_json(source)
            if isinstance(payload, dict):
                for container_key in ("text_config",):
                    container = payload.get(container_key)
                    if isinstance(container, dict):
                        for dtype_key in ("dtype", "torch_dtype"):
                            if dtype_key in container:
                                container[dtype_key] = dtype_name
                for dtype_key in ("dtype", "torch_dtype"):
                    if dtype_key in payload:
                        payload[dtype_key] = dtype_name
            _write_json(target, payload)
            continue
        shutil.copy2(source, target)


def _disk_guard(out_dir: str, needed: int) -> None:
    parent = os.path.abspath(os.path.dirname(out_dir.rstrip(os.sep)) or ".")
    probe = parent
    while not os.path.isdir(probe):
        nxt = os.path.dirname(probe)
        if nxt == probe:
            break
        probe = nxt
    free = shutil.disk_usage(probe).free
    # 输出与基座同量级，另需 5% 余量给 index/辅助文件与文件系统开销。
    requirement = int(needed * 1.05)
    if free < requirement:
        raise MergeError(
            f"磁盘空间不足：需要约 {requirement / (1 << 30):.1f} GiB，"
            f"{probe} 可用 {free / (1 << 30):.1f} GiB"
        )


def verify_tensors(base: BaseCheckpoint, tensors: dict, planned: dict, out_dir: str,
                   picks: list[str], dtype_name: str) -> float:
    """回读抽样张量，重算 base + delta 并与写出值比对，返回最大绝对误差。"""
    from safetensors import safe_open

    out_index = _read_json(os.path.join(out_dir, "model.safetensors.index.json"))
    weight_map = out_index["weight_map"]
    keys = [key for key in picks if key in weight_map]
    if not keys:
        raise MergeError("抽样校验失败：输出索引里没有待校验张量")
    atol = {"float16": 5e-3, "bfloat16": 5e-2, "float32": 1e-5}[dtype_name]
    worst = 0.0
    for key in keys:
        module, scale = planned[key]
        a, b = tensors[module]
        base_handle = base.open_shard(base.weight_map[key])
        reference = (base_handle.get_tensor(key).to(torch.float32)
                     + _merge_delta(a, b, scale))
        del base_handle
        merged_handle = safe_open(os.path.join(out_dir, weight_map[key]),
                                  framework="pt", device="cpu")
        actual = merged_handle.get_tensor(key).to(torch.float32)
        del merged_handle
        error = float((reference - actual).abs().max().item())
        worst = max(worst, error)
        if error > atol:
            raise MergeError(
                f"抽样校验失败 {key}: 最大绝对误差 {error:.3e} > {atol:.1e}"
                f"（模块 {module}，scaling {_describe(scale)}）"
            )
    return worst


def _adapter_dirs(root: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for name in sorted(os.listdir(root)):
        candidate = os.path.join(root, name)
        if os.path.isfile(os.path.join(candidate, ADAPTER_WEIGHT_FILE)):
            found.append((name, candidate))
    return found


def audit_adapter(adapter_dir: str) -> dict:
    """不需要基座的自检：报模块模式、缩放、参数量与不支持项，并给出结论。

    覆盖 dense（如 Qwen3.5-9B）与 MoE（如 Qwen3.5-35B-A3B，含 shared_expert /
    shared_expert_gate；若适配器按 ``mlp.experts.<i>.<proj>`` 逐专家训练也能识别）。
    """
    meta = read_adapter_meta(adapter_dir)
    tensors, skipped = load_adapter_tensors(adapter_dir)
    pattern_counts: dict[str, int] = {}
    params = 0
    scaling_by_pattern: dict[str, str] = {}
    moe_modules: list[str] = []
    for module in tensors:
        a, b = tensors[module]
        params += a.numel() + b.numel()
        pattern = re.sub(r"layers\.\d+\.", "layers.N.", module)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        scaling_by_pattern[pattern] = _describe(adapter_scaling(meta, module))
        if ".mlp.experts." in module:
            moe_modules.append(module)
    rank_bits = sorted({int(tensors[module][0].shape[0]) for module in tensors})
    print(
        f"[merge_lora][audit] adapter={os.path.basename(adapter_dir.rstrip('/'))} "
        f"peft=LORA r={meta['rank']} alpha={_describe(meta['alpha'])} "
        f"rslora={meta['use_rslora']} modules={len(tensors)} "
        f"delta_params={params / 1e6:.1f}M rank_in_tensors={rank_bits}",
        flush=True,
    )
    print("[merge_lora][audit] 模块模式（layers.N 归一化）:", flush=True)
    for pattern in sorted(pattern_counts):
        print(
            f"    {pattern_counts[pattern]:4d}  {pattern}  scaling={scaling_by_pattern[pattern]}",
            flush=True,
        )
    if skipped:
        print("[merge_lora][audit] 跳过（非 Linear 权重）: " + ", ".join(skipped[:8]), flush=True)
    has_shared_expert = any("shared_expert" in pattern for pattern in pattern_counts)
    if moe_modules:
        print(
            f"[merge_lora][audit] 含逐专家专家权重 {len(moe_modules)} 个"
            "（要求基座也是逐专家命名 mlp.experts.<i>.<proj>.weight）",
            flush=True,
        )
    elif has_shared_expert:
        print(
            "[merge_lora][audit] MoE 结构：只命中 mlp.shared_expert.* / mlp.shared_expert_gate"
            "（普通 Linear），未命中路由专家集群 → MoE 基座可直接融合",
            flush=True,
        )
    else:
        print(
            "[merge_lora][audit] 纯 dense 结构（无 MoE 专家权重）→ dense 基座直接融合",
            flush=True,
        )
    unsupported = sorted(
        {pattern for pattern in pattern_counts if _is_skipped(pattern.replace("layers.N.", "0."))}
    )
    verdict = "OK" if not unsupported else "有跳过项"
    print(
        f"[merge_lora][audit] 结论: {verdict}；基座要求 = 与模块同名的 "
        "<module>.weight，形状 [out, in]",
        flush=True,
    )
    return {
        "adapter": adapter_dir,
        "modules": len(tensors),
        "patterns": len(pattern_counts),
        "delta_params": params,
        "moe_expert_modules": len(moe_modules),
        "skipped": skipped,
        "verdict": verdict,
    }


def merge_one(base: BaseCheckpoint, adapter_dir: str, out_dir: str, args) -> dict:
    meta = read_adapter_meta(adapter_dir)
    if meta["base_model_name_or_path"] and os.path.basename(
            meta["base_model_name_or_path"].rstrip("/")) not in os.path.basename(
            base.path.rstrip("/")):
        print(
            "[merge_lora][WARN] 适配器的 base_model_name_or_path 与 --base 名称不同："
            f"{meta['base_model_name_or_path']} vs {base.path}",
            flush=True,
        )
    tensors, skipped = load_adapter_tensors(adapter_dir)
    planned, shapes = plan_merges(base, tensors, meta)
    scales = sorted({_describe(planned[key][1]) for key in planned})
    print(
        f"[merge_lora] adapter={os.path.basename(adapter_dir.rstrip('/'))} "
        f"r={meta['rank']} alpha={_describe(meta['alpha'])} "
        f"rslora={meta['use_rslora']} modules={len(tensors)} "
        f"targets={len(planned)} skipped={len(skipped)} scaling={scales}",
        flush=True,
    )
    print(
        f"[merge_lora] W += scaling·(B@A)，scaling = alpha/r = "
        f"{_describe(meta['alpha'])}/{meta['rank']} = {_describe(meta['alpha'] / meta['rank'])}"
        + ("（rsLoRA：alpha/sqrt(r)）" if meta["use_rslora"] else "")
        + f"；dtype={args.dtype}",
        flush=True,
    )
    verify_picks = resolve_verify_spec(args.verify_tensors, list(planned), planned)
    if verify_picks:
        covered = len({module_pattern(planned[key][0]) for key in verify_picks})
        scope = "全部融合张量" if len(verify_picks) == len(planned) else str(len(verify_picks)) + " 个张量"
        print(f"[merge_lora] 校验计划: {scope}（覆盖 {covered} 种模块模式）", flush=True)
    else:
        print("[merge_lora] 校验计划: 关闭", flush=True)
    if skipped:
        print("[merge_lora] 跳过（非 Linear 权重）: " + ", ".join(skipped[:6]), flush=True)
    if args.check:
        for key, (module, scale) in list(planned.items())[:5]:
            print(f"[merge_lora][check] {key} <- {module} scaling={_describe(scale)}",
                  flush=True)
        needed = base.estimate_bytes()
        print(f"[merge_lora][check] 预计输出 ≈ {needed / (1 << 30):.1f} GiB", flush=True)
        return {"out": out_dir, "merged": len(planned), "checked": True}
    _disk_guard(out_dir, base.estimate_bytes())
    info = merge_checkpoint(
        base, tensors, meta, planned, shapes, out_dir,
        args.dtype, args.shard_size_gb, args.overwrite,
    )
    print(
        f"[merge_lora] 写出 {info['out']}: {info['shards']} 个分片 "
        f"{info['bytes'] / (1 << 30):.2f} GiB, 融合 {info['merged']} 个张量, "
        f"总计 {info['keys']} 个张量",
        flush=True,
    )
    worst = None
    if verify_picks:
        worst = verify_tensors(base, tensors, planned, out_dir, verify_picks, args.dtype)
        print(
            f"[merge_lora] 校验通过（{len(verify_picks)} 个张量重算比对），"
            f"最大绝对误差 {worst:.3e}",
            flush=True,
        )
    info.update({
        "rank": meta["rank"], "alpha": meta["alpha"], "dtype": args.dtype,
        "verified": len(verify_picks), "max_abs_error": worst,
    })
    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.merge_lora",
        description="把 PEFT LoRA 适配器永久融合进基座 checkpoint，产出独立模型目录",
    )
    parser.add_argument("--base", help="基座模型目录（safetensors 分片或单文件）；--audit-adapter 时不需要")
    parser.add_argument("--adapter", help="单个 PEFT 适配器目录")
    parser.add_argument("--adapters-dir", help="批量模式：包含多个适配器子目录的根目录")
    parser.add_argument("--all", action="store_true", help="配合 --adapters-dir 逐个融合")
    parser.add_argument("--out",
                        help="输出目录（批量模式为输出根目录）；省略则用 <适配器目录>-merged")
    parser.add_argument("--audit-adapter", action="store_true",
                        help="只审适配器（模块模式/缩放/MoE 情况），不需要基座与输出目录")
    parser.add_argument("--dtype", choices=["auto"] + sorted(DTYPES), default="auto",
                        help="输出全部权重的 dtype；auto 按基座 config.json 推断（310P 部署可显式传 float16）")
    parser.add_argument("--shard-size-gb", type=float, default=4.0, help="输出分片大小（GiB）")
    parser.add_argument("--check", action="store_true", help="只做映射与估值，不写权重")
    parser.add_argument("--verify-tensors", default="auto",
                        help="写出后重算校验：auto（默认，32 个且覆盖全部模块模式）/ all / off / 正整数")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖非空输出目录")
    parser.add_argument("--keep-going", action="store_true", help="批量模式单个失败后继续")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.adapter and not (args.adapters_dir and args.all):
        parser.error("需要 --adapter，或 --adapters-dir 配合 --all")
    if args.adapter and (args.adapters_dir or args.all):
        parser.error("--adapter 与 --adapters-dir/--all 不能同时使用")
    if not args.audit_adapter and not args.base:
        parser.error("融合需要 --base（只审适配器请用 --audit-adapter）")
    if not args.audit_adapter and not args.out:
        if args.adapter:
            args.out = os.path.abspath(args.adapter.rstrip("/")) + "-merged"
        else:
            args.out = os.path.join(os.path.abspath(args.adapters_dir), "merged")
        print(f"[merge_lora] 未指定 --out，使用 {args.out}", flush=True)
    if not args.audit_adapter and args.dtype == "auto":
        args.dtype, dtype_source = detect_base_dtype(args.base)
        print(f"[merge_lora] dtype: auto -> {args.dtype}（{dtype_source}）", flush=True)
    try:
        if args.audit_adapter:
            if args.adapter:
                print(json.dumps(audit_adapter(args.adapter), ensure_ascii=False), flush=True)
                return 0
            pairs = _adapter_dirs(args.adapters_dir)
            if not pairs:
                raise MergeError(
                    f"{args.adapters_dir} 下没有找到含 {ADAPTER_WEIGHT_FILE} 的子目录")
            for _name, adapter_dir in pairs:
                print(json.dumps(audit_adapter(adapter_dir), ensure_ascii=False), flush=True)
            return 0
        base = BaseCheckpoint(args.base)
        if args.adapter:
            info = merge_one(base, args.adapter, args.out, args)
            print(json.dumps(info, ensure_ascii=False), flush=True)
            return 0
        pairs = _adapter_dirs(args.adapters_dir)
        if not pairs:
            raise MergeError(f"{args.adapters_dir} 下没有找到含 {ADAPTER_WEIGHT_FILE} 的子目录")
        failures = 0
        for name, adapter_dir in pairs:
            out_dir = os.path.join(args.out, name)
            try:
                info = merge_one(base, adapter_dir, out_dir, args)
                print(f"[merge_lora] {name}: {json.dumps(info, ensure_ascii=False)}", flush=True)
            except MergeError as exc:
                failures += 1
                print(f"[merge_lora][ERROR] {name}: {exc}", file=sys.stderr, flush=True)
                if not args.keep_going:
                    return 1
        return 0 if failures == 0 else 1
    except MergeError as exc:
        print(f"[merge_lora][ERROR] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
