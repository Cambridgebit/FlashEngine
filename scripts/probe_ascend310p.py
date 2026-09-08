"""Ascend 310P 板端逐算子数值契约 + 基础设施探测（ascend310p-backend 计划任务 8）。

定位：在 310P 板端跑一次，产出 JSON 报告——引擎 attention_impl="auto" 档据此裁决
（recommended_attention_impl），图三档链 / spec 解锁候选与基础设施结论也以本报告为据。
本脚本独立自包含：只 import 标准库 / torch / torch_npu，不 import nanovllm 引擎代码。

数值契约组（每项：小合成形状 vs CPU float32 参考，rtol=atol=0.05 板端实证口径）：
- fia_v2_tnd_prefill       npu_fused_infer_attention_score_v2，TND varlen + bool 因果掩码
                           （参数组对照 nanovllm/backends/ascend.py prefill_attention eager 分支）
- fia_v2_bnsd_paged_decode 同算子 BNSD + block_table + block_size + actual_seq_kvlen
                           （参数组对照 ascend.py decode_attention）
- reshape_and_cache        _npu_reshape_and_cache 写入后读回（含 slot 扁平语义与 -1 跳负槽）
- flash_attention_nz       _npu_flash_attention（私有算子）+ NZ 因果掩码——复刻参考实现
                           probe_310p_native_attention.py 的板端实证形态（flash_incre prefill 主路径）
- paged_attention          _npu_paged_attention（UNVERIFIED 候选，kwarg 组 = vllm-ascend 310P 官方形态）
- incre_flash_attention    npu_incre_flash_attention 逐序列 BNSD——复刻参考实现 decode 兜底的
                           板端实证形态（nano_vllm_310p_compat.py:245-296，flash_incre decode 主路径）
- rms_norm / add_rms_norm / swiglu   三个融合归一化/激活算子的返回值解包契约

spec verify 解锁候选组（计划 Scope #8，T13；裁决序 A→B→C，首个数值契约 PASS 者写入
报告顶层 spec_unlock_candidate，全 FAIL/absent 记 null——supports_spec 翻转（T14）的
前置证据键。每序列 q 长度均质 rows=k+1（EAGLE3 verify 语义，对照 model_runner._sg_verify：
行 j 可见 key = 位置 0..l-1+j，单序列 key 总数 = l+k），夹具刻意取批内 key 长度不齐 [7,5]）：
- spec_verify_fia_bnsd_paged  候选 A：npu_fused_infer_attention_score_v2 BNSD paged 多 q
                              （q [bs,H,rows,D] + 整池 4D 直读 + block_table +
                              actual_seq_kvlen + verify 因果掩码 [rows, W]，行 j 遮蔽位
                              c ≥ W-(k-j) 右对齐窗口语义——与 ascend.py
                              _paged_verify_fia / _verify_causal_mask_rows 同构）
- spec_verify_paged_attention 候选 B：_npu_paged_attention q_len>1 形态（无掩码入参，
                              可见性依赖算子内建因果，预期 bottom-right 对齐=verify 语义；
                              q 形态两连试 [bs,rows,H,D] → [bs,H,rows,D]，对照
                              _check_paged 先例扩展多 q）
- spec_verify_gather_flash    候选 C：全量 gather（_gather_kv_history 同款）+
                              _npu_flash_attention 不等长 q/kv + 自定义掩码（行 j 遮蔽位
                              c ≥ l+j；掩码形态两连试 ND fp16 精确尺寸 → NZ 16 对齐
                              padding，与 ascend.py _paged_verify_gather_flash 同构）

基础设施组：npugraph（NPUGraph 存在性 + pool kwarg TypeError 探测 + 最小捕获回放）、
torchair（torch_npu.dynamo → torchair import，图三档链第二档前提）、
pin_memory（分配 + non_blocking H2D 往返）、memory_stats_keys（引擎显存统计依赖的键名）、
toolchain（torch_npu / CANN / torch 版本）。

JSON 报告契约（字段名/取值域定死，消费端按此解析，不得擅改）：
{
  "toolchain": {"torch_npu": "...", "cann": "...", "torch": "..."},
  "repo_hash": "<git rev-parse --short HEAD，失败记 unknown>",
  "recommended_attention_impl": "fia_v2" | "flash_incre",
      # fia_v2 仅当 fia_v2_tnd_prefill 与 fia_v2_bnsd_paged_decode 双 PASS，否则 flash_incre
  "spec_unlock_candidate": "A" | "B" | "C" | null,
      # spec 解锁候选（裁决序 A→B→C，首个 PASS 者；全 FAIL/absent 记 null）——
      # 引擎 T14 翻转 supports_spec 的前置证据键
  "checks": {"<name>": {"status": "PASS" | "FAIL" | "absent", "detail": "..."}}
}
检查名固定为：fia_v2_tnd_prefill / fia_v2_bnsd_paged_decode / reshape_and_cache /
flash_attention_nz / paged_attention / spec_verify_fia_bnsd_paged /
spec_verify_paged_attention / spec_verify_gather_flash / incre_flash_attention /
rms_norm / add_rms_norm / swiglu / npugraph / torchair / pin_memory /
memory_stats_keys / toolchain。

用法（仓库根目录，两种方式等价）：
    python3 -m scripts.probe_ascend310p [--save-out PATH]
    python3 scripts/probe_ascend310p.py [--save-out PATH]
--save-out 缺省写 .omo/evidence/ascend310p-backend/probe_report.json（相对当前目录，
目录不存在则创建）。退出码：无 torch_npu 环境退出 1；正常跑完时全部检查 absent 才 1，
否则 0（单项 FAIL 不中断、不改变退出码）。--help 在任意机器可用（不触碰 torch_npu）。
"""

import argparse
import json
import math
import os
import subprocess
import sys

# 数值容差：板端实证口径（参考实现 probe 同款），不得改动。
RTOL = 0.05
ATOL = 0.05
# NZ 分形格式（参考实现同款常量）。
ACL_FORMAT_FRACTAL_NZ = 29
DEFAULT_SAVE = ".omo/evidence/ascend310p-backend/probe_report.json"

# torch 延迟 import：--help 路径不触碰 torch（torch_npu 缺失机器上 help 必须可用）。
torch = None


def make_tensor(shape, offset):
    """确定性合成张量（参考实现 probe 同款取值链：arange→rem97→偏移→/32→fp16→NPU）。

    值域 [-1.5, 1.5)，取值只由 shape 与 offset 决定——CPU 参考可离板重建同值张量。
    """
    count = math.prod(shape)
    return (
        torch.arange(offset, offset + count, dtype=torch.float32, device="cpu")
        .remainder(97)
        .sub(48)
        .div(32)
        .reshape(shape)
        .to(torch.float16)
        .to("npu")
    )


def ref_gqa_attention(q, k, v, causal, mask=None):
    """CPU float32 参考注意力（GQA：k/v 头 repeat_interleave 到 q 头数）。

    q [Sq, Hq, D] / k, v [Skv, Hkv, D] → [Sq, Hq, D]。causal=True 时要求 Sq==Skv
    （prefill 场景）：第 i 个 q 位置仅可见前 i+1 个 k/v 位置；causal=False 用于
    decode（单 q 位置 attend 全部历史）。mask（bool，True=遮蔽，[Sq, Skv]）为
    spec verify 候选组专用：与 causal 互斥使用，逐位覆盖可见性（verify 行 j
    可见列 0..l+j-1 之类的非对齐结构无法用 causal 布尔表达）。
    """
    reps = q.shape[-2] // k.shape[-2]
    d = q.shape[-1]
    qc = q.float().cpu()
    kc = k.float().cpu().repeat_interleave(reps, dim=-2)
    vc = v.float().cpu().repeat_interleave(reps, dim=-2)
    scores = qc @ kc.transpose(-2, -1) * d ** -0.5
    if causal:
        upper = torch.triu(torch.ones(scores.shape[-2:], dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(upper, float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ vc


# ---------------------------------------------------------------------------
# 数值契约组
# ---------------------------------------------------------------------------


def _check_fia_tnd(tn):
    """fia_v2_tnd_prefill：FIA v2 TND varlen prefill 契约。

    参数组对照 ascend.py prefill_attention 的 eager 分支（图路径引擎传 int32 张量，
    eager 传 list——probe 取 eager 口径）。形状（合成，非真实模型配置）：
    2 条序列 q 长度 [2, 3] → total=5；q [total, H_q=4, D=64]（token 数, 查询头数, 头维）；
    k/v [total, H_kv=2, 64]（GQA 组数 2）。bool 因果掩码大于最长序列
    （引擎 max_context_len 全量掩码同思路），sparse_mode=3 = 左上对齐下三角。
    """
    hq, hkv, d = 4, 2, 64
    lens = [2, 3]
    total = sum(lens)
    q = make_tensor((total, hq, d), 0)
    k = make_tensor((total, hkv, d), 11)
    v = make_tensor((total, hkv, d), 23)
    mask = ~torch.tril(torch.ones((8, 8), dtype=torch.bool, device="npu"))  # 上三角 True=遮蔽
    out = tn.npu_fused_infer_attention_score_v2(
        q, k, v,
        num_query_heads=hq,
        num_key_value_heads=hkv,
        input_layout="TND",
        softmax_scale=d ** -0.5,
        atten_mask=mask,
        actual_seq_qlen=lens,       # 每序列 q 长度（list，ascend.py eager 口径）
        actual_seq_kvlen=lens,
        sparse_mode=3,
        inner_precise=1,
    )
    if isinstance(out, (tuple, list)):      # 引擎侧为无条件 [0]（v2 返回元组契约）
        out = out[0]
    assert tuple(out.shape) == (total, hq, d), f"输出形状 {tuple(out.shape)} != {(total, hq, d)}"
    refs = []
    off = 0
    for n in lens:
        refs.append(ref_gqa_attention(q[off:off + n], k[off:off + n], v[off:off + n], causal=True))
        off += n
    torch.testing.assert_close(
        out.float().cpu(), torch.cat(refs, dim=0), rtol=RTOL, atol=ATOL)
    return f"TND varlen 2 序列（q len {lens}）、GQA {hq}/{hkv}、bool 因果掩码 sparse_mode=3 数值一致"


def _paged_fixture():
    """paged 形态共用夹具：KV 池 + 2 序列 block_table + q。

    形状含义：池 [num_blocks=4, block_size=4, H_kv=2, D=64]（块数, 块内槽位数, KV 头数, 头维）；
    q [bs=2, H_q=4, D=64]（批, 查询头数, 头维）；ctx=[5, 3] → 块表 [[0,1],[2,-1]]，
    未用表槽位 -1（引擎图路径 padding 同款语义）。
    """
    nb, bs, hkv, hq, d = 4, 4, 2, 4, 64
    ctx = [5, 3]
    k_cache = make_tensor((nb, bs, hkv, d), 31)
    v_cache = make_tensor((nb, bs, hkv, d), 57)
    q = make_tensor((len(ctx), hq, d), 83)
    table = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32, device="npu")
    return k_cache, v_cache, q, table, ctx, (nb, bs, hkv, hq, d)


def _paged_refs(q, k_cache, v_cache, table, ctx, block_size, hkv, d):
    """按 block_table 从池中 gather 历史 k/v（扁平槽位语义）后的 CPU 参考注意力 → [bs, H_q, D]。"""
    kc, vc = k_cache.float().cpu(), v_cache.float().cpu()
    tab = table.cpu()
    refs = []
    for i, n in enumerate(ctx):
        n_blk = (n + block_size - 1) // block_size
        ids = tab[i, :n_blk].long()
        kh = kc[ids].reshape(-1, hkv, d)[:n]
        vh = vc[ids].reshape(-1, hkv, d)[:n]
        refs.append(ref_gqa_attention(q[i:i + 1], kh, vh, causal=False))
    return torch.cat(refs, dim=0)


def _verify_fixture():
    """spec verify 三候选共用夹具：KV 池 + 2 序列块表 + 均质 verify q（varlen 拼接）。

    EAGLE3 verify 语义（对照引擎 model_runner._sg_verify / ascend.py
    _verify_causal_mask_rows，两处须保持同构）：每序列 q 行数 rows=k+1=3（k=2，
    spec_verify_q_lens 口径），行 j 的 rope 位置 = l-1+j（l=提交前缀长），可见
    key = 位置 0..l-1+j（历史全可见 + q 段因果）；store_kvcache 先行写入全部
    verify 行后单序列 key 总数 = l+k。形状：池 [num_blocks=4, block_size=4,
    H_kv=2, D=64]；历史 l=[5,3] → key 总数 kv=[7,5]（刻意不齐，暴露共享掩码的
    对齐方向）→ 块表 [[0,1],[2,3]]（ceil(7/4)=ceil(5/4)=2 块/序列）；
    q varlen [bs*rows=6, H_q=4, D=64]（引擎口径 cu_seqlens_q=arange*rows）。
    """
    nb, block_size, hkv, hq, d = 4, 4, 2, 4, 64
    hist = [5, 3]
    k_draft = 2
    rows = k_draft + 1
    kv = [n + k_draft for n in hist]        # 单序列 key 总数 [7, 5]
    k_cache = make_tensor((nb, block_size, hkv, d), 31)
    v_cache = make_tensor((nb, block_size, hkv, d), 57)
    q = make_tensor((len(hist) * rows, hq, d), 83)
    table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="npu")
    return k_cache, v_cache, q, table, hist, kv, rows, (block_size, hkv, hq, d)


def _verify_refs(q, k_cache, v_cache, table, kv, rows, block_size, hkv, d):
    """verify CPU 参考：逐序列 gather key 总数（l+k）行 + 行 j 可见列 0..l+j-1
    （遮蔽 c ≥ l+j）的掩码注意力 → [bs*rows, H_q, D]（varlen 拼接，与引擎 q 顺序
    一致；mask 走 ref_gqa_attention 的逐位掩码入参）。"""
    kc, vc = k_cache.float().cpu(), v_cache.float().cpu()
    tab = table.cpu()
    k = rows - 1
    refs = []
    for i, n in enumerate(kv):
        l = n - k                    # 提交前缀长
        n_blk = (n + block_size - 1) // block_size
        ids = tab[i, :n_blk].long()
        kh = kc[ids].reshape(-1, hkv, d)[:n]
        vh = vc[ids].reshape(-1, hkv, d)[:n]
        qs = q.float().cpu()[i * rows:(i + 1) * rows]
        mask = torch.arange(n).unsqueeze(0) >= (l + torch.arange(rows)).unsqueeze(1)
        refs.append(ref_gqa_attention(qs, kh, vh, causal=False, mask=mask))
    return torch.cat(refs, dim=0)


def _check_fia_bnsd(tn):
    """fia_v2_bnsd_paged_decode：FIA v2 BNSD paged decode 契约。

    参数组对照 ascend.py decode_attention：q [bs, H_q, D] → BNSD [bs, H_q, 1, D]
    （批, 头, q 位置数=1, 头维）+ 4D 池直读 + block_table + block_size +
    actual_seq_kvlen + sparse_mode=0（decode 单位置 attend 全历史，无掩码）。
    actual_seq_kvlen 用 CPU int32（任务约定，参考实现口径）；失败时回退设备 int32
    （引擎 eager 实际形态）并记录——两种元数据形态哪边可用都写进 detail。
    """
    k_cache, v_cache, q, table, ctx, (nb, bs, hkv, hq, d) = _paged_fixture()
    refs = _paged_refs(q, k_cache, v_cache, table, ctx, bs, hkv, d)
    q_in = q.view(len(ctx), 1, hq, d).transpose(1, 2).contiguous()  # 引擎同款 BNSD 变形

    def _call(lens):
        out = tn.npu_fused_infer_attention_score_v2(
            q_in, k_cache, v_cache,
            num_query_heads=hq,
            num_key_value_heads=hkv,
            input_layout="BNSD",
            softmax_scale=d ** -0.5,
            block_table=table,          # 设备 int32（引擎 eager 形态）
            block_size=bs,
            actual_seq_kvlen=lens,
            sparse_mode=0,
            inner_precise=1,
        )
        return out[0] if isinstance(out, (tuple, list)) else out

    lens_cpu = torch.tensor(ctx, dtype=torch.int32)             # CPU int32（任务约定）
    try:
        out = _call(lens_cpu)
        meta = "actual_seq_kvlen=CPU int32"
    except Exception as e_cpu:
        try:
            out = _call(lens_cpu.to("npu"))
            meta = f"actual_seq_kvlen=CPU int32 失败（{type(e_cpu).__name__}: {e_cpu}），device int32 通过"
        except Exception as e_dev:
            raise RuntimeError(
                f"actual_seq_kvlen=CPU int32 失败: {e_cpu}；device int32 亦失败: {e_dev}")
    # 引擎同款输出整形：[bs, H, 1, D] → [bs, H, D]
    got = out.transpose(1, 2).reshape(len(ctx), hq, d)
    torch.testing.assert_close(got.float().cpu(), refs, rtol=RTOL, atol=ATOL)
    return f"BNSD paged decode 2 序列 ctx={ctx}、块表 [[0,1],[2,-1]]、{meta}，数值一致"


def _check_reshape_cache(tn):
    """reshape_and_cache：_npu_reshape_and_cache 写入后读回契约。

    参数组对照 ascend.py store_kvcache（k/v [num_tokens, H_kv, D]、4D 池直写、
    扁平 slot_mapping 设备 int32）。形状：池 [3 块, 4 槽/块, H_kv=2, D=64]
    （共 12 扁平槽位，slot = 块号*4 + 块内偏移）；10 个 token，slot 含一个 -1
    （图路径 padding 槽，算子跳负槽是引擎依赖语义——ascend.py store 注释）。
    池预填哨兵值验证三点：① 正槽写入位置正确（扁平语义）；
    ② 未写槽保持哨兵；③ -1 未写穿池尾（若按 C 负索引回绕必污染末块）。
    """
    nb, bs, hkv, d = 3, 4, 2, 64
    n_tok = 10
    k_cache = torch.full((nb, bs, hkv, d), 123.0, dtype=torch.float16, device="npu")
    v_cache = torch.full_like(k_cache, 45.0)
    key = make_tensor((n_tok, hkv, d), 5)
    value = make_tensor((n_tok, hkv, d), 71)
    slots = list(range(9)) + [-1]       # 9 个正槽 + 1 个 -1（跳负槽语义）
    tn._npu_reshape_and_cache(
        key, value, k_cache, v_cache,
        torch.tensor(slots, dtype=torch.int32, device="npu"),   # 引擎传设备 int32
    )
    torch.npu.synchronize()
    kc, vc = k_cache.float().cpu(), v_cache.float().cpu()
    for i, s in enumerate(slots):
        if s < 0:
            continue
        b, o = s // bs, s % bs
        torch.testing.assert_close(kc[b, o], key[i].float().cpu(), rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(vc[b, o], value[i].float().cpu(), rtol=RTOL, atol=ATOL)
    written = {(s // bs, s % bs) for s in slots if s >= 0}
    for b in range(nb):
        for o in range(bs):
            if (b, o) not in written:
                assert kc[b, o].min().item() > 122.0 and vc[b, o].max().item() < 46.0, \
                    f"哨兵槽 ({b},{o}) 被污染——跳负槽语义缺失（引擎图路径依赖被打破）"
    return "9 个正槽扁平语义写入正确；-1 槽未污染哨兵（跳负槽语义成立）"


def _check_flash_nz(tn):
    """flash_attention_nz：_npu_flash_attention（私有算子）+ NZ 因果掩码契约。

    复刻参考实现 probe_310p_native_attention.py 的板端实证形态（vllm-ascend 310P
    prefill 同款私有算子，flash_incre 档主路径）：q/k/v [S, N, D]（序列长, 头数, 头维，
    B=1 隐含）单序列等长；seq_len 用 CPU int32（参考实现口径）；掩码 fp16 -inf 上三角经
    reshape→transpose→npu_format_cast(ACL_FORMAT_FRACTAL_NZ) 变换（nd_to_nz_2d 等价
    写法，掩码边长须 16 对齐、可大于序列——参考实现 512/seq=2 同理，此处 16/seq=4）。
    """
    hq, hkv, d = 4, 2, 64
    s = 4                        # q/kv 等长单序列（flash_incre prefill 入口约束）
    msize = 16                   # NZ 变换 16 对齐
    q = make_tensor((s, hq, d), 0)
    k = make_tensor((s, hkv, d), 11)
    v = make_tensor((s, hkv, d), 23)
    mask = torch.zeros((msize, msize), dtype=torch.float16)
    mask.masked_fill_(torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1), float("-inf"))
    mask_nz = tn.npu_format_cast(
        mask.reshape(1, msize, msize // 16, 16).transpose(1, 2).contiguous().to("npu"),
        ACL_FORMAT_FRACTAL_NZ,
    )
    out = torch.empty_like(q)
    tn._npu_flash_attention(
        query=q, key=k, value=v,
        mask=mask_nz,
        seq_len=torch.tensor([s], dtype=torch.int32),    # CPU int32（参考实现口径）
        scale_value=d ** -0.5,
        num_heads=hq,
        num_kv_heads=hkv,
        out=out,
    )
    torch.npu.synchronize()
    ref = ref_gqa_attention(q, k, v, causal=True)
    torch.testing.assert_close(out.float().cpu(), ref, rtol=RTOL, atol=ATOL)
    fmt = getattr(tn, "get_npu_format", lambda t: "?")(mask_nz)
    return f"q/kv 等长单序列 S={s}、NZ 因果掩码（format={fmt}）数值一致"


def _check_paged(tn):
    """paged_attention：_npu_paged_attention 契约——UNVERIFIED 候选（板端未实证，本 probe 即实证手段）。

    kwarg 组 = vllm-ascend 310P 官方形态（vllm_ascend/_310p/attention/attention_v1.py
    forward_paged_attention：query/key_cache/value_cache/num_kv_heads/num_heads/
    scale_value/block_table/context_lens/out，官方把 context_lens 移到设备）。
    注意：官方 cache 为 5D 对齐布局 (2, B, (H*D)/16, S, 16)，本 probe 用引擎同款
    4D 池 [B, S, H_kv, D]——两种布局形态均 UNVERIFIED，以板端本项结果为准。
    q 形态两连试：A=[bs,1,H,D]（计划 T7 候选 BNSD 形态）→ 失败再试 B=[bs,H,D]；
    任一形态与 CPU 参考一致即 PASS，detail 记录全部尝试。
    """
    k_cache, v_cache, q2, table, ctx, (nb, bs, hkv, hq, d) = _paged_fixture()
    refs = _paged_refs(q2, k_cache, v_cache, table, ctx, bs, hkv, d)   # [bs, H_q, D]
    attempts = []
    for tag, q_shape in (
        ("A: q=[bs,1,H,D]", (len(ctx), 1, hq, d)),
        ("B: q=[bs,H,D]", (len(ctx), hq, d)),
    ):
        try:
            q = make_tensor(q_shape, 83)
            out = torch.empty(q_shape, dtype=torch.float16, device="npu")
            tn._npu_paged_attention(
                query=q,
                key_cache=k_cache,
                value_cache=v_cache,
                num_kv_heads=hkv,
                num_heads=hq,
                scale_value=d ** -0.5,
                block_table=table,
                context_lens=torch.tensor(ctx, dtype=torch.int32, device="npu"),
                out=out,
            )
            torch.npu.synchronize()
            torch.testing.assert_close(
                out.reshape(len(ctx), hq, d).float().cpu(), refs, rtol=RTOL, atol=ATOL)
            attempts.append(f"{tag} PASS")
            return "；".join(attempts) + "（UNVERIFIED 候选获板端实证）"
        except Exception as e:
            attempts.append(f"{tag} FAIL: {type(e).__name__}: {e}")
    raise RuntimeError("；".join(attempts))


def _check_spec_fia_bnsd(tn):
    """spec_verify_fia_bnsd_paged：候选 A——FIA v2 BNSD paged 多 q verify 契约。

    UNVERIFIED 候选（本 probe 即实证手段）。参数组与引擎 ascend.py
    _paged_verify_fia 同构：q [bs,H,rows,D]（BNSD，varlen [bs*rows,H,D] 均质切分
    转置）+ 整池 4D 池直读 + block_table + block_size + actual_seq_kvlen=逐序列
    key 总数（l+k）+ verify 因果掩码（共享 2D [rows, W=max kv]，行 j 遮蔽位
    c ≥ W-(k-j)，sparse_mode=0 无内建因果）。批内 key 长度不齐 kv=[7,5] 刻意暴露
    共享掩码的对齐方向：右对齐窗口语义 PASS / 左对齐或按掩码宽对齐 FAIL。
    两连试各一（同 _check_fia_bnsd / prefill_attention 先例）：actual_seq_kvlen
    CPU int32 → 设备 int32；bool 掩码 → fp16 -inf 掩码（部分 CANN 拒收 bool）。
    任一参数组与 verify CPU 参考一致即 PASS，detail 记录全部尝试。
    """
    k_cache, v_cache, q, table, hist, kv, rows, (block_size, hkv, hq, d) = _verify_fixture()
    refs = _verify_refs(q, k_cache, v_cache, table, kv, rows, block_size, hkv, d)
    k = rows - 1
    bs = len(hist)
    width = max(kv)
    q_in = q.view(bs, rows, hq, d).transpose(1, 2).contiguous()     # [bs, H, rows, D] BNSD
    # 共享 2D 掩码：行 j 遮蔽位 c ≥ width-(k-j)（右缘锚定；与引擎
    # _verify_causal_mask_rows 同公式。width=7 时 [3,7]：行 0 遮 {5,6}、行 1 遮 {6}、
    # 行 2 全可见——上三角阶梯结构）。
    mask_bool = (torch.arange(width, device="npu").unsqueeze(0)
                 >= (width - k + torch.arange(rows, device="npu")).unsqueeze(1))
    mask_fp16 = torch.zeros((rows, width), dtype=torch.float16, device="npu")
    mask_fp16.masked_fill_(mask_bool, float("-inf"))
    attempts = []
    for lens_tag, lens in (
        ("CPU int32", torch.tensor(kv, dtype=torch.int32)),
        ("device int32", torch.tensor(kv, dtype=torch.int32, device="npu")),
    ):
        for mask_tag, mask in (("bool", mask_bool), ("fp16 -inf", mask_fp16)):
            try:
                out = tn.npu_fused_infer_attention_score_v2(
                    q_in, k_cache, v_cache,
                    num_query_heads=hq,
                    num_key_value_heads=hkv,
                    input_layout="BNSD",
                    softmax_scale=d ** -0.5,
                    block_table=table,
                    block_size=block_size,
                    actual_seq_kvlen=lens,
                    atten_mask=mask,
                    sparse_mode=0,
                    inner_precise=1,
                )
                out = out[0] if isinstance(out, (tuple, list)) else out
                got = out.transpose(1, 2).reshape(bs * rows, hq, d)
                torch.testing.assert_close(got.float().cpu(), refs, rtol=RTOL, atol=ATOL)
                attempts.append(f"actual_seq_kvlen={lens_tag}+{mask_tag} 掩码 PASS")
                return "；".join(attempts) + "（UNVERIFIED 候选获板端实证）"
            except Exception as e:
                attempts.append(
                    f"actual_seq_kvlen={lens_tag}+{mask_tag} 掩码 FAIL: {type(e).__name__}: {e}")
    raise RuntimeError("；".join(attempts))


def _check_spec_paged_attention(tn):
    """spec_verify_paged_attention：候选 B——_npu_paged_attention q_len>1 形态。

    UNVERIFIED 候选（存在性由注册表 attrs 门控）。kwarg 组复用 _check_paged 的
    vllm-ascend 310P 官方形态（query/key_cache/value_cache/num_kv_heads/num_heads/
    scale_value/block_table/context_lens/out，官方把 context_lens 移到设备），扩展
    q_len>1（每序列 rows=k+1 均质）；算子无掩码入参——q_len>1 的可见性依赖算子
    内建因果，预期 bottom-right 对齐（行 j 可见 key 窗口前 l+j 列 = verify 语义），
    对齐方向由本项数值比对裁决。q 形态两连试（对照 _check_paged 先例扩展多 q）：
    A=[bs,rows,H,D]（引擎 _paged_verify_paged_attention 首案）→ B=[bs,H,rows,D]
    （BNSD）；任一形态与 verify CPU 参考一致即 PASS，detail 记录全部尝试。
    """
    k_cache, v_cache, q, table, hist, kv, rows, (block_size, hkv, hq, d) = _verify_fixture()
    refs = _verify_refs(q, k_cache, v_cache, table, kv, rows, block_size, hkv, d)
    bs = len(hist)

    def _try(tag, q_in, out_shape, reshape):
        out = torch.empty(out_shape, dtype=torch.float16, device="npu")
        tn._npu_paged_attention(
            query=q_in,
            key_cache=k_cache,
            value_cache=v_cache,
            num_kv_heads=hkv,
            num_heads=hq,
            scale_value=d ** -0.5,
            block_table=table,
            context_lens=torch.tensor(kv, dtype=torch.int32, device="npu"),
            out=out,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(
            reshape(out).float().cpu(), refs, rtol=RTOL, atol=ATOL)
        return f"{tag} PASS"

    attempts = []
    try:
        attempts.append(_try(
            "A: q=[bs,rows,H,D]",
            q.view(bs, rows, hq, d).contiguous(), (bs, rows, hq, d),
            lambda o: o.reshape(bs * rows, hq, d)))
        return "；".join(attempts) + "（UNVERIFIED 候选获板端实证）"
    except Exception as e:
        attempts.append(f"A: q=[bs,rows,H,D] FAIL: {type(e).__name__}: {e}")
    try:
        attempts.append(_try(
            "B: q=[bs,H,rows,D]",
            q.view(bs, rows, hq, d).transpose(1, 2).contiguous(), (bs, hq, rows, d),
            lambda o: o.transpose(1, 2).reshape(bs * rows, hq, d)))
        return "；".join(attempts) + "（UNVERIFIED 候选获板端实证）"
    except Exception as e:
        attempts.append(f"B: q=[bs,H,rows,D] FAIL: {type(e).__name__}: {e}")
    raise RuntimeError("；".join(attempts))


def _check_spec_gather_flash(tn):
    """spec_verify_gather_flash：候选 C——全量 gather + _npu_flash_attention
    不等长 q/kv + 自定义掩码。

    UNVERIFIED 候选。逐序列（引擎 ascend.py _paged_verify_gather_flash 同构）：
    块表 gather key 总数（l+k）行（_gather_kv_history 同款：上取整整块 → 展平 →
    截断）→ q_part [rows,H,D] / k,v [l+k,H_kv,D] 不等长前向 + 行 j 遮蔽位 c ≥ l+j
    的自定义掩码（逐序列精确尺寸，可见性结构与 A 的共享掩码不同源——此处无对齐
    方向问题，考的是算子对不等长 q/kv 与非 NZ 掩码的接受度）。掩码形态两连试：
    ND fp16 精确尺寸 → NZ 分形（16 对齐 padding：q 尾 pad 行掩码全 0 可见防全
    -inf 行 softmax NaN、输出行丢弃；kv 尾 pad 列被遮蔽公式覆盖为 -inf；
    npu_format_cast 同 _check_flash_nz 实证形态）；任一形态与 verify CPU 参考
    一致即 PASS，detail 记录全部尝试。
    """
    k_cache, v_cache, q, table, hist, kv, rows, (block_size, hkv, hq, d) = _verify_fixture()
    refs = _verify_refs(q, k_cache, v_cache, table, kv, rows, block_size, hkv, d)
    k = rows - 1

    def _gather(i, n, cache):
        n_blk = (n + block_size - 1) // block_size
        ids = table[i, :n_blk].long()
        return cache.index_select(0, ids).reshape(-1, hkv, d)[:n]

    def _run(mask_tag):
        outs = []
        for i, n in enumerate(kv):
            l = n - k
            q_part = q[i * rows:(i + 1) * rows]     # [rows, H_q, D]
            k_h, v_h = _gather(i, n, k_cache), _gather(i, n, v_cache)
            if mask_tag == "ND fp16":
                mask = torch.zeros((rows, n), dtype=torch.float16, device="npu")
                mask.masked_fill_(
                    torch.arange(n, device="npu").unsqueeze(0)
                    >= (l + torch.arange(rows, device="npu")).unsqueeze(1),
                    float("-inf"))
                q_in, k_in, v_in, out = q_part, k_h, v_h, torch.empty_like(q_part)
            else:
                rows_pad = (rows + 15) // 16 * 16
                kv_pad = (n + 15) // 16 * 16
                mask = torch.zeros((rows_pad, kv_pad), dtype=torch.float16, device="npu")
                mask.masked_fill_(
                    ~(torch.arange(kv_pad, device="npu").unsqueeze(0)
                      < (l + torch.arange(rows_pad, device="npu")).unsqueeze(1)),
                    float("-inf"))
                mask[rows:, :] = 0.0        # pad q 行全可见（防 NaN，输出行丢弃）
                q_in = torch.zeros((rows_pad, hq, d), dtype=torch.float16, device="npu")
                q_in[:rows] = q_part
                k_in = torch.zeros((kv_pad, hkv, d), dtype=torch.float16, device="npu")
                k_in[:n] = k_h
                v_in = torch.zeros((kv_pad, hkv, d), dtype=torch.float16, device="npu")
                v_in[:n] = v_h
                mask = tn.npu_format_cast(
                    mask.reshape(1, rows_pad, kv_pad // 16, 16).transpose(1, 2).contiguous(),
                    ACL_FORMAT_FRACTAL_NZ)
                out = torch.empty((rows_pad, hq, d), dtype=torch.float16, device="npu")
            tn._npu_flash_attention(
                query=q_in, key=k_in, value=v_in,
                mask=mask,
                seq_len=torch.tensor([rows], dtype=torch.int32),    # q 侧行数（UNVERIFIED 语义）
                scale_value=d ** -0.5,
                num_heads=hq,
                num_kv_heads=hkv,
                out=out,
            )
            torch.npu.synchronize()
            outs.append(out[:rows])
        got = torch.cat(outs, dim=0)
        torch.testing.assert_close(got.float().cpu(), refs, rtol=RTOL, atol=ATOL)

    attempts = []
    for mask_tag in ("ND fp16", "NZ 16对齐 padding"):
        try:
            _run(mask_tag)
            attempts.append(f"{mask_tag} 掩码 PASS")
            return "；".join(attempts) + "（UNVERIFIED 候选获板端实证）"
        except Exception as e:
            attempts.append(f"{mask_tag} 掩码 FAIL: {type(e).__name__}: {e}")
    raise RuntimeError("；".join(attempts))


def _check_incre(tn):
    """incre_flash_attention：npu_incre_flash_attention 逐序列 BNSD 契约。

    复刻参考实现 decode 兜底的板端实证形态（nano_vllm_310p_compat.py:245-296，
    flash_incre decode 主路径）：每序列 q_part [1, H_q, 1, D]（批, 头, 位置数, 头维）、
    历史 k/v [1, H_kv, ctx, D]，actual_seq_lengths=[ctx]（list）、inner_precise=1；
    输出 squeeze(0).transpose(0,1) 拼回 [bs, H_q, D]。probe 用 ctx [5, 3] 两条序列的
    稠密历史（paged gather 数学已由 reshape_and_cache / fia_bnsd / paged 项覆盖）。
    """
    hq, hkv, d = 4, 2, 64
    outs = []
    for i, n in enumerate((5, 3)):
        q1 = make_tensor((1, hq, d), 83 + i * 7)                     # 单 token 的 q
        k_h = make_tensor((n, hkv, d), 131 + i * 3)
        v_h = make_tensor((n, hkv, d), 163 + i * 3)
        q_part = q1.transpose(0, 1).unsqueeze(0).contiguous()        # [1, H_q, 1, D]
        k_part = k_h.transpose(0, 1).unsqueeze(0).contiguous()       # [1, H_kv, ctx, D]
        v_part = v_h.transpose(0, 1).unsqueeze(0).contiguous()
        out = tn.npu_incre_flash_attention(
            q_part, k_part, v_part,
            actual_seq_lengths=[n],                                  # compat 同款 list
            num_heads=hq,
            num_key_value_heads=hkv,
            input_layout="BNSD",
            scale_value=d ** -0.5,
            inner_precise=1,
        )
        outs.append(out.squeeze(0).transpose(0, 1).contiguous())     # compat 同款整形 → [1, H_q, D]
    got = torch.cat(outs, dim=0)
    refs = []
    for i, n in enumerate((5, 3)):
        refs.append(ref_gqa_attention(
            make_tensor((1, hq, d), 83 + i * 7),
            make_tensor((n, hkv, d), 131 + i * 3),
            make_tensor((n, hkv, d), 163 + i * 3),
            causal=False))
    torch.testing.assert_close(got.float().cpu(), torch.cat(refs, dim=0), rtol=RTOL, atol=ATOL)
    return "逐序列 BNSD（ctx 5/3，actual_seq_lengths=[ctx] list）数值一致"


def _check_rms(tn):
    """rms_norm：npu_rms_norm 返回 (x, rstd) 解包（ascend.py rms_norm 同款），比对归一化输出。

    形状：x [8, 64]（token 数, 隐藏维）、weight [64]。参考 = x * rsqrt(mean(x²)+eps) * w。
    """
    d = 64
    x = make_tensor((8, d), 0)
    w = make_tensor((d,), 3)
    eps = 1e-6
    x_out, _ = tn.npu_rms_norm(x, w, eps)          # 引擎只要第一个返回值
    xf, wf = x.float().cpu(), w.float().cpu()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * wf
    torch.testing.assert_close(x_out.float().cpu(), ref, rtol=RTOL, atol=ATOL)
    return "归一化输出一致（(x, rstd) 解包）"


def _check_add_rms(tn):
    """add_rms_norm：npu_add_rms_norm 返回 (x, rstd, residual) 解包（ascend.py add_rms_norm 同款）。

    语义：residual_out = x + residual；output = rmsnorm(residual_out) * w——两者都比对。
    """
    d = 64
    x = make_tensor((8, d), 0)
    res = make_tensor((8, d), 37)
    w = make_tensor((d,), 3)
    eps = 1e-6
    x_out, _, res_out = tn.npu_add_rms_norm(x, res, w, eps)
    added = x.float().cpu() + res.float().cpu()
    ref = added * torch.rsqrt(added.pow(2).mean(-1, keepdim=True) + eps) * w.float().cpu()
    torch.testing.assert_close(x_out.float().cpu(), ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(res_out.float().cpu(), added, rtol=RTOL, atol=ATOL)
    return "归一化输出与残差和均一致（(x, rstd, residual) 三元组解包）"


def _check_swiglu(tn):
    """swiglu：npu_swiglu 默认 dim=-1（ascend.py silu_and_mul 同款），SwiGLU = silu(前半) * 后半。

    形状：x [8, 128]（token 数, 2*中间维）。
    """
    x = make_tensor((8, 128), 0)
    y = tn.npu_swiglu(x)                           # 默认 dim=-1
    a, b = x.float().cpu().chunk(2, dim=-1)
    ref = torch.nn.functional.silu(a) * b
    torch.testing.assert_close(y.float().cpu(), ref, rtol=RTOL, atol=ATOL)
    return "SwiGLU（dim=-1 切半）数值一致"


# ---------------------------------------------------------------------------
# 基础设施组
# ---------------------------------------------------------------------------


def _check_npugraph(report):
    """npugraph：NPUGraph 存在性 + pool kwarg TypeError 探测 + 最小捕获回放。

    捕获体 = 一段 matmul + copy_（引擎 decode 图的最小同构）；先预热两轮再捕获；
    优先带 pool kwarg 捕获（pool=None 为首捕获合法值），TypeError（厂商 torch 不支持
    pool 形参）则回退无 pool 捕获并把结论记入 detail——图三档链第一档的前提探测。
    回放验证：改写静态输入后 replay 一次，输出须等于新输入的 eager 结果。
    """
    g_cls = getattr(torch.npu, "NPUGraph", None)
    g_ctx = getattr(torch.npu, "graph", None)
    if g_cls is None or g_ctx is None:
        report.record("npugraph", "absent", "torch.npu.NPUGraph / torch.npu.graph 不存在")
        return
    a = make_tensor((8, 8), 0)          # 图内静态输入（批, 维）
    w = make_tensor((8, 8), 41)         # 静态权重
    out = torch.empty_like(a)
    for _ in range(2):                  # 捕获前预热：目标算子先在设备上初始化
        out.copy_(a @ w)
    torch.npu.synchronize()
    g = g_cls()
    try:
        with g_ctx(g, pool=None):
            out.copy_(a @ w)
        pool_note = "pool kwarg 可用（pool=None）"
    except TypeError as e:
        pool_note = f"pool kwarg TypeError: {e} → 无 pool 回退捕获"
        with g_ctx(g):
            out.copy_(a @ w)
    torch.npu.synchronize()
    a.copy_(make_tensor((8, 8), 97))    # 回放前改写静态输入
    g.replay()
    torch.npu.synchronize()
    ref = a @ w
    torch.testing.assert_close(out.float().cpu(), ref.float().cpu(), rtol=RTOL, atol=ATOL)
    report.record("npugraph", "PASS", f"最小捕获（matmul+copy_）回放数值一致；{pool_note}")


def _check_torchair(report):
    """torchair：先 torch_npu.dynamo 再 torchair——图三档链第二档（REDUCE_OVERHEAD）前提探测。

    缺失记 absent（计划 C4 第二档据此跳过并告警）；可 import 则记录版本。
    """
    try:
        import torch_npu.dynamo  # noqa: F401
    except Exception as e:
        report.record("torchair", "absent", f"torch_npu.dynamo 不可 import: {e}")
        return
    try:
        import torchair
    except Exception as e:
        report.record("torchair", "absent", f"torch_npu.dynamo 可用但 torchair 不可 import: {e}")
        return
    ver = getattr(torchair, "__version__", "unknown")
    report.record("torchair", "PASS", f"torchair {ver} 可 import（经 torch_npu.dynamo，第二档前提成立）")


def _check_pin_memory(report):
    """pin_memory：CPU 张量 pin_memory() 分配 + .to('npu', non_blocking=True) 往返。

    引擎 pinned 元数据 blob（MetaBlob）与 prepare_block_tables 均依赖此链路。
    """
    t = torch.arange(256, dtype=torch.float32)
    try:
        p = t.pin_memory()
    except Exception as e:
        report.record("pin_memory", "FAIL", f"CPU 张量 pin_memory() 分配失败: {e}")
        return
    d = p.to("npu", non_blocking=True)
    torch.npu.synchronize()
    back = d.float().cpu()
    torch.testing.assert_close(back, t, rtol=RTOL, atol=ATOL)
    report.record("pin_memory", "PASS", "pin_memory 分配 + non_blocking H2D 往返一致")


def _check_memory_stats(report):
    """memory_stats_keys：torch.npu.memory_stats() 是否含引擎显存统计依赖的键名。

    引擎 KV 池预算与显存打印消费 allocated_bytes.all.peak / .current。
    """
    stats = torch.npu.memory_stats()
    if not stats:
        report.record("memory_stats_keys", "FAIL", "memory_stats() 返回空")
        return
    need = ("allocated_bytes.all.peak", "allocated_bytes.all.current")
    missing = [k for k in need if k not in stats]
    if missing:
        sample = ", ".join(list(stats)[:8])
        report.record("memory_stats_keys", "FAIL", f"缺键 {missing}（现有键样例: {sample}）")
    else:
        report.record("memory_stats_keys", "PASS", "allocated_bytes.all.peak/current 均存在")


def _collect_toolchain(tn):
    """版本采集：torch_npu 版本 + CANN 版本（torch.version.cann，取不到记 unknown）+ torch 版本。"""
    cann = getattr(getattr(torch, "version", None), "cann", None)
    return {
        "torch_npu": getattr(tn, "__version__", None) or "unknown",
        "cann": cann or "unknown",
        "torch": getattr(torch, "__version__", "unknown"),
    }


# ---------------------------------------------------------------------------
# 报告输出（独立可调用：JSON schema 自检用假数据直接走这几个函数）
# ---------------------------------------------------------------------------

# 检查注册表（顺序即报告 JSON 的键顺序）：name → (缺失即 absent 的算子属性, 契约函数)
NUMERIC_CHECKS = (
    ("fia_v2_tnd_prefill", ("npu_fused_infer_attention_score_v2",), _check_fia_tnd),
    ("fia_v2_bnsd_paged_decode", ("npu_fused_infer_attention_score_v2",), _check_fia_bnsd),
    ("reshape_and_cache", ("_npu_reshape_and_cache",), _check_reshape_cache),
    ("flash_attention_nz", ("_npu_flash_attention", "npu_format_cast"), _check_flash_nz),
    ("paged_attention", ("_npu_paged_attention",), _check_paged),
    ("spec_verify_fia_bnsd_paged", ("npu_fused_infer_attention_score_v2",), _check_spec_fia_bnsd),
    ("spec_verify_paged_attention", ("_npu_paged_attention",), _check_spec_paged_attention),
    ("spec_verify_gather_flash", ("_npu_flash_attention", "npu_format_cast"), _check_spec_gather_flash),
    ("incre_flash_attention", ("npu_incre_flash_attention",), _check_incre),
    ("rms_norm", ("npu_rms_norm",), _check_rms),
    ("add_rms_norm", ("npu_add_rms_norm",), _check_add_rms),
    ("swiglu", ("npu_swiglu",), _check_swiglu),
)

INFRA_CHECKS = (
    ("npugraph", _check_npugraph),
    ("torchair", _check_torchair),
    ("pin_memory", _check_pin_memory),
    ("memory_stats_keys", _check_memory_stats),
)


class _Report:
    """checks 收集器：单项 FAIL/absent 只记录不中断（status ∈ PASS|FAIL|absent）。"""

    def __init__(self):
        self.checks = {}

    def record(self, name, status, detail):
        self.checks[name] = {"status": status, "detail": str(detail)}


def _run_numeric(report, tn):
    """逐项跑数值契约：算子缺失 → absent；异常/断言失败 → FAIL（不中断）。"""
    for name, attrs, fn in NUMERIC_CHECKS:
        missing = [a for a in attrs if not hasattr(tn, a)]
        if missing:
            report.record(name, "absent", f"算子缺失: {', '.join(missing)}")
            continue
        try:
            report.record(name, "PASS", fn(tn))
        except Exception as e:
            report.record(name, "FAIL", f"{type(e).__name__}: {e}")


def _run_infra(report):
    """逐项跑基础设施探测：函数内部自记 absent；其余异常兜底为 FAIL。"""
    for name, fn in INFRA_CHECKS:
        try:
            fn(report)
        except Exception as e:
            report.record(name, "FAIL", f"{type(e).__name__}: {e}")


def build_report(toolchain, repo_hash, recommended, checks, spec_unlock_candidate=None):
    """组报告 dict（顶层五键 = 定死契约：toolchain / repo_hash /
    recommended_attention_impl / spec_unlock_candidate / checks；spec_unlock_candidate
    缺省 None 仅为向后兼容旧四参调用，契约要求键恒在——全 FAIL/报告缺项即 null）。"""
    return {
        "toolchain": toolchain,
        "repo_hash": repo_hash,
        "recommended_attention_impl": recommended,
        "spec_unlock_candidate": spec_unlock_candidate,
        "checks": checks,
    }


def decide_recommended(checks):
    """档位裁决：fia_v2 仅当 FIA v2 两项契约双 PASS，否则 flash_incre（保守默认）。"""
    tnd = checks.get("fia_v2_tnd_prefill", {}).get("status")
    bnsd = checks.get("fia_v2_bnsd_paged_decode", {}).get("status")
    return "fia_v2" if tnd == "PASS" and bnsd == "PASS" else "flash_incre"


# spec 解锁候选裁决序（计划 Scope #8）：A→B→C，检查名 → 候选编号。
_SPEC_CANDIDATE_ORDER = (
    ("spec_verify_fia_bnsd_paged", "A"),
    ("spec_verify_paged_attention", "B"),
    ("spec_verify_gather_flash", "C"),
)


def decide_spec_unlock_candidate(checks):
    """spec 解锁候选裁决：按 A→B→C 序取首个数值契约 PASS 者；全 FAIL/absent 记
    None（JSON 序列化为 null）——T14 翻转 supports_spec 的前置证据键，null 即
    维持引擎层拒绝（ascend.py supports_spec 注释引用本键与证据路径）。"""
    for name, candidate in _SPEC_CANDIDATE_ORDER:
        if checks.get(name, {}).get("status") == "PASS":
            return candidate
    return None


def report_exit_code(checks):
    """退出码契约：全部 absent 才 1（没有任何真实探测发生），否则 0。"""
    statuses = {c["status"] for c in checks.values()}
    return 1 if statuses == {"absent"} else 0


def emit_report(payload, save_out):
    """stdout 打印 JSON + 落盘（目录不存在则创建）。落盘提示走 stderr，不污染 stdout JSON。"""
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if save_out:
        parent = os.path.dirname(os.path.abspath(save_out))
        os.makedirs(parent, exist_ok=True)
        with open(save_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[probe_ascend310p] 报告已写入 {save_out}", file=sys.stderr)


def _git_repo_hash():
    """git rev-parse --short HEAD（消费端漂移告警用）；失败/非 git 目录记 unknown。"""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=10, text=True,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def _print_env_guide(title, err):
    """无 torch_npu 环境的中文说明（口径对齐 nanovllm/backends/ascend.py 的 _torch_npu() 门）。"""
    print(f"[probe_ascend310p] {title}：{err}")
    print(
        "本探测脚本需在 Ascend 310P 板端（Atlas 300I Duo / 200I Pro）运行，产出 JSON 报告\n"
        "供引擎 attention_impl=\"auto\" 档位裁决；当前机器无 torch_npu，无法执行任何探测项。\n"
        "\n"
        "环境安装指引（对齐 vllm-ascend 310P 口径，版本钉由 ascend 后端启动门校验）：\n"
        "  1. torch-npu==2.10.0.post4 / CANN 9.1.0；\n"
        "  2. 先 source CANN 安装目录的 set_env.sh（AscendCL 运行时），再激活 venv；\n"
        "  3. 安装 torch_npu（pip install torch_npu==2.10.0.post4 或按板卡厂商指引装整包）；\n"
        "  4. 装好后重跑：python3 -m scripts.probe_ascend310p --save-out probe_report.json\n"
        "\n"
        "提示：--help 可在任意机器查看用法，不依赖 torch_npu。"
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ascend 310P 逐算子数值契约 + 基础设施探测，输出 JSON 报告"
                    "（引擎 attention_impl=auto 档位裁决依据，计划任务 8）",
        epilog="示例：python3 -m scripts.probe_ascend310p --save-out probe_report.json\n"
               "工具链钉口径：torch-npu==2.10.0.post4 / CANN 9.1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--save-out", default=DEFAULT_SAVE,
        help="JSON 报告落盘路径（默认 .omo/evidence/ascend310p-backend/probe_report.json，"
             "相对当前目录，建议从仓库根目录运行；目录不存在则创建）",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)            # --help 在此即返回，不触碰 torch/torch_npu
    global torch
    try:
        import torch as _torch
    except ImportError as e:
        _print_env_guide("未找到 torch", str(e))
        return 1
    torch = _torch
    try:
        import torch_npu
    except ImportError as e:
        _print_env_guide("未找到 torch_npu", str(e))
        return 1

    report = _Report()
    _run_numeric(report, torch_npu)     # 数值契约组：FAIL/absent 只记录不中断
    _run_infra(report)                  # 基础设施组
    toolchain = _collect_toolchain(torch_npu)
    if toolchain["torch_npu"] == "unknown" and toolchain["cann"] == "unknown":
        report.record("toolchain", "absent", "torch_npu / CANN 版本均不可得")
    else:
        report.record(
            "toolchain", "PASS",
            f"torch_npu={toolchain['torch_npu']}；CANN={toolchain['cann']}；torch={toolchain['torch']}")

    payload = build_report(
        toolchain, _git_repo_hash(), decide_recommended(report.checks), report.checks,
        decide_spec_unlock_candidate(report.checks))
    emit_report(payload, args.save_out)
    return report_exit_code(report.checks)


if __name__ == "__main__":
    sys.exit(main())
