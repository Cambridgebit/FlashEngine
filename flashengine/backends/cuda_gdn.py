"""CUDA GDN 多步递推：状态驻留寄存器，按验证行直接写入终态槽。"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS"])
def _recurrent_kernel(Q, K, V, G, Beta, Cache, Carry, Initial, Out,
                      QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr,
                      KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
                      VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
                      GS0: tl.constexpr, BS0: tl.constexpr,
                      H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                      ROWS, BASE: tl.constexpr,
                      VERIFY: tl.constexpr, HAS_INITIAL: tl.constexpr,
                      DUMMY: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    seq, head, vb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kk = tl.arange(0, BK)
    vv = vb * BV + tl.arange(0, BV)
    mask = (kk[:, None] < DK) & (vv[None, :] < DV)
    carry = tl.load(Carry + seq)
    offsets = head * DK * DV + kk[:, None] * DV + vv[None, :]
    state = tl.load(Cache + carry * H * DK * DV + offsets, mask=mask, other=0).to(tl.float32)
    if HAS_INITIAL:
        initial = tl.load(Initial + seq)
        state = tl.where(initial, state, 0.)
    # 同一 CTA 保留一个 value 分片的所有 key 维状态。
    for row in range(ROWS):
        token = seq * ROWS + row
        q = tl.load(Q + token * QS0 + head * QS1 + kk * QS2, mask=kk < DK, other=0).to(tl.float32)
        k = tl.load(K + token * KS0 + head * KS1 + kk * KS2, mask=kk < DK, other=0).to(tl.float32)
        v = tl.load(V + token * VS0 + head * VS1 + vv * VS2, mask=vv < DV, other=0).to(tl.float32)
        g = tl.load(G + token * GS0 + head).to(tl.float32)
        beta = tl.load(Beta + token * BS0 + head).to(tl.float32)
        state = state * tl.exp(g)
        memory = tl.sum(state * k[:, None], axis=0)
        delta = (v - memory) * beta
        state = state + k[:, None] * delta[None, :]
        out = tl.sum(state * q[:, None], axis=0)
        tl.store(Out + (token * H + head) * DV + vv, out, mask=vv < DV)
        if VERIFY:
            tl.store(Cache + (BASE + token) * H * DK * DV + offsets, state, mask=mask)
        else:
            # 图 padding 可有多个 dummy 行，不让多个 CTA 写同一槽。
            if row == ROWS - 1:
                tl.store(Cache + carry * H * DK * DV + offsets, state, mask=mask & (carry != DUMMY))


def recurrent(query, key, value, g, beta, cache, carry, *, rows=1, base=None, initial=None):
    """输入 q/k 已按参考实现归一化；保留 fp32 状态及各行接受后回滚能力。"""
    tokens, heads, dk = query.shape
    dv = value.shape[-1]
    output = torch.empty_like(value, memory_format=torch.contiguous_format)
    _recurrent_kernel[(tokens // rows, heads, triton.cdiv(dv, 32))](
        query, key, value, g, beta, cache, carry, initial if initial is not None else carry, output,
        *query.stride(), *key.stride(), *value.stride(), g.stride(0), beta.stride(0),
        heads, dk, dv, rows, base or 0, base is not None, initial is not None,
        cache.shape[0] - 1, triton.next_power_of_2(dk), 32,
        num_warps=4, enable_fp_fusion=False)
    return output


@triton.jit
def _norm_kernel(X, Weight, Gate, Residual, Out, ResidualOut,
                 XS: tl.constexpr, GS: tl.constexpr, RS: tl.constexpr,
                 D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
                 GEMMA: tl.constexpr, ADD: tl.constexpr, SIGMOID: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * XS + cols, mask=cols < D, other=0).to(tl.float32)
    if ADD:
        r = tl.load(Residual + row * RS + cols, mask=cols < D, other=0).to(tl.float32)
        # 残差先按模型 dtype 舍入，不能偷偷改为 fp32 残差流。
        x = (x + r).to(X.dtype.element_ty).to(tl.float32)
        tl.store(ResidualOut + row * D + cols, x, mask=cols < D)
    variance = tl.sum(x * x, axis=0) / D
    norm = x * tl.rsqrt(variance + EPS)
    weight = tl.load(Weight + cols, mask=cols < D, other=0).to(tl.float32)
    if GEMMA:
        norm = norm * (1. + weight)
    else:
        # HF GDN 的归一化、weight 乘法各在模型 dtype 处舍入。
        norm = norm.to(X.dtype.element_ty).to(tl.float32)
        norm = (norm * weight).to(X.dtype.element_ty).to(tl.float32)
        gate = tl.load(Gate + row * GS + cols, mask=cols < D, other=0).to(tl.float32)
        activation = tl.sigmoid(gate)
        if not SIGMOID:
            activation = gate * activation
        norm = norm * activation
    tl.store(Out + row * D + cols, norm, mask=cols < D)


def norm(x, weight, eps, *, residual=None, gate=None, activation="silu"):
    """Gemma 残差 RMSNorm / GDN 门控 RMSNorm，保留 HF 中间舍入顺序。"""
    d = x.shape[-1]
    flat = x.reshape(-1, d)
    # 当前模型 head / hidden 的末维连续；其余 stride 可含 q/gate 间隙。
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    r = residual.reshape(-1, d) if residual is not None else flat
    z = gate.reshape(-1, d) if gate is not None else flat
    if z.stride(-1) != 1:
        z = z.contiguous()
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    summed = torch.empty_like(output) if residual is not None else output
    _norm_kernel[(flat.shape[0],)](flat, weight, z, r, output, summed,
        flat.stride(0), z.stride(0), r.stride(0), d, eps, triton.next_power_of_2(d),
        gate is None, residual is not None, activation == "sigmoid",
        num_warps=4, enable_fp_fusion=False)
    return (output, summed) if residual is not None else output


@triton.jit
def _conv_kernel(X, Weight, Cache, Carry, Initial, Out,
                 XS0: tl.constexpr, XS1: tl.constexpr, C: tl.constexpr, KW: tl.constexpr,
                 ROWS: tl.constexpr, BASE: tl.constexpr, VERIFY: tl.constexpr,
                 HAS_INITIAL: tl.constexpr, DUMMY: tl.constexpr,
                 BC: tl.constexpr, BT: tl.constexpr):
    seq = tl.program_id(0)
    channels = tl.program_id(1) * BC + tl.arange(0, BC)
    time = tl.arange(0, BT)
    carry = tl.load(Carry + seq)
    active = tl.load(Initial + seq) if HAS_INITIAL else True
    total = tl.full((BT, BC), 0., tl.float32)
    for tap in tl.static_range(KW):
        position = time + tap - (KW - 1)
        previous = tl.load(Cache + carry * C * (KW - 1) + channels[None, :] * (KW - 1)
                           + (time + tap)[:, None],
                           mask=(position[:, None] < 0) & (time[:, None] < ROWS)
                           & (channels[None, :] < C), other=0).to(tl.float32)
        previous = tl.where(active, previous, 0.)
        current = tl.load(X + (seq * ROWS + position[:, None]) * XS0 + channels[None, :] * XS1,
                          mask=(position[:, None] >= 0) & (time[:, None] < ROWS)
                          & (channels[None, :] < C), other=0).to(tl.float32)
        value = tl.where(position[:, None] < 0, previous, current)
        weight = tl.load(Weight + channels * KW + tap, mask=channels < C, other=0).to(tl.float32)
        total = total + value * weight[None, :]
    # Conv1d 返回模型 dtype，再应用 SiLU；不可跨越此舍入点融合。
    total = total.to(X.dtype.element_ty).to(tl.float32)
    output = total * tl.sigmoid(total)
    tl.store(Out + (seq * ROWS + time[:, None]) * C + channels[None, :], output,
             mask=(time[:, None] < ROWS) & (channels[None, :] < C))
    for column in tl.static_range(KW - 1):
        position = time + column + 1 - (KW - 1)
        previous = tl.load(Cache + carry * C * (KW - 1) + channels[None, :] * (KW - 1)
                           + (time + column + 1)[:, None],
                           mask=(position[:, None] < 0) & (time[:, None] < ROWS)
                           & (channels[None, :] < C), other=0)
        previous = tl.where(active, previous, 0.)
        current = tl.load(X + (seq * ROWS + position[:, None]) * XS0 + channels[None, :] * XS1,
                          mask=(position[:, None] >= 0) & (time[:, None] < ROWS)
                          & (channels[None, :] < C), other=0)
        value = tl.where(position[:, None] < 0, previous, current)
        destination = BASE + seq * ROWS + time if VERIFY else carry + tl.zeros((BT,), tl.int32)
        tl.store(Cache + destination[:, None] * C * (KW - 1)
                 + channels[None, :] * (KW - 1) + column, value,
                 mask=(time[:, None] < ROWS) & (channels[None, :] < C)
                 & (VERIFY | (carry != DUMMY)))


def conv(inputs, weight, cache, carry, *, rows=1, base=None, initial=None):
    """短因果卷积 + SiLU + 各行 raw 历史更新，避免分组 Conv1d 的小算子开销。"""
    channels = inputs.shape[-1]
    output = torch.empty(inputs.shape, dtype=inputs.dtype, device=inputs.device)
    _conv_kernel[(inputs.shape[0] // rows, triton.cdiv(channels, 128))](
        inputs, weight, cache, carry, initial if initial is not None else carry, output,
        *inputs.stride(), channels, weight.shape[-1], rows, base or 0,
        base is not None, initial is not None, cache.shape[0] - 1,
        128, triton.next_power_of_2(rows), num_warps=4, enable_fp_fusion=False)
    return output


@triton.jit
def _commit_kernel(Pool, Counts, Targets, Slots: tl.constexpr, Size: tl.constexpr,
                   CS: tl.constexpr, ROWS: tl.constexpr, BASE: tl.constexpr,
                   BLOCK: tl.constexpr):
    layer, seq, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    target = tl.load(Targets + seq)
    count = tl.load(Counts + seq * CS)
    source = BASE + seq * ROWS + count - 1
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = (offsets < Size) & (target != Slots - 1)
    values = tl.load(Pool + (layer * Slots + source) * Size + offsets, mask=mask, other=0)
    tl.store(Pool + (layer * Slots + target) * Size + offsets, values, mask=mask)


def commit(pool, commit_buf, targets, rows, base):
    """直接读取图内接受数，将所选验证终态拷回请求主槽；不造整块临时状态。"""
    size = pool[0, 0].numel()
    _commit_kernel[(pool.shape[0], targets.numel(), triton.cdiv(size, 1024))](
        pool, commit_buf[:, rows], targets, pool.shape[1], size, commit_buf.stride(0),
        rows, base, 1024)
