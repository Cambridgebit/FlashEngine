"""字节级语法机：段 → 候选集合 → token 掩码。

约束解码拆成三层：

- **段（Segment）**：一段可复用的确定性字节状态机（字面量、自由文本、数值文本、JSON 值）。
  ``step`` 返回下一个状态、``None``（该字节非法）、``DONE``（消费本字节后本段结束）、
  ``PENDING``（本段已经可以结束，但本字节没被消费，要交给后继段重试）、或
  ``DoneWith(succs)``（消费后结束并指定后继，用于"按匹配到的字面量分支"）。
- **语法图（Grammar）**：段 + 每个段的后继段集合。**状态就是候选集合**
  ``frozenset{(seg_id, seg_state)}``：一个字节同时喂给所有候选，失败的候选掉队。这样
  "或"分支（多个可选续写）不需要单独的 NFA 结构，叉开由候选集合天然表达。
- **掩码（mask）**：给定候选集合，逐 token 走一遍字节得到"这一步允许哪些 token"。
  状态可哈希，掩码按状态缓存——同一份语法在所有请求之间共享缓存。

``END`` 是候选里的伪段：它没有字节可消费，出现在候选集合里就表示"语法现在可以结束"，
此时 EOS 进掩码。
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from flashengine.constraints import Utf8Tracker

END = -1

# 段的 step() 返回值哨兵；None 表示该字节非法。
class _Sentinel:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"


DONE = _Sentinel("done")
PENDING = _Sentinel("pending")


class DoneWith(tuple):
    """消费本字节后结束，并显式指定后继段（``DoneWith((seg_id, ...))``）。"""

    __slots__ = ()


class Segment:
    """字节级语法段。状态必须可哈希（掩码缓存以整个候选集合为键）。"""

    def start(self):
        raise NotImplementedError

    def step(self, state, byte):
        raise NotImplementedError

    def opaque(self, state):
        """自由文本区返回 ``(kind, param)``，让掩码整 token 放行；否则 None。"""
        return None

    def accepting(self, state):
        """当前状态能否"不再消费字节"就结束本段。"""
        return False

    def pending_succ(self, state):
        """``PENDING`` 时使用的后继；默认沿用语法图里登记的后继。"""
        return None


class Literal(Segment):
    """固定字面量。"""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = bytes(data)
        if not self.data:
            raise ValueError("Literal 不能是空串")

    def start(self):
        return 0

    def step(self, pos, byte):
        if byte != self.data[pos]:
            return None
        pos += 1
        return DONE if pos == len(self.data) else pos

    def __repr__(self):
        return f"Literal({self.data[:24]!r})"


class AltLiteral(Segment):
    """多个字面量之一；用于"工具名"这类候选集合。

    状态是 ``((literal_index, matched_bytes), ...)``：前缀相同的候选同时存活，某个字节只
    能让其中一部分继续，掉队的自然被淘汰。全部匹配完还能唯一的候选才算完成；如果某候选
    已完成而另一个还能继续（字面量互为前缀），状态里保留已完成候选，等下一个字节再决定
    （不消费即 ``PENDING``，交给后继重试）。
    """

    __slots__ = ("literals", "succs")

    def __init__(self, literals, succs=None):
        self.literals = tuple(bytes(item) for item in literals)
        if not self.literals:
            raise ValueError("AltLiteral 至少需要一个候选")
        if len(set(self.literals)) != len(self.literals):
            raise ValueError("AltLiteral 候选字面量不能重复")
        self.succs = None if succs is None else tuple(tuple(item) for item in succs)
        if self.succs is not None and len(self.succs) != len(self.literals):
            raise ValueError("AltLiteral 的后继列表必须与候选一一对应")

    def start(self):
        return tuple((index, 0) for index in range(len(self.literals)))

    def step(self, state, byte):
        literals = self.literals
        advanced = [
            (index, matched + 1)
            for index, matched in state
            if matched < len(literals[index]) and literals[index][matched] == byte
        ]
        if advanced:
            if len(advanced) == 1:
                index, matched = advanced[0]
                if matched == len(literals[index]):
                    return self._done(index)
            return tuple(advanced)
        complete = [index for index, matched in state if matched == len(literals[index])]
        if complete:
            return PENDING
        return None

    def _done(self, index):
        if self.succs is None:
            return DONE
        return DoneWith(self.succs[index])

    def accepting(self, state):
        return any(matched == len(self.literals[index]) for index, matched in state)

    def pending_succ(self, state):
        if self.succs is None:
            return None
        complete = tuple(
            index for index, matched in state if matched == len(self.literals[index])
        )
        if not complete:
            return None
        merged = []
        for index in complete:
            merged.extend(self.succs[index])
        return tuple(dict.fromkeys(merged))

    def __repr__(self):
        return f"AltLiteral({[item[:16] for item in self.literals]})"


class FreeText(Segment):
    """自由文本区：任意合法 UTF-8 字节，直到 stop 序列出现。

    用于 XML 工具参数值（stop = ``</param>``）。stop 的首字节只作为 stop 的起点出现，
    因此"值里出现 stop"这件事不会产生歧义；``forbidden`` 用来把会破坏外层的字节挡住
    （例如实体起始符 ``&``）。
    """

    __slots__ = ("stop", "forbidden", "_stop0", "_opaque")

    def __init__(self, stop, forbidden=()):
        self.stop = bytes(stop)
        if not self.stop:
            raise ValueError("FreeText 的 stop 不能为空")
        self.forbidden = frozenset(int(byte) for byte in forbidden)
        self._stop0 = self.stop[0]
        self._opaque = frozenset(self.forbidden | {self._stop0})

    def start(self):
        return (0, 0)  # (matched_stop, utf8_pending)

    def step(self, state, byte):
        matched, utf8 = state
        if matched:
            if byte != self.stop[matched]:
                return None
            matched += 1
            return DONE if matched == len(self.stop) else (matched, 0)
        if byte in self.forbidden:
            return None
        if byte == self._stop0:
            return DONE if len(self.stop) == 1 else (1, 0)
        nxt = Utf8Tracker.step(utf8, byte)
        if nxt is None:
            return None
        return (0, nxt)

    def opaque(self, state):
        matched, utf8 = state
        if matched == 0 and utf8 == 0:
            return ("utf8", self._opaque)
        return None

    def __repr__(self):
        return f"FreeText(stop={self.stop!r})"


class NumberText(Segment):
    """数值字面量文本（纯文本协议用），自身吃掉终止符 stop（XML 的 ``</param>``）。

    状态是 ``(数值子状态, stop 已匹配字节数)``：数值只在"已完整"的子状态下才允许把
    stop 的首字节当成终止；数字中途出现 ``<`` 一律非法（不会产出 ``1.2.`` 这类残片）。
    """

    __slots__ = ("integer", "stop", "_stop0")

    _SIGN = 1
    _INT = 2
    _ZERO = 3
    _DOT = 4
    _FRAC = 5
    _EXP_START = 6
    _EXP_SIGN = 7
    _EXP = 8
    _END_OK = frozenset({_ZERO, _INT, _FRAC, _EXP})

    def __init__(self, integer=False, stop=b"</param>"):
        self.integer = bool(integer)
        self.stop = bytes(stop)
        if not self.stop:
            raise ValueError("NumberText 的 stop 不能为空")
        self._stop0 = self.stop[0]

    def start(self):
        return (0, 0)  # (数值子状态, stop 已匹配字节数)

    def step(self, state, byte):
        sub, matched = state
        if matched:
            if byte != self.stop[matched]:
                return None
            matched += 1
            return DONE if matched == len(self.stop) else (sub, matched)
        if byte == self._stop0:
            if sub in self._END_OK:
                return DONE if len(self.stop) == 1 else (sub, 1)
            return None
        advanced = self._number_step(sub, byte)
        if advanced is None:
            return None
        return (advanced, 0)

    def _number_step(self, state, byte):
        if state == 0:
            if byte == 0x2D:  # '-'
                return self._SIGN
            return self._digit(byte)
        if state == self._SIGN:
            return self._digit(byte)
        if state == self._INT:
            if 0x30 <= byte <= 0x39:
                return self._INT
            if not self.integer and byte == 0x2E:  # '.'
                return self._DOT
            if not self.integer and byte in (0x65, 0x45):  # 'e' 'E'
                return self._EXP_START
            return None
        if state == self._ZERO:
            if not self.integer and byte == 0x2E:
                return self._DOT
            if not self.integer and byte in (0x65, 0x45):
                return self._EXP_START
            return None
        if state == self._DOT:
            if 0x30 <= byte <= 0x39:
                return self._FRAC
            return None
        if state == self._FRAC:
            if 0x30 <= byte <= 0x39:
                return self._FRAC
            if byte in (0x65, 0x45):
                return self._EXP_START
            return None
        if state == self._EXP_START:
            if 0x30 <= byte <= 0x39:
                return self._EXP
            if byte in (0x2B, 0x2D):
                return self._EXP_SIGN
            return None
        if state == self._EXP_SIGN:
            if 0x30 <= byte <= 0x39:
                return self._EXP
            return None
        if state == self._EXP:
            if 0x30 <= byte <= 0x39:
                return self._EXP
            return None
        return None

    @staticmethod
    def _digit(byte):
        if 0x30 <= byte <= 0x39:
            return NumberText._ZERO if byte == 0x30 else NumberText._INT
        return None

    def accepting(self, state):
        return state[0] in self._END_OK

    def __repr__(self):
        return f"NumberText(integer={self.integer}, stop={self.stop!r})"


class Grammar:
    """段 + 后继构成的字节级语法图。"""

    def __init__(self, segments, entry, vocab, eos_ids, name="grammar"):
        self.items = [(segment, tuple(succs)) for segment, succs in segments]
        self.vocab = vocab
        self.name = name
        self.entry = tuple(int(item) for item in entry)
        if not self.entry:
            raise ValueError("语法必须有入口段")
        for seg_id, (_, succs) in enumerate(self.items):
            for succ in succs:
                if succ != END and not 0 <= succ < len(self.items):
                    raise ValueError(f"段 {seg_id} 的后继 {succ} 越界")
        eos = sorted({int(item) for item in eos_ids if 0 <= int(item) < vocab.vocab_size})
        if not eos:
            raise ValueError("语法需要至少一个可用的 EOS token，否则无法在语法结束时停止")
        self._eos = frozenset(eos)
        self._eos_np = np.asarray(eos, dtype=np.int64)
        self._masks: "OrderedDict[frozenset, np.ndarray]" = OrderedDict()
        self._mask_bytes = 0
        self._mask_limit = 64 << 20
        self._first_cache: "OrderedDict[frozenset, dict]" = OrderedDict()

    # -- 状态推进 ---------------------------------------------------------

    def initial(self):
        return frozenset(
            (seg_id, self.items[seg_id][0].start()) for seg_id in self.entry
        )

    def _start(self, seg_id):
        return (END, None) if seg_id == END else (seg_id, self.items[seg_id][0].start())

    def _succs_of(self, seg_id, state):
        segment, succs = self.items[seg_id]
        override = segment.pending_succ(state)
        return succs if override is None else override

    def _feed(self, cand, byte, out):
        seg_id, state = cand
        if seg_id == END:
            return
        segment, succs = self.items[seg_id]
        result = segment.step(state, byte)
        if result is None:
            return
        if result is DONE:
            for succ in succs:
                out.add(self._start(succ))
        elif result is PENDING:
            for succ in self._succs_of(seg_id, state):
                if succ == END:
                    continue  # 语法已可结束，多出来的字节只能算非法
                self._feed(self._start(succ), byte, out)
        elif isinstance(result, DoneWith):
            for succ in result:
                out.add(self._start(succ))
        else:
            out.add((seg_id, result))

    def step(self, gstate, byte):
        out = set()
        for cand in gstate:
            self._feed(cand, byte, out)
        return frozenset(out) if out else None

    def accepting(self, gstate):
        for seg_id, state in gstate:
            if seg_id == END:
                return True
            segment, _ = self.items[seg_id]
            if segment.accepting(state):
                if END in self._succs_of(seg_id, state):
                    return True
        return False

    def advance(self, gstate, token_id):
        """按真实生成 token 推进状态；被语法禁止的 token 说明掩码与状态不一致。"""
        token_id = int(token_id)
        if token_id in self._eos:
            return gstate
        raw = self.vocab.bytes[token_id] if 0 <= token_id < self.vocab.vocab_size else b""
        state = gstate
        for byte in raw:
            state = self.step(state, byte)
            if state is None:
                raise ValueError(
                    f"token {token_id}（{raw!r}）不在语法 {self.name} 允许范围内"
                )
        return state

    def is_accepting_token(self, gstate, token_id):
        return int(token_id) in self._eos and self.accepting(gstate)

    # -- 掩码 -------------------------------------------------------------

    def _walk_from(self, gstate, token_id, start):
        raw = self.vocab.bytes[token_id]
        if not raw and start == 0:
            return None  # 补零行/空 token：任何字节序列都不成立，绝不能被掩码放行
        state = gstate
        for byte in raw[start:]:
            state = self.step(state, byte)
            if state is None:
                return None
        return state

    def _token_ok(self, gstate, token_id):
        return self._walk_from(gstate, token_id, 0) is not None

    def mask(self, gstate):
        cached = self._masks.get(gstate)
        if cached is not None:
            self._masks.move_to_end(gstate)
            return cached
        out = np.zeros(self.vocab.vocab_size, dtype=bool)
        opaque = self.opaque(gstate)
        if opaque is not None:
            kind, param = opaque
            base = self.vocab.opaque_allowed(kind, param)
            out |= base
            for token_id in np.nonzero(~base)[0].tolist():
                if self._token_ok(gstate, token_id):
                    out[token_id] = True
        else:
            first = self._first_bytes(gstate)
            buckets = self.vocab.buckets
            for byte, after in first.items():
                for token_id in buckets[byte].tolist():
                    if self._walk_from(after, token_id, 1) is not None:
                        out[token_id] = True
        # EOS 只在语法允许结束时放行：自由文本区的整 token 位图会把 ASCII 形态的特殊
        # token（</s>、<|im_end|>）一并算进去，必须在这里统一清掉。
        out[self._eos_np] = False
        if self.accepting(gstate):
            out[self._eos_np] = True
        if not out.any():
            raise RuntimeError(
                f"语法 {self.name} 在当前状态没有可用 token：{self.describe(gstate)}"
            )
        self._store_mask(gstate, out)
        return out

    def _first_bytes(self, gstate):
        cached = self._first_cache.get(gstate)
        if cached is not None:
            self._first_cache.move_to_end(gstate)
            return cached
        first = {}
        for byte in range(256):
            after = self.step(gstate, byte)
            if after is not None:
                first[byte] = after
        self._first_cache[gstate] = first
        while len(self._first_cache) > 4096:
            self._first_cache.popitem(last=False)
        return first

    def _store_mask(self, gstate, mask):
        self._masks[gstate] = mask
        self._mask_bytes += mask.nbytes
        while self._mask_bytes > self._mask_limit and self._masks:
            _, old = self._masks.popitem(last=False)
            self._mask_bytes -= old.nbytes

    def opaque(self, gstate):
        if len(gstate) != 1:
            return None
        seg_id, state = next(iter(gstate))
        if seg_id == END:
            return None
        return self.items[seg_id][0].opaque(state)

    def describe(self, gstate):
        parts = []
        for seg_id, state in sorted(gstate, key=lambda item: item[0]):
            if seg_id == END:
                parts.append("END")
            else:
                parts.append(f"#{seg_id}:{state!r}")
        return "{" + ", ".join(parts) + "}"

    def reset_cache(self):
        self._masks.clear()
        self._mask_bytes = 0
        self._first_cache.clear()
