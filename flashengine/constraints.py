"""禁词约束：编译成词表掩码，在采样之前禁止，因此与随机采样和投机解码正交。

与"采样后检查文本、触词就换下一名"的写法相比，掩码在 logits 上做，位置在采样之前：

- 掩码与 temperature/top-p/top-k 无关，随机采样同样被约束；
- 投机 verify 的每一行掩码可由草稿 token 预计算（行 j 的状态 = 已提交前缀 + d1..d{j-1}），
  因此可以在图内以 ``bias_table[state]`` 的形式一次 gather 完成，无需把 logits 搬回主机；
- 判定使用 UTF-8 字节子串，覆盖跨 token 的字节片段；实际解码输出还需在发送边界检查。

模块导入期只依赖标准库；张量构造在 :meth:`BannedWordMasker.bias_matrix` 内部惰性导入 torch。
"""

DEFAULT_BANNED_WORDS = ("航空", "秘密")

_NEG_INF = float("-inf")


def _gpt2_byte_decoder():
    """GPT-2/Qwen 系 byte-level BPE 的可见字符 → 原始字节反查表。"""
    direct = (list(range(ord("!"), ord("~") + 1))
              + list(range(0xA1, 0xAC + 1))
              + list(range(0xAE, 0xFF + 1)))
    visible = list(direct)
    extra = 0
    for byte in range(256):
        if byte not in direct:
            direct.append(byte)
            visible.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(direct, visible)}


_BYTE_DECODER = _gpt2_byte_decoder()


def normalize_banned_words(words):
    """校验并去重词库；是否生效由 enable_constraints 单独控制。"""
    if not isinstance(words, (list, tuple)):
        raise ValueError("banned_words 必须是非空字符串组成的 JSON 数组")
    if any(not isinstance(word, str) or not word.strip() for word in words):
        raise ValueError("banned_words 的每个词必须为非空字符串")
    return tuple(dict.fromkeys(words))


def resolve_constraint_words(enabled, words, **compat):
    """校验启停布尔值；开启时空词库直接报错，不用清空数组隐式关闭。"""
    if type(enabled) is not bool:
        raise ValueError("enable_constraints 必须是 JSON 布尔值 true 或 false")
    normalized = normalize_banned_words(words)
    if not enabled:
        return ()
    if not normalized:
        raise ValueError("开启约束时 banned_words 至少包含一个词；关闭请使用 enable_constraints=false")
    validate_constraint_mode(normalized, **compat)
    return normalized


def validate_constraint_mode(words, *, tensor_parallel_size=1, spec_model="", temperature=0.0):
    """掩码路线只保留并行约束；随机采样与投机解码都允许，参数保留以兼容调用方。"""
    if words and tensor_parallel_size != 1:
        raise ValueError("禁词约束当前只支持 tensor_parallel_size=1")
    return None


def token_bytes(token):
    """把 tokenizer 给出的 token 文本还原为原始字节串。

    byte-level BPE 的 token 是字节的"可见化"表示，需要按反查表还原；不在表内的字符
    （sentencepiece、中文直出等）本就是原文，直接取 UTF-8 字节。
    """
    if isinstance(token, bytes):
        return token
    if not isinstance(token, str):
        raise ValueError(f"tokenizer 没有给出 token 文本（id 越界或词表损坏）：{token!r}")
    out = bytearray()
    for char in token:
        byte = _BYTE_DECODER.get(char)
        if byte is None:
            out += char.encode("utf-8")
        else:
            out.append(byte)
    return bytes(out)


class Utf8Tracker:
    """追踪字节流的 UTF-8 合法性：状态 = 还差几个续字节（0 表示可接受新字符）。

    仅用于约束模式的输出质量保护：禁词掩码把自然续写堵死后，模型可能选到"半个汉字"的
    碎片 token 而产出非法字节（实测表现为乱码替换符），这里把这类 token 一并禁止。
    """

    # 0=完整，1/2/3=剩余普通续字节；4..7 是首个续字节有范围约束的情况。
    STATES = 8

    @staticmethod
    def step(state, byte):
        """返回下一个状态；拒绝过长编码、代理项和超出 Unicode 范围的编码。"""
        if state == 0:
            if byte < 0x80: return 0
            if 0xC2 <= byte <= 0xDF: return 1
            if byte == 0xE0: return 4
            if byte == 0xED: return 5
            if 0xE1 <= byte <= 0xEF: return 2
            if byte == 0xF0: return 6
            if byte == 0xF4: return 7
            if 0xF1 <= byte <= 0xF3: return 3
            return None
        if state == 4: return 1 if 0xA0 <= byte <= 0xBF else None
        if state == 5: return 1 if 0x80 <= byte <= 0x9F else None
        if state == 6: return 2 if 0x90 <= byte <= 0xBF else None
        if state == 7: return 2 if 0x80 <= byte <= 0x8F else None
        return state - 1 if 0x80 <= byte <= 0xBF else None


class BannedWordAutomaton:
    """把禁词表编译成字节级 Aho-Corasick 自动机；命中任一禁词即判定不允许该 token。

    默认精确匹配；可选忽略 ASCII 空格、制表符及换行。该选项不覆盖全角空格等 Unicode 空白。
    """

    IGNORED_BYTES = frozenset({0x20, 0x09, 0x0A, 0x0D})

    def __init__(self, words, ignore_spaces=False):
        self.words = normalize_banned_words(words)
        # 忽略空白能堵住"航 空"这类插空格绕过，但会把模型逼到劣质 token 上（实测出现乱码），
        # 因此默认关闭；需要更严格的场景显式打开并接受输出质量下降。
        self.ignore_spaces = bool(ignore_spaces)
        self.goto = [{}]
        self.fail = [0]
        self.terminal = [False]
        for word in self.words:
            state = 0
            for byte in word.encode("utf-8"):
                nxt = self.goto[state].get(byte)
                if nxt is None:
                    nxt = len(self.goto)
                    self.goto[state][byte] = nxt
                    self.goto.append({})
                    self.fail.append(0)
                    self.terminal.append(False)
                state = nxt
            self.terminal[state] = True
        self._build_fail_links()

    def _build_fail_links(self):
        queue = []
        for byte, nxt in self.goto[0].items():
            self.fail[nxt] = 0
            queue.append((nxt, byte))
        head = 0
        while head < len(queue):
            state, _ = queue[head]
            head += 1
            self.terminal[state] = self.terminal[state] or self.terminal[self.fail[state]]
            for byte, nxt in self.goto[state].items():
                fallback = self.fail[state]
                while fallback and byte not in self.goto[fallback]:
                    fallback = self.fail[fallback]
                target = self.goto[fallback].get(byte, 0)
                self.fail[nxt] = target if target != nxt else 0
                queue.append((nxt, byte))

    @property
    def num_states(self):
        return len(self.goto)

    def step(self, state, byte):
        """消费一个字节；命中禁词返回 None，否则返回下一个状态。空白字节不改变状态。"""
        if self.ignore_spaces and byte in self.IGNORED_BYTES:
            return state
        while state and byte not in self.goto[state]:
            state = self.fail[state]
        state = self.goto[state].get(byte, 0)
        return None if self.terminal[state] else state

    def initial(self):
        return 0

    def allowed_from(self, state, token_bytes_table, token_ids=None):
        """给出一批 token 是否可用；token_ids 为 None 时遍历整个词表。"""
        ids = range(len(token_bytes_table)) if token_ids is None else token_ids
        allowed = []
        for token_id in ids:
            current = state
            for byte in token_bytes_table[token_id]:
                current = self.step(current, byte)
                if current is None:
                    break
            allowed.append(current is not None)
        return allowed


class BannedWordMasker:
    """持有自动机、token 字节表与状态→掩码行缓存，供引擎在采样前取偏置。"""

    def __init__(self, words, token_bytes_table, always_allow=(), structural=(), blocked=(),
                 ignore_spaces=False, utf8_safe=False):
        self.automaton = BannedWordAutomaton(words, ignore_spaces=ignore_spaces)
        # 状态编码 = 禁词自动机状态 * UTF8_STATES + UTF-8 待续字节数（utf8_safe 时）。
        self.utf8_safe = bool(utf8_safe)
        self.token_bytes_table = list(token_bytes_table)
        # 稠密表占用随状态数 × 词表增长；拒绝过大词库，避免耗尽设备内存。
        if self.num_states * len(self.token_bytes_table) > 20_000_000:
            raise ValueError("禁词词库过大：掩码表超过 2000 万项，请缩小词库")
        self.always_allow = frozenset(int(token_id) for token_id in always_allow)
        # 结构 token（<|im_end|> 等）不推进自动机状态，否则其字面文本会把状态卡在词中间。
        self.structural = frozenset(int(token_id) for token_id in structural) | self.always_allow
        # 未定义的补零行（词表尾部）一律禁止，避免 argmax 落到无字节的空 token 上绕过掩码。
        self.blocked = frozenset(int(token_id) for token_id in blocked)
        self._rows = None
        self._cached = None
        self._next_cached = None

    @classmethod
    def from_tokenizer(cls, tokenizer, words, vocab_size=None, always_allow=(),
                       ignore_spaces=False, utf8_safe=False):
        """从 HF tokenizer 构造：逐 id 取 token 文本再还原原始字节。

        宽度取 ``max(vocab_size, len(tokenizer))``：``tokenizer.vocab_size`` 不含 added
        token（Qwen3 上是 151643 vs 实际 151669），用它会把 EOS 截到表外。未定义的补零
        行（Qwen3 的 151669..151935）显式禁止，不能当作"空字节 = 永远允许"。
        """
        backend = getattr(tokenizer, "backend_tokenizer", None)
        decoder = getattr(backend, "decoder", None)
        if type(decoder).__name__.lower() != "bytelevel":
            raise ValueError("禁词掩码目前仅支持已验证的 ByteLevel tokenizer")
        defined = len(tokenizer)
        size = int(vocab_size or defined)
        if size < defined:
            raise ValueError(f"vocab_size={size} 小于 tokenizer 词表 {defined}，特殊 token 会被截断")
        added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
        table = [b""] * size
        for token_id in range(defined):
            entry = added.get(token_id)
            if entry is not None:
                # added token 的 content 是字面文本，不是 byte-level 可见化表示。
                table[token_id] = entry.content.encode("utf-8")
                continue
            table[token_id] = token_bytes(tokenizer.convert_ids_to_tokens(token_id))
        stop_ids = set(int(token_id) for token_id in always_allow)
        for attribute in ("eos_token_id", "pad_token_id"):
            value = getattr(tokenizer, attribute, None)
            for token_id in (value if isinstance(value, (list, tuple)) else (value,)):
                if isinstance(token_id, int) and token_id >= 0:
                    stop_ids.add(token_id)
        return cls(words, table, always_allow=stop_ids,
                   structural=set(getattr(tokenizer, "all_special_ids", ())),
                   blocked=set(range(defined, size)),
                   ignore_spaces=ignore_spaces, utf8_safe=utf8_safe)

    @property
    def num_states(self):
        return self.automaton.num_states * (Utf8Tracker.STATES if self.utf8_safe else 1)

    def _split(self, state):
        if not self.utf8_safe:
            return state, 0
        return divmod(state, Utf8Tracker.STATES)

    def _join(self, ac_state, u8_state):
        if not self.utf8_safe:
            return ac_state
        return ac_state * Utf8Tracker.STATES + u8_state

    def _walk(self, ac_state, u8_state, token_id):
        """沿一个 token 的字节推进两个状态；任一违规返回 None。"""
        if token_id in self.blocked:
            return None
        if token_id in self.always_allow and self.utf8_safe and u8_state != 0:
            return None
        if token_id in self.structural:
            return self._join(ac_state, u8_state)
        for byte in self.token_bytes_table[token_id]:
            ac_state = self.automaton.step(ac_state, byte)
            if ac_state is None:
                return None
            if self.utf8_safe:
                u8_state = Utf8Tracker.step(u8_state, byte)
                if u8_state is None:
                    return None
        return self._join(ac_state, u8_state)

    def initial_states(self, count):
        return [self.automaton.initial()] * count

    def snapshot(self):
        return {"enabled": True, "banned_words": list(self.automaton.words),
                "configured_banned_words": list(self.automaton.words),
                "mode": "token_mask", "selection_backend": "logits_bias"}

    def advance(self, state, token_id):
        """提交一个已生成 token 后的新状态；结构 token 不推进状态。"""
        ac_state, u8_state = self._split(state)
        advanced = self._walk(ac_state, u8_state, token_id)
        if advanced is None:
            raise RuntimeError("提交了被禁词约束禁止的 token，掩码与状态推进不一致")
        return advanced

    def _build_rows(self):
        """为每个状态预生成一行允许标记；状态数 = 1 + 各禁词不重复字节前缀数。

        按字节建 trie，"航空""秘密" 两个词共 13 个状态（1 + 3+3 + 3+3），
        对应 bias 表 13 x 词表，FP16 下约 3.9 MB，一次性上设备后无需再传。
        """
        rows = []
        for state in range(self.num_states):
            ac_state, u8_state = self._split(state)
            row = []
            for token_id, raw in enumerate(self.token_bytes_table):
                if token_id in self.always_allow:
                    row.append(not self.utf8_safe or u8_state == 0)
                    continue
                if token_id in self.blocked:
                    row.append(False)
                    continue
                if token_id in self.structural:
                    row.append(True)
                    continue
                current_ac, current_u8 = ac_state, u8_state
                ok = True
                for byte in raw:
                    current_ac = self.automaton.step(current_ac, byte)
                    if current_ac is None:
                        ok = False
                        break
                    if self.utf8_safe:
                        current_u8 = Utf8Tracker.step(current_u8, byte)
                        if current_u8 is None:
                            ok = False
                            break
                row.append(ok)
            rows.append(row)
        self._rows = rows
        return rows

    def bias_matrix(self, device="cpu", dtype=None):
        """返回 [状态数, 词表] 的加性偏置：允许 0，禁止 -inf。

        传进图后每步只需 ``logits += bias[row_state]``，不必把掩码或 logits 搬来搬去。
        表一律先在 CPU 上一次性构造再整体搬到设备：引擎把默认设备设成了加速卡，
        逐元素写设备张量会变成百万级标量算子。
        """
        import torch

        if self._rows is None:
            self._build_rows()
        if self._cached is not None and self._cached[0] == str(device) and (
                dtype is None or self._cached[1] == dtype):
            return self._cached[2]
        dtype = dtype or torch.float16
        allowed = torch.tensor(self._rows, dtype=torch.bool, device="cpu")
        bias = torch.zeros(allowed.shape, dtype=torch.float32, device="cpu")
        bias.masked_fill_(~allowed, _NEG_INF)
        bias = bias.to(dtype=dtype, device=device)
        self._cached = (str(device), dtype, bias)
        return bias

    def next_state_table(self, device="cpu"):
        """返回 [状态数, 词表] 的下一状态表；-1 表示该 token 会凑出禁词。

        投机路径用它把"草稿 token → 行状态"的推进放到图内：按 ``state * 词表 + token``
        展平后取一行，每步只搬 bs 个整数。同样先在 CPU 构造再整体搬运。
        """
        import torch

        if self._rows is None:
            self._build_rows()
        if self._next_cached is not None and self._next_cached[0] == str(device):
            return self._next_cached[1]
        size = len(self._rows[0])
        states = self.num_states
        table = [[-1] * size for _ in range(states)]
        for state in range(states):
            row = table[state]
            ac_state, u8_state = self._split(state)
            for token_id in range(size):
                if token_id in self.structural:
                    advanced = self._walk(ac_state, u8_state, token_id)
                    if advanced is not None:
                        row[token_id] = advanced
                    continue
                if token_id in self.blocked:
                    continue                       # 保持 -1：禁止
                advanced = self._walk(ac_state, u8_state, token_id)
                if advanced is not None:
                    row[token_id] = advanced
        flat = torch.tensor(table, dtype=torch.int32, device="cpu").reshape(-1).to(device)
        self._next_cached = (str(device), flat)
        return flat

    def row_states(self, prefix_states, tokens):
        """把一串待提交 token 依次推进；遇到被禁止的 token 就停在那里不再前进。

        投机的第 j 行掩码假设"前 j 个草稿都被接受"，因此草稿里一旦出现被禁 token，
        后续行沿用最后一个合法状态即可——那些行的 argmax 本来就会被丢弃。
        """
        states = list(prefix_states)
        for token_id in tokens:
            try:
                states.append(self.advance(states[-1], token_id))
            except RuntimeError:
                states.append(states[-1])
        return states
