"""词表字节索引：结构化约束的掩码推导只依赖这里准备好的三张表。

- ``bytes``：每个 token 的原始字节。ByteLevel BPE 的 token 文本是字节的"可见化"表示，
  必须用 :func:`flashengine.constraints.token_bytes` 还原，不能直接 utf-8 编码；
- ``buckets``：按首字节分桶的 token id，掩码枚举时先按首字节筛掉绝大多数候选；
- ``opaque_allowed``："自由文本区"里可以整 token 放行的词表位（见下）。

自由文本区（JSON 字符串内容、工具参数值）占词表的大部分，逐 token 走字节会慢到不可接受。
这类区域有一个共同点：只要 token 的字节在区域内合法，推进后区域状态不变（字符串内容仍是
字符串内容）。因此把"合法"编译成词表位图一次性算好，之后每个状态只有少量特殊 token
（含引号/反斜杠/控制字符/非完整 UTF-8 的）需要逐字节精算。
"""

from __future__ import annotations

import numpy as np

from flashengine.constraints import token_bytes


def _byte_level_decoder(backend):
    """判断后端 tokenizer 的解码器是不是 ByteLevel（Sequence 里嵌套也要认）。"""
    decoder = getattr(backend, "decoder", None)
    seen = 0
    while decoder is not None and seen < 8:
        name = type(decoder).__name__.lower()
        if "bytelevel" in name:
            return True
        if "sequence" in name:
            inner = getattr(decoder, "decoders", None) or ()
            decoder = inner[0] if inner else None
        else:
            return False
        seen += 1
    return False


class VocabTable:
    """一个 tokenizer 的字节视图；与语法无关，可按 tokenizer 缓存复用。"""

    def __init__(self, tokenizer, *, vocab_size=None):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if not _byte_level_decoder(backend):
            decoder = type(getattr(backend, "decoder", None)).__name__
            raise ValueError(
                "结构化约束目前只支持 ByteLevel tokenizer（Qwen3 / MiniCPM 系）；"
                f"当前 decoder={decoder}，请换用 --tool-protocol none"
            )
        defined = len(tokenizer)
        size = int(vocab_size or defined)
        if size < defined:
            raise ValueError(
                f"vocab_size={size} 小于 tokenizer 实际长度 {defined}，特殊 token 会被截断"
            )
        added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
        table = [b""] * size
        for token_id in range(defined):
            entry = added.get(token_id)
            if entry is not None:
                # added token 的 content 是字面文本，不是 byte-level 可见化表示。
                table[token_id] = entry.content.encode("utf-8")
            else:
                table[token_id] = token_bytes(tokenizer.convert_ids_to_tokens(token_id))
        self.bytes = table
        self.vocab_size = size
        self.defined = defined
        specials = set()
        for token_id in getattr(tokenizer, "all_special_ids", ()) or ():
            if isinstance(token_id, int):
                specials.add(token_id)
        for token_id, entry in added.items():
            if getattr(entry, "special", False):
                specials.add(int(token_id))
        # 自由文本区不允许"整 token 放行"特殊 token（</s>、<|im_end|>、</param>…）：
        # 它们要么是结束标记、要么是外层语法的一部分，必须逐字节走精确判定。
        self.special_ids = frozenset(specials)
        self._buckets = None
        self._utf8_cache: dict[frozenset[int], np.ndarray] = {}
        self._byte_set_cache: dict[frozenset[int], np.ndarray] = {}
        self._byte_present = None

    @property
    def buckets(self) -> list[np.ndarray]:
        """按首字节分桶的 token id；空 token（补零行）不进入任何桶。"""
        if self._buckets is None:
            groups: list[list[int]] = [[] for _ in range(256)]
            for token_id, raw in enumerate(self.bytes):
                if raw:
                    groups[raw[0]].append(token_id)
            self._buckets = [
                np.asarray(group, dtype=np.int32) for group in groups
            ]
        return self._buckets

    def _present(self) -> np.ndarray:
        if self._byte_present is None:
            present = np.zeros((256, self.vocab_size), dtype=bool)
            for token_id, raw in enumerate(self.bytes):
                for byte in raw:
                    present[byte, token_id] = True
            self._byte_present = present
        return self._byte_present

    def bytes_allowed(self, allowed: frozenset[int]) -> np.ndarray:
        """返回"所有字节都在 allowed 里"的词表位图（空 token 一律不准）。"""
        cached = self._byte_set_cache.get(allowed)
        if cached is not None:
            return cached
        if not allowed:
            mask = np.zeros(self.vocab_size, dtype=bool)
        else:
            present = self._present()
            complement = frozenset(set(range(256)) - set(allowed))
            if not complement:
                mask = np.array([bool(raw) for raw in self.bytes], dtype=bool)
            else:
                blocked = np.any(present[sorted(complement)], axis=0)
                mask = ~blocked
                mask &= np.asarray([bool(raw) for raw in self.bytes], dtype=bool)
        self._byte_set_cache[allowed] = mask
        return mask

    def utf8_allowed(self, forbidden: frozenset[int]) -> np.ndarray:
        """返回"是完整 UTF-8 序列且不含 forbidden 字节"的词表位图。

        用于 JSON 字符串内容、XML 参数值这类自由文本区：合法 token 推进后区域状态不变，
        所以可以整 token 放行，不必逐字节走语法。
        """
        cached = self._utf8_cache.get(forbidden)
        if cached is not None:
            return cached
        mask = np.zeros(self.vocab_size, dtype=bool)
        for token_id, raw in enumerate(self.bytes):
            if not raw or token_id in self.special_ids:
                continue
            if any(byte in forbidden for byte in raw):
                continue
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            mask[token_id] = True
        self._utf8_cache[forbidden] = mask
        return mask

    def opaque_allowed(self, kind: str, param: frozenset[int]) -> np.ndarray:
        """自由文本区的整 token 放行位图；kind 由语法机的 :meth:`Segment.opaque` 给出。"""
        if kind == "utf8":
            return self.utf8_allowed(param)
        if kind == "bytes":
            return self.bytes_allowed(param)
        raise ValueError(f"未知的自由文本区类型：{kind}")
