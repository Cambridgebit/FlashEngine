"""结构化（约束）解码：把 JSON Schema / 工具调用协议变成采样前的 token 掩码。

对外只暴露两件事：

- :func:`tool_constraint` / :func:`json_constraint`：按工具定义或 JSON Schema 造一个
  :class:`PathConstraint`，挂到请求上；
- :class:`PathConstraint`：引擎每步用 ``mask(state)`` 取允许的 token，提交 token 后
  ``advance(state, token)`` 推进状态，``is_accepting(state)`` 判断语法是否已经完整。

语法对象按 (协议, 工具定义, 词表) 缓存复用：掩码按状态缓存，同一份工具定义在后续请求里
基本不用重算。
"""

from __future__ import annotations

import json
from collections import OrderedDict

from flashengine.structured.machines import (  # noqa: F401  （对外导出便于测试）
    DONE,
    END,
    PENDING,
    AltLiteral,
    FreeText,
    Grammar,
    Literal,
    NumberText,
    Segment,
)
from flashengine.structured.protocols import (  # noqa: F401
    PROTOCOLS,
    build_json_grammar,
    build_tool_grammar,
    tool_pairs,
)
from flashengine.structured.vocab import VocabTable

_VOCAB_TABLES: dict[tuple[int, int], VocabTable] = {}
_GRAMMARS: "OrderedDict[tuple, Grammar]" = OrderedDict()
_GRAMMAR_LIMIT = 16


class PathConstraint:
    """挂在一条请求上的结构化约束（状态由引擎按生成历史推进）。"""

    __slots__ = ("grammar", "name", "description", "_initial")

    def __init__(self, grammar: Grammar, name: str, description: str = ""):
        self.grammar = grammar
        self.name = name
        self.description = description
        self._initial = grammar.initial()

    def initial_state(self):
        return self._initial

    def advance(self, state, token_id):
        return self.grammar.advance(state, token_id)

    def mask(self, state):
        return self.grammar.mask(state)

    def is_accepting(self, state):
        return self.grammar.accepting(state)

    def snapshot(self):
        return {
            "enabled": True,
            "mode": "structured",
            "type": self.name,
            "detail": self.description,
        }


def vocab_table(tokenizer, vocab_size=None):
    """按 tokenizer 实例缓存字节视图（词表 13 万时构建约 0.3s，不能每请求重来）。"""
    key = (id(tokenizer), int(vocab_size or 0))
    table = _VOCAB_TABLES.get(key)
    if table is None:
        table = VocabTable(tokenizer, vocab_size=vocab_size)
        _VOCAB_TABLES[key] = table
    return table


def _cached_grammar(key, factory):
    grammar = _GRAMMARS.get(key)
    if grammar is not None:
        _GRAMMARS.move_to_end(key)
        return grammar
    grammar = factory()
    _GRAMMARS[key] = grammar
    while len(_GRAMMARS) > _GRAMMAR_LIMIT:
        _GRAMMARS.popitem(last=False)
    return grammar


def tool_constraint(tools, protocol, tokenizer, eos_ids, *, choice=None,
                    vocab_size=None, max_tools=64):
    """按 OpenAI ``tools`` 定义造工具调用约束；``choice`` 指定只允许某一个函数。"""
    if protocol not in PROTOCOLS:
        raise ValueError(f"未知的工具协议 {protocol!r}，可选：{', '.join(PROTOCOLS)}")
    if len(tools) > max_tools:
        raise ValueError(f"工具数量 {len(tools)} 超过上限 {max_tools}")
    table = vocab_table(tokenizer, vocab_size)
    key = (
        "tools",
        protocol,
        choice,
        json.dumps(tools, sort_keys=True, ensure_ascii=False),
        id(table),
    )
    grammar = _cached_grammar(
        key,
        lambda: build_tool_grammar(tools, protocol, table, eos_ids, choice=choice),
    )
    names = [name for name, _ in tool_pairs(tools, choice)]
    detail = f"{protocol} / {len(names)} tool(s) / choice={choice or 'required'}"
    return PathConstraint(grammar, f"tool_call:{protocol}", detail)


def json_constraint(schema, tokenizer, eos_ids, *, vocab_size=None, name="response"):
    """按 JSON Schema 造"整个输出是一个 JSON 值"的约束（OpenAI response_format）。"""
    if not isinstance(schema, dict):
        raise ValueError("response_format.json_schema.schema 必须是 JSON 对象")
    table = vocab_table(tokenizer, vocab_size)
    key = (
        "json",
        name,
        json.dumps(schema, sort_keys=True, ensure_ascii=False),
        id(table),
    )
    grammar = _cached_grammar(
        key, lambda: build_json_grammar(schema, table, eos_ids, name=name)
    )
    return PathConstraint(grammar, f"json_schema:{name}", "response_format=json_schema")


__all__ = [
    "PathConstraint",
    "PROTOCOLS",
    "Grammar",
    "Segment",
    "Literal",
    "AltLiteral",
    "FreeText",
    "NumberText",
    "VocabTable",
    "vocab_table",
    "tool_constraint",
    "json_constraint",
    "END",
    "DONE",
    "PENDING",
]
