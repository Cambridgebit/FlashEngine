"""工具调用协议 → 字节级生成语法。

支持三种协议（都只约束"调用的形状"：名字必须是给定工具之一，参数按工具 schema 生成）：

- ``xml_function``：MiniCPM5 系模板的 XML 形态，
  ``<function name="f"><param name="p">v</param></function>``，多个调用换行分隔；
- ``hermes``：Qwen3 / Hermes 系，
  ``<tool_call>\\n{"name":"f","arguments":{...}}\\n</tool_call>``；
- ``deepseek_v3``：DeepSeek V3.1+ 的专用 token 形态，
  ``<｜｜tool▁calls▁begin｜｜><｜｜tool▁call▁begin｜｜>function<｜｜tool▁sep｜｜>f...``。

另外提供 ``response_format`` 用的纯 JSON 语法（整个输出就是一个符合 schema 的 JSON 值）。
"""

from __future__ import annotations

from flashengine.structured.json_schema import JsonValue, compile_schema, text_literals
from flashengine.structured.machines import (
    END,
    AltLiteral,
    FreeText,
    Grammar,
    Literal,
    NumberText,
)

PROTOCOLS = ("xml_function", "hermes", "deepseek_v3")

_CALLS_BEGIN = "<｜｜tool▁calls▁begin｜｜>"
_CALL_BEGIN = "<｜｜tool▁call▁begin｜｜>"
_SEP = "<｜｜tool▁sep｜｜>"
_CALL_END = "<｜｜tool▁call▁end｜｜>"
_CALLS_END = "<｜｜tool▁calls▁end｜｜>"


class _Builder:
    """段表 + 后继边；返回段 id，后继可以先建后回填。"""

    def __init__(self):
        self.items = []

    def add(self, segment, succs=()):
        self.items.append((segment, tuple(succs)))
        return len(self.items) - 1

    def set_succ(self, seg_id, succs):
        self.items[seg_id] = (self.items[seg_id][0], tuple(succs))


def tool_pairs(tools, choice=None):
    """把 OpenAI ``tools`` 列表规整成 ``[(name, parameters_schema)]``。"""
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools 必须是非空数组")
    pairs = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{index}] 必须是对象")
        kind = tool.get("type", "function")
        if kind != "function":
            raise ValueError(f"tools[{index}]：只支持 type=\"function\"，收到 {kind!r}")
        function = tool.get("function") or {}
        if not isinstance(function, dict):
            raise ValueError(f"tools[{index}].function 必须是对象")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tools[{index}]：function.name 必须是非空字符串")
        parameters = function.get("parameters") or {
            "type": "object", "properties": {}
        }
        if not isinstance(parameters, dict):
            raise ValueError(f"tools[{index}]：function.parameters 必须是对象")
        pairs.append((name, parameters))
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"tools 里函数名重复：{names}")
    if choice is not None:
        if choice not in names:
            raise ValueError(f"tool_choice 指定的函数 {choice!r} 不在 tools 里")
        pairs = [(name, schema) for name, schema in pairs if name == choice]
    return pairs


# ---------------------------------------------------------------------------
# XML（MiniCPM5）
# ---------------------------------------------------------------------------

def _xml_value_alternatives(builder, node, next_succs):
    """XML 参数值的一/多个候选段；所有段的"值结束"后继都接到 ``next_succs``。

    字符串与数值段自己吃掉终止符 ``</param>``（匹配终止符本身就是它们的状态）；枚举、
    布尔、JSON 值不带终止符，统一再挂一个字面量 ``</param>``。
    """
    literals = text_literals(node.raw or {})
    if literals is not None:
        return tuple(
            builder.add(Literal(text + b"</param>"), next_succs)
            for text in literals
        )
    if node.kind == "literal":
        return tuple(
            builder.add(Literal(text + b"</param>"), next_succs)
            for text in node.literals
        )
    if node.kind == "string":
        # 值里不能出现 XML 实体起始符；< 留给 </param> 终止符，避免歧义。
        return (builder.add(FreeText(b"</param>", forbidden=b"&"), next_succs),)
    if node.kind == "number":
        return (builder.add(NumberText(integer=node.integer), next_succs),)
    if node.kind in ("object", "array"):
        close = builder.add(Literal(b"</param>"), next_succs)
        return (builder.add(JsonValue(node), (close,)),)
    if node.kind == "union":
        # XML 参数值不支持取并集，退化成自由文本（解析时按 schema 再转类型）。
        return (builder.add(FreeText(b"</param>", forbidden=b"&"), next_succs),)
    raise ValueError(f"不支持的 XML 参数类型：{node.kind}")


def _xml_tool_successors(builder, node, call_end):
    """按参数声明顺序铺参数链；必填参数不能跳过，可选参数可以跳过。"""
    memo = {}

    def chain(index):
        if index in memo:
            return memo[index]
        if index >= len(node.props):
            end = builder.add(Literal(b"</function>"))
            builder.set_succ(end, call_end)
            memo[index] = (end,)
            return memo[index]
        succs = []
        for candidate in range(index, node.cand_end[index] + 1):
            key, child = node.props[candidate]
            literal = builder.add(
                Literal(b'<param name="' + key.encode("utf-8") + b'">')
            )
            builder.set_succ(literal, _xml_value_alternatives(
                builder, child, chain(candidate + 1)
            ))
            succs.append(literal)
        if not node.required_tail[index]:
            end = builder.add(Literal(b"</function>"))
            builder.set_succ(end, call_end)
            succs.append(end)
        memo[index] = tuple(succs)
        return memo[index]

    return chain(0)


def _xml_grammar(pairs):
    builder = _Builder()
    entries = {}
    for name, _ in pairs:
        entries[name] = builder.add(
            Literal(b'<function name="' + name.encode("utf-8") + b'">')
        )
    # 多个调用之间允许换行；调用块也可以直接结束。
    separator = builder.add(Literal(b"\n"))
    call_end = tuple(entries.values()) + (separator, END)
    builder.set_succ(separator, call_end)
    for name, schema in pairs:
        node = compile_schema(schema, name=name)
        successors = _xml_tool_successors(builder, node, call_end)
        builder.set_succ(entries[name], successors)
    return builder.items, tuple(entries.values())


# ---------------------------------------------------------------------------
# Hermes / Qwen3
# ---------------------------------------------------------------------------

def _hermes_grammar(pairs):
    builder = _Builder()
    head = builder.add(Literal(b'<tool_call>\n{"name":"'))
    entries = []
    for name, schema in pairs:
        node = compile_schema(schema, name=name)
        arguments = builder.add(JsonValue(node))
        close = builder.add(Literal(b"}\n</tool_call>"))
        builder.set_succ(close, (head, END))
        builder.set_succ(arguments, (close,))
        entries.append(
            builder.add(
                Literal(name.encode("utf-8") + b'","arguments":'), (arguments,)
            )
        )
    builder.set_succ(head, tuple(entries))
    return builder.items, (head,)


# ---------------------------------------------------------------------------
# DeepSeek V3.1+ 专用 token 形态
# ---------------------------------------------------------------------------

def _deepseek_grammar(pairs):
    builder = _Builder()
    calls_begin = builder.add(Literal(_CALLS_BEGIN.encode("utf-8")))
    call_head = builder.add(
        Literal((_CALL_BEGIN + "function" + _SEP).encode("utf-8"))
    )
    calls_end = builder.add(Literal(_CALLS_END.encode("utf-8")), (END,))
    call_tail = builder.add(Literal(("\n```" + _CALL_END).encode("utf-8")))
    builder.set_succ(call_tail, (call_head, calls_end))
    entries = []
    for name, schema in pairs:
        node = compile_schema(schema, name=name)
        arguments = builder.add(JsonValue(node))
        builder.set_succ(arguments, (call_tail,))
        entries.append(
            builder.add(
                Literal(name.encode("utf-8") + b"\n```json\n"), (arguments,)
            )
        )
    builder.set_succ(call_head, tuple(entries))
    builder.set_succ(calls_begin, (call_head,))
    return builder.items, (calls_begin,)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def build_tool_grammar(tools, protocol, vocab, eos_ids, *, choice=None, name=None):
    if protocol not in PROTOCOLS:
        raise ValueError(
            f"未知的工具协议 {protocol!r}，可选：{', '.join(PROTOCOLS)}"
        )
    pairs = tool_pairs(tools, choice)
    if protocol == "xml_function":
        items, entry = _xml_grammar(pairs)
    elif protocol == "hermes":
        items, entry = _hermes_grammar(pairs)
    else:
        items, entry = _deepseek_grammar(pairs)
    return Grammar(
        items, entry, vocab, eos_ids, name=name or f"tools:{protocol}"
    )


def build_json_grammar(schema, vocab, eos_ids, *, name="response"):
    builder = _Builder()
    node = compile_schema(schema, name=name)
    segment = builder.add(JsonValue(node), (END,))
    return Grammar(builder.items, (segment,), vocab, eos_ids, name=f"json:{name}")
