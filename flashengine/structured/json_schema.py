"""JSON Schema（实用子集）→ 字节级生成语法。

设计取舍（与 vLLM/xgrammar 一类实现同向）：生成"规范形状"的 JSON，而不是穷举 schema 允许
的所有写法——对象按 ``properties`` 声明顺序出键，必填键必须出现、可选键可以跳过，键集合
之外不允许额外键；字符串/数值紧凑输出、不带空白；数组按 ``items`` 生成，``minItems`` /
``maxItems`` 生效。``pattern`` / ``minLength`` / ``format`` / ``minimum`` 等"内容约束"忽略，
它们不影响产出的 JSON 是否合法。

不支持的结构在编译期就明确报错（而不是生成期静默放宽）：任意 JSON（空 schema）、
``additionalProperties`` 为真的对象、``anyOf`` 分支首字节重叠、递归 ``$ref``。
"""

from __future__ import annotations

import json

from flashengine.constraints import Utf8Tracker
from flashengine.structured.machines import DONE, PENDING, DoneWith, Segment

# JSON 字符串内容里不允许直接出现的字节：控制字符、引号、反斜杠（后两者另走语法分支）。
_STRING_FORBIDDEN = frozenset(range(0x20)) | {0x22, 0x5C}

_HEX = frozenset(b"0123456789abcdefABCDEF")


class Node:
    """编译后的 schema 节点；状态机里按对象身份判等（同一份 schema 只编译一次）。"""

    __slots__ = (
        "kind", "label", "start_bytes", "props", "key_bytes", "cand_end",
        "required_tail", "items", "min_items", "max_items", "literals",
        "branches", "integer", "raw",
    )

    def __init__(
        self, kind, *, label="", start_bytes=frozenset(), props=(), key_bytes=(),
        cand_end=(), required_tail=(), items=None, min_items=0, max_items=0,
        literals=(), branches=(), integer=False, raw=None,
    ):
        self.raw = raw
        self.kind = kind
        self.label = label
        self.start_bytes = frozenset(start_bytes)
        self.props = tuple(props)
        self.key_bytes = tuple(key_bytes)
        self.cand_end = tuple(cand_end)
        self.required_tail = tuple(required_tail)
        self.items = items
        self.min_items = int(min_items)
        self.max_items = int(max_items)
        self.literals = tuple(literals)
        self.branches = tuple(branches)
        self.integer = bool(integer)

    def __repr__(self):
        return f"Node({self.kind}{':' + self.label if self.label else ''})"


# ---------------------------------------------------------------------------
# schema → Node
# ---------------------------------------------------------------------------

_IGNORED_KEYWORDS = frozenset({
    "title", "description", "default", "examples", "example", "$comment",
    "deprecated", "format", "pattern", "minLength", "maxLength", "minimum",
    "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minProperties",
    "maxProperties", "uniqueItems", "readOnly", "writeOnly", "$schema", "$id",
    "additionalItems", "contains", "unevaluatedProperties", "unevaluatedItems",
    "if", "then", "else", "not", "dependentRequired", "dependentSchemas",
    "propertyNames", "patternProperties", "contentMediaType", "contentEncoding",
})

_DEFAULT_MAX_ITEMS = 32
_MAX_ITEMS_CAP = 128
_MAX_DEPTH = 12


def compile_schema(schema, *, name="schema"):
    """把 JSON Schema 编译成 :class:`Node`；不支持的结构立刻抛 ValueError。"""
    return _Compiler(schema, name).compile()


def _literal_node(values, label):
    texts = []
    for value in values:
        texts.append(_json_text(value))
    texts = tuple(dict.fromkeys(texts))
    return Node(
        "literal",
        label=label,
        start_bytes={text[0] for text in texts},
        literals=texts,
    )


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _text_form(value):
    """枚举值在"纯文本协议"（XML 参数值）里的写法：字符串不加引号，其余用 JSON 文本。"""
    if isinstance(value, str):
        return value.encode("utf-8")
    return _json_text(value)


class _Compiler:
    def __init__(self, schema, name):
        self.root = schema
        self.name = name
        if not isinstance(schema, dict):
            raise ValueError(f"{name}：schema 必须是 JSON 对象")
        self.defs = schema.get("$defs") or schema.get("definitions") or {}
        if not isinstance(self.defs, dict):
            raise ValueError(f"{name}：$defs / definitions 必须是对象")

    def compile(self):
        return self._node(self.root, path="#", depth=0)

    # -- 内部 -------------------------------------------------------------

    def _node(self, schema, path, depth):
        """编译入口：顺带把原始 schema 挂在节点上（纯文本协议要拿枚举的文本写法）。"""
        node = self._compile_node(schema, path, depth)
        node.raw = schema
        return node

    def _resolve(self, ref, path, depth):
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise ValueError(f"{self.name}：只支持本地 $ref（以 # 开头），收到 {ref!r}")
        target = self.root if ref == "#" else self.defs
        if ref == "#":
            if self.root.get("$ref") == "#":
                raise ValueError(f"{self.name}：不支持递归 schema（$ref 指向自身）")
            return self.root
        prefix = "#/$defs/"
        prefix2 = "#/definitions/"
        if ref.startswith(prefix):
            key = ref[len(prefix):]
        elif ref.startswith(prefix2):
            key = ref[len(prefix2):]
        else:
            raise ValueError(f"{self.name}：不支持的 $ref 形式 {ref!r}")
        if key not in self.defs:
            raise ValueError(f"{self.name}：$ref 指向的 {key!r} 不在 $defs 里")
        return self.defs[key]

    def _compile_node(self, schema, path, depth):
        if depth > _MAX_DEPTH:
            raise ValueError(f"{self.name}：schema 嵌套超过 {_MAX_DEPTH} 层")
        if schema is True or schema == {}:
            raise ValueError(
                f"{self.name}@{path}：不支持「任意 JSON」（空 schema）；"
                "请给出明确 type（对象/数组/字符串/数值/布尔/null）"
            )
        if schema is False:
            raise ValueError(f"{self.name}@{path}：schema 为 false，无法生成")
        if not isinstance(schema, dict):
            raise ValueError(f"{self.name}@{path}：schema 必须是 JSON 对象")
        if "$ref" in schema:
            target = self._resolve(schema["$ref"], path, depth)
            if target is schema:
                raise ValueError(f"{self.name}@{path}：不支持递归 schema")
            return self._node(target, path + "->$ref", depth + 1)
        if "allOf" in schema:
            return self._merge_all_of(schema, path, depth)
        if "anyOf" in schema or "oneOf" in schema:
            return self._union(schema, path, depth)
        if "const" in schema:
            return _literal_node([schema["const"]], label=path)
        if "enum" in schema:
            values = schema["enum"]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{self.name}@{path}：enum 必须是非空数组")
            return _literal_node(values, path)

        type_ = schema.get("type")
        if isinstance(type_, list):
            if len(type_) == 1:
                type_ = type_[0]
            else:
                branches = []
                for item in type_:
                    if not isinstance(item, str):
                        raise ValueError(f"{self.name}@{path}：type 数组里必须是字符串")
                    sub = dict(schema)
                    sub["type"] = item
                    branches.append(self._node(sub, f"{path}[{item}]", depth + 1))
                return self._union_node(branches, path)
        if type_ is None:
            if "properties" in schema or "additionalProperties" in schema:
                type_ = "object"
            elif "items" in schema:
                type_ = "array"
            else:
                raise ValueError(
                    f"{self.name}@{path}：缺少 type，且无法从 properties/items 推断"
                )
        if type_ == "object":
            return self._object(schema, path, depth)
        if type_ == "array":
            return self._array(schema, path, depth)
        if type_ in ("string", "number", "integer", "boolean", "null"):
            return self._scalar(type_, schema, path)
        raise ValueError(f"{self.name}@{path}：不支持的 type {type_!r}")

    def _scalar(self, type_, schema, path):
        if type_ == "string":
            return Node("string", label=path, start_bytes={0x22})
        if type_ in ("number", "integer"):
            start = set(range(0x30, 0x3A)) | {0x2D}
            return Node("number", label=path, start_bytes=start, integer=type_ == "integer")
        if type_ == "boolean":
            return Node("literal", label=path, start_bytes={0x74, 0x66},
                        literals=(b"true", b"false"))
        return Node("literal", label=path, start_bytes={0x6E}, literals=(b"null",))

    def _object(self, schema, path, depth):
        props_schema = schema.get("properties") or {}
        if not isinstance(props_schema, dict):
            raise ValueError(f"{self.name}@{path}：properties 必须是对象")
        extra = schema.get("additionalProperties", False)
        if extra is True:
            raise ValueError(
                f"{self.name}@{path}：additionalProperties=true 暂不支持"
                "（对象需要固定的键集合）；请改成 false 或删掉该字段"
            )
        keys = list(props_schema)
        key_bytes = []
        for key in keys:
            if not isinstance(key, str) or not key:
                raise ValueError(f"{self.name}@{path}：properties 的键必须是非空字符串")
            if any(char in key for char in ('"', "\\")):
                raise ValueError(
                    f"{self.name}@{path}：暂不支持需要转义的属性名 {key!r}"
                )
            key_bytes.append(key.encode("utf-8"))
        required = schema.get("required") or []
        if not isinstance(required, list):
            raise ValueError(f"{self.name}@{path}：required 必须是数组")
        required_set = set(required)
        unknown = required_set - set(keys)
        if unknown:
            raise ValueError(
                f"{self.name}@{path}：required 里的 {sorted(unknown)} 没有出现在 properties"
            )
        props = [
            (key, self._node(props_schema[key], f"{path}.{key}", depth + 1))
            for key in keys
        ]
        cand_end = []
        for index in range(len(keys)):
            end = len(keys) - 1
            for later in range(index, len(keys)):
                if keys[later] in required_set:
                    end = later
                    break
            cand_end.append(end)
        required_tail = tuple(
            any(keys[later] in required_set for later in range(index, len(keys)))
            for index in range(len(keys) + 1)
        )
        return Node(
            "object",
            label=path,
            start_bytes={0x7B},
            props=props,
            key_bytes=key_bytes,
            cand_end=cand_end,
            required_tail=required_tail,
        )

    def _array(self, schema, path, depth):
        items = schema.get("items")
        if items is None:
            raise ValueError(
                f"{self.name}@{path}：数组必须给出 items（任意元素类型暂不支持）"
            )
        if isinstance(items, list):
            raise ValueError(f"{self.name}@{path}：不支持元组式 items 数组")
        item_node = self._node(items, path + "[]", depth + 1)
        min_items = int(schema.get("minItems", 0) or 0)
        max_items = int(schema.get("maxItems", _DEFAULT_MAX_ITEMS) or _DEFAULT_MAX_ITEMS)
        if min_items < 0 or max_items < min_items:
            raise ValueError(
                f"{self.name}@{path}：minItems={min_items} / maxItems={max_items} 不合法"
            )
        if max_items > _MAX_ITEMS_CAP:
            max_items = _MAX_ITEMS_CAP
        return Node(
            "array", label=path, start_bytes={0x5B}, items=item_node,
            min_items=min_items, max_items=max_items,
        )

    def _merge_all_of(self, schema, path, depth):
        """allOf 只支持"多个对象 schema 合并"这一种常见形态（属性名不能重复）。"""
        branches = schema["allOf"]
        if not isinstance(branches, list) or not branches:
            raise ValueError(f"{self.name}@{path}：allOf 必须是非空数组")
        merged = {"type": "object", "properties": {}, "required": []}
        for branch in branches:
            if not isinstance(branch, dict):
                raise ValueError(f"{self.name}@{path}：allOf 的每个分支必须是对象")
            resolved = self._resolve(branch["$ref"], path, depth) if "$ref" in branch else branch
            if not isinstance(resolved, dict):
                raise ValueError(f"{self.name}@{path}：allOf 分支无法解析为对象 schema")
            if resolved.get("type", "object") != "object":
                raise ValueError(f"{self.name}@{path}：allOf 目前只支持对象分支")
            if resolved.get("additionalProperties") is True:
                raise ValueError(f"{self.name}@{path}：allOf 分支不能用 additionalProperties=true")
            for key, child in (resolved.get("properties") or {}).items():
                if key in merged["properties"]:
                    raise ValueError(
                        f"{self.name}@{path}：allOf 合并时属性 {key!r} 重复"
                    )
                merged["properties"][key] = child
            for key in resolved.get("required") or []:
                if key not in merged["required"]:
                    merged["required"].append(key)
        return self._object(merged, path, depth)

    def _union(self, schema, path, depth):
        branches = schema.get("anyOf") or schema.get("oneOf")
        if not isinstance(branches, list) or not branches:
            raise ValueError(f"{self.name}@{path}：anyOf/oneOf 必须是非空数组")
        nodes = [
            self._node(branch, f"{path}|{index}", depth + 1)
            for index, branch in enumerate(branches)
        ]
        return self._union_node(nodes, path)

    def _union_node(self, nodes, path):
        seen = {}
        for index, node in enumerate(nodes):
            for byte in node.start_bytes:
                if byte in seen:
                    raise ValueError(
                        f"{self.name}@{path}：anyOf 分支 {seen[byte]} 与 {index} 的首字节重叠"
                        f"（0x{byte:02x}），当前只支持首字节互不相交的分支"
                    )
                seen[byte] = index
        if not seen:
            raise ValueError(f"{self.name}@{path}：anyOf 分支没有可生成的类型")
        return Node(
            "union", label=path, start_bytes=seen.keys(), branches=tuple(nodes)
        )


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------

_M_START = 0
_M_KEY_START = 1
_M_KEY_NEED = 2
_M_KEY_MATCH = 3
_M_COLON = 4
_M_OBJ_END = 5
_M_ARR_START = 6
_M_ARR_NEED = 7
_M_ARR_END = 8
_M_STR = 9
_M_ESC = 10
_M_U = 11
_M_NUM = 12
_M_LIT = 13


class _Marker:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


JSON_DONE = _Marker("json_done")
JSON_PENDING = _Marker("json_pending")


def initial_state(node):
    return (_M_START, node, (("end",),))


def _resume(stack):
    frame = stack[-1]
    rest = stack[:-1]
    tag = frame[0]
    if tag == "end":
        return JSON_DONE
    if tag == "obj":
        return (_M_OBJ_END, frame[1], frame[2], rest)
    return (_M_ARR_END, frame[1], frame[2], rest)


def _pending(stack, byte):
    """标量（数值/字面量）在"已完整"状态下遇到的下一个字节。

    标量值的结束不消费字节：先回到外层状态，再把同一个字节喂给它。外层只剩根节点时
    返回 ``JSON_PENDING``，交给语法层的后继段处理（例如外层还有 ``}``）。
    """
    resumed = _resume(stack)
    if resumed is JSON_DONE:
        return JSON_PENDING
    return step(resumed, byte)


def _value_done(stack):
    """一个值（字面量）刚消费完最后一个字节后的状态：回到外层，或整个 JSON 结束。"""
    resumed = _resume(stack)
    return JSON_DONE if resumed is JSON_DONE else resumed


_NUM_END_OK = frozenset({"zero", "int", "frac", "exp"})


def accepting(state):
    mode = state[0]
    if mode == _M_NUM:
        return state[1] in _NUM_END_OK
    if mode == _M_LIT:
        return any(len(lit) == matched for lit, matched in state[1])
    return False


def opaque(state):
    if state[0] == _M_STR and state[1] == 0:
        return ("utf8", _STRING_FORBIDDEN)
    return None


def _enter_value(node, stack, byte):
    kind = node.kind
    if kind == "object":
        return (_M_KEY_START, node, 0, stack) if byte == 0x7B else None
    if kind == "array":
        return (_M_ARR_START, node, 0, stack) if byte == 0x5B else None
    if kind == "string":
        return (_M_STR, 0, stack) if byte == 0x22 else None
    if kind == "number":
        if byte == 0x2D:
            return (_M_NUM, "sign", node.integer, stack)
        if byte == 0x30:
            return (_M_NUM, "zero", node.integer, stack)
        if 0x31 <= byte <= 0x39:
            return (_M_NUM, "int", node.integer, stack)
        return None
    if kind == "literal":
        candidates = tuple((lit, 1) for lit in node.literals if lit[0] == byte)
        if not candidates:
            return None
        if len(candidates) == 1 and len(candidates[0][0]) == 1:
            return _value_done(stack)
        return (_M_LIT, candidates, stack)
    if kind == "union":
        chosen = None
        for branch in node.branches:
            if byte in branch.start_bytes:
                if chosen is not None:
                    return None  # 编译期已保证不相交，这里只是兜底
                chosen = branch
        if chosen is None:
            return None
        return _enter_value(chosen, stack, byte)
    raise ValueError(f"未知的节点类型 {kind}")


def step(state, byte):
    mode = state[0]
    if mode == _M_START:
        return _enter_value(state[1], state[2], byte)

    if mode == _M_KEY_START:
        _, node, index, stack = state
        if byte == 0x7D:  # '}'
            if not node.required_tail[index]:
                return _resume(stack)
            return None
        if byte == 0x22 and index < len(node.props):
            return _key_match(node, index, stack)
        return None

    if mode == _M_KEY_NEED:
        _, node, index, stack = state
        if byte == 0x22 and index < len(node.props):
            return _key_match(node, index, stack)
        return None

    if mode == _M_KEY_MATCH:
        _, node, candidates, stack = state
        if byte == 0x22:
            for index, matched in candidates:
                if matched == len(node.key_bytes[index]):
                    return (_M_COLON, node, index + 1, stack)
            return None
        advanced = []
        for index, matched in candidates:
            raw = node.key_bytes[index]
            if matched < len(raw) and raw[matched] == byte:
                advanced.append((index, matched + 1))
        if not advanced:
            return None
        return (_M_KEY_MATCH, node, tuple(advanced), stack)

    if mode == _M_COLON:
        _, node, next_index, stack = state
        if byte != 0x3A:  # ':'
            return None
        child = node.props[next_index - 1][1]
        return (_M_START, child, stack + (("obj", node, next_index),))

    if mode == _M_OBJ_END:
        _, node, index, stack = state
        if byte == 0x7D and not node.required_tail[index]:
            return _resume(stack)
        if byte == 0x2C and index < len(node.props):  # ','
            return (_M_KEY_NEED, node, index, stack)
        return None

    if mode == _M_ARR_START:
        _, node, count, stack = state
        if byte == 0x5D and count >= node.min_items:  # ']'
            return _resume(stack)
        if count >= node.max_items:
            return None
        return _enter_value(node.items, stack + (("arr", node, count + 1),), byte)

    if mode == _M_ARR_NEED:
        _, node, count, stack = state
        return _enter_value(node.items, stack + (("arr", node, count + 1),), byte)

    if mode == _M_ARR_END:
        _, node, count, stack = state
        if byte == 0x5D and count >= node.min_items:
            return _resume(stack)
        if byte == 0x2C and count < node.max_items:
            return (_M_ARR_NEED, node, count, stack)
        return None

    if mode == _M_STR:
        _, utf8, stack = state
        if byte == 0x22:
            return _resume(stack)
        if byte == 0x5C:
            return (_M_ESC, stack)
        if byte < 0x20:
            return None
        nxt = Utf8Tracker.step(utf8, byte)
        if nxt is None:
            return None
        return (_M_STR, nxt, stack)

    if mode == _M_ESC:
        stack = state[1]
        if byte in b'"\\/bfnrt':
            return (_M_STR, 0, stack)
        if byte == 0x75:  # 'u'
            return (_M_U, 0, stack)
        return None

    if mode == _M_U:
        _, position, stack = state
        if byte not in _HEX:
            return None
        position += 1
        if position == 4:
            return (_M_STR, 0, stack)
        return (_M_U, position, stack)

    if mode == _M_NUM:
        _, sub, integer, stack = state
        if sub == "sign":
            if byte == 0x30:
                return (_M_NUM, "zero", integer, stack)
            if 0x31 <= byte <= 0x39:
                return (_M_NUM, "int", integer, stack)
            return None
        if sub == "int":
            if 0x30 <= byte <= 0x39:
                return (_M_NUM, "int", integer, stack)
            if not integer and byte == 0x2E:
                return (_M_NUM, "dot", integer, stack)
            if not integer and byte in (0x65, 0x45):
                return (_M_NUM, "expstart", integer, stack)
            return _pending(stack, byte)
        if sub == "zero":
            if not integer and byte == 0x2E:
                return (_M_NUM, "dot", integer, stack)
            if not integer and byte in (0x65, 0x45):
                return (_M_NUM, "expstart", integer, stack)
            return _pending(stack, byte)
        if sub == "dot":
            if 0x30 <= byte <= 0x39:
                return (_M_NUM, "frac", integer, stack)
            return None
        if sub == "frac":
            if 0x30 <= byte <= 0x39:
                return (_M_NUM, "frac", integer, stack)
            if byte in (0x65, 0x45):
                return (_M_NUM, "expstart", integer, stack)
            return _pending(stack, byte)
        if sub == "expstart":
            if 0x30 <= byte <= 0x39:
                return (_M_NUM, "exp", integer, stack)
            if byte in (0x2B, 0x2D):
                return (_M_NUM, "expsign", integer, stack)
            return None
        if sub == "expsign":
            if 0x30 <= byte <= 0x39:
                return (_M_NUM, "exp", integer, stack)
            return None
        if sub == "exp":
            if 0x30 <= byte <= 0x39:
                return (_M_NUM, "exp", integer, stack)
            return _pending(stack, byte)
        return None

    if mode == _M_LIT:
        _, candidates, stack = state
        advanced = [
            (lit, matched + 1)
            for lit, matched in candidates
            if matched < len(lit) and lit[matched] == byte
        ]
        if advanced:
            if len(advanced) == 1 and advanced[0][1] == len(advanced[0][0]):
                return _value_done(stack)
            return (_M_LIT, tuple(advanced), stack)
        if any(matched == len(lit) for lit, matched in candidates):
            return _pending(stack, byte)
        return None

    raise ValueError(f"未知的 JSON 状态 {state!r}")


def _key_match(node, index, stack):
    end = node.cand_end[index]
    candidates = tuple((item, 0) for item in range(index, end + 1))
    return (_M_KEY_MATCH, node, candidates, stack)


class JsonValue(Segment):
    """把 JSON 值包装成语法段，供工具调用协议的外层序列复用。"""

    __slots__ = ("node",)

    def __init__(self, node):
        self.node = node

    def start(self):
        return initial_state(self.node)

    def step(self, state, byte):
        result = step(state, byte)
        if result is JSON_DONE:
            return DONE
        if result is JSON_PENDING:
            return PENDING
        return result

    def accepting(self, state):
        return accepting(state)

    def opaque(self, state):
        return opaque(state)

    def __repr__(self):
        return f"JsonValue({self.node!r})"


def text_literals(schema):
    """给纯文本协议（XML 参数值）取枚举字面量；非枚举返回 None。"""
    if not isinstance(schema, dict):
        return None
    if "const" in schema:
        return (_text_form(schema["const"]),)
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return tuple(_text_form(value) for value in schema["enum"])
    return None
