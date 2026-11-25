"""OpenAI 工具调用 ↔ 模型原生文本协议的中间层。

这一层是 server 与"模型家族自己的格式"之间的翻译：

- **请求方向**：把 OpenAI 的 ``messages`` + ``tools`` 规整成 chat template 认的形状
  （保留 ``tool_calls`` / ``tool_call_id`` / ``reasoning_content``，``arguments`` 从
  JSON 字符串还原成 dict），再交给 ``apply_chat_template`` 渲染成 token；
- **响应方向**：把模型原生生成的文本按协议解析回 ``content`` / ``reasoning_content`` /
  ``tool_calls``（``arguments`` 按工具 schema 从纯文本转回 JSON 值）；
- **协议探测**：按 chat template 的特征串判断是 XML（MiniCPM5）、Hermes（Qwen3）还是
  DeepSeek 专用 token 形态。

只做文本/协议层，不碰引擎；约束解码语法在 :mod:`flashengine.structured` 里。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

PROTOCOLS = ("xml_function", "hermes", "deepseek_v3")
_AUTO = "auto"

_XML_CALL = re.compile(r"<function\s+name=\"([^\"]+)\"\s*>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<param\s+name=\"([^\"]+)\"\s*>(.*?)</param>", re.S)
_CDATA = re.compile(r"<\!\[CDATA\[(.*?)\]\]>", re.S)
_HERMES_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
# Qwen3 / Hermes 的 XML 形态：<tool_call><function=NAME><parameter=KEY>值</parameter></function></tool_call>
# 与上面的 JSON 形态（hermes）并存。放开 tool_choice=auto（不挂语法约束）时，模型几乎
# 只用这一种形态；漏解析会让客户端看到原始 XML、以为模型没调工具。
_HX_CALL = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)\s*>(.*?)</function>\s*</tool_call>", re.S
)
_HX_PARAM = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.S)
_DS_CALL_BEGIN = "<｜｜tool▁call▁begin｜｜>"
_DS_CALL_END = "<｜｜tool▁call▁end｜｜>"
_DS_SEP = "<｜｜tool▁sep｜｜>"

_ENTITY = {
    "&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&apos;": "'",
}

_THINK_OPEN = ("<think>", "<think\n", "<think>")
_THINK_CLOSE = ("</think>", "</think\n", "</think>")


# ---------------------------------------------------------------------------
# 协议探测
# ---------------------------------------------------------------------------

def detect_tool_protocol(tokenizer) -> str:
    """按 chat template 的写法判断模型家族用的工具调用协议。

    template 里出现 ``<function name="…">`` 的 XML 规范说明是 MiniCPM5 系；
    出现 ``<tool_call>`` 的是 Qwen3/Hermes 系；词表里有 DeepSeek 专用分隔 token 的
    才可能是 ``deepseek_v3``。
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = {}
    if "<｜｜tool▁call▁begin｜｜>" in vocab or _DS_CALL_BEGIN in template:
        return "deepseek_v3"
    if "<function name=" in template:
        return "xml_function"
    if "<tool_call>" in template:
        return "hermes"
    if "<function" in template:
        return "xml_function"
    return "hermes"


def resolve_protocol(protocol: str, tokenizer) -> str:
    if protocol in (None, "", _AUTO):
        return detect_tool_protocol(tokenizer)
    if protocol not in PROTOCOLS:
        raise ValueError(
            f"未知的工具协议 {protocol!r}，可选：auto/{'/'.join(PROTOCOLS)}"
        )
    return protocol


# ---------------------------------------------------------------------------
# 请求方向
# ---------------------------------------------------------------------------

def _flatten_content(content):
    """OpenAI content 可以是字符串或分段数组；模板只吃字符串。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "image_url":
                    raise ValueError("当前服务不支持图片输入")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _json_arguments(arguments):
    """模板层要 dict（MiniCPM5 会对 arguments.items() 迭代），从字符串还原。"""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"_raw": arguments}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {}


def normalize_messages(messages) -> list[dict]:
    """OpenAI messages → chat template 认的 messages（保留工具调用字段）。"""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须是非空数组")
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] 必须是对象")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"messages[{index}].role 不支持：{role!r}")
        item = {"role": role, "content": _flatten_content(message.get("content"))}
        if role == "assistant":
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                item["reasoning_content"] = reasoning
            calls = message.get("tool_calls")
            if calls:
                if not isinstance(calls, list):
                    raise ValueError(f"messages[{index}].tool_calls 必须是数组")
                rendered = []
                for call in calls:
                    if not isinstance(call, dict):
                        raise ValueError(f"messages[{index}].tool_calls 里必须是对象")
                    function = call.get("function") or {}
                    if not isinstance(function, dict) or not function.get("name"):
                        raise ValueError(
                            f"messages[{index}].tool_calls[].function.name 缺失"
                        )
                    rendered.append({
                        "id": call.get("id") or f"call_{uuid.uuid4().hex[:16]}",
                        "type": call.get("type") or "function",
                        "function": {
                            "name": function["name"],
                            "arguments": _json_arguments(function.get("arguments")),
                        },
                    })
                item["tool_calls"] = rendered
        if role == "tool":
            if message.get("tool_call_id") is not None:
                item["tool_call_id"] = message["tool_call_id"]
            if message.get("name") is not None:
                item["name"] = message["name"]
        normalized.append(item)
    return normalized


def render_chat(tokenizer, messages, *, tools=None, template_kwargs=None,
                enable_thinking=None) -> list[int]:
    """渲染成 token id（模板字符串再 encode，与仓库既有口径一致）。"""
    kwargs = dict(template_kwargs or {})
    if enable_thinking is not None and "enable_thinking" not in kwargs:
        kwargs["enable_thinking"] = bool(enable_thinking)
    extra = {}
    if tools:
        extra["tools"] = tools
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra, **kwargs
        )
    except Exception as exc:  # 模板不认的 kwargs / 消息形状错误统一转成 400 口径
        raise ValueError(f"chat template 渲染失败：{exc}") from exc
    if isinstance(text, (list, tuple)):
        return list(text)
    return list(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# 响应方向
# ---------------------------------------------------------------------------

@dataclass
class ParsedOutput:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def split_thinking(text: str) -> tuple[str | None, str]:
    """把思考段与正文分开；两种标记族的写法不统一，这里只按出现顺序切。"""
    if not text:
        return None, ""
    for marker in _THINK_OPEN:
        start = text.find(marker)
        if start < 0:
            continue
        rest = text[start + len(marker):]
        for closer in _THINK_CLOSE:
            end = rest.find(closer)
            if end >= 0:
                reasoning = rest[:end].strip("\n").strip()
                return (reasoning or None), rest[end + len(closer):]
        return rest.strip("\n").strip() or None, ""
    return None, text


def _xml_unescape(text: str) -> str:
    def replace(match):
        token = match.group(0)
        if token in _ENTITY:
            return _ENTITY[token]
        if token.startswith("&#x") or token.startswith("&#X"):
            try:
                return chr(int(token[3:-1], 16))
            except ValueError:
                return token
        if token.startswith("&#"):
            try:
                return chr(int(token[2:-1]))
            except ValueError:
                return token
        return token

    return re.sub(r"&(?:lt|gt|amp|quot|apos|#\d+|#[xX][0-9a-fA-F]+);", replace, text)


def _xml_param_value(raw: str) -> str:
    cdata = _CDATA.search(raw)
    if cdata is not None:
        return cdata.group(1)
    return _xml_unescape(raw)


def _coerce(value: str, schema: dict | None):
    if not isinstance(schema, dict):
        return value
    type_ = schema.get("type")
    if isinstance(type_, list):
        type_ = next((item for item in type_ if item != "null"), None)
    text = value.strip()
    try:
        if type_ == "integer":
            return int(text)
        if type_ == "number":
            return float(text)
        if type_ == "boolean":
            lowered = text.lower()
            if lowered in ("true", "1", "yes"):
                return True
            if lowered in ("false", "0", "no"):
                return False
            return value
        if type_ in ("object", "array"):
            return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return value
    return value


def _tool_parameters(tools) -> dict[str, dict]:
    parameters = {}
    for tool in tools or []:
        function = (tool or {}).get("function") or {}
        name = function.get("name")
        if name:
            parameters[name] = function.get("parameters") or {}
    return parameters


def _property_schema(parameters: dict, key: str):
    properties = parameters.get("properties")
    if isinstance(properties, dict):
        child = properties.get(key)
        if isinstance(child, dict):
            return child
    return None


def _new_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _parse_xml(body: str, tools) -> tuple[str, list[dict]]:
    parameters = _tool_parameters(tools)
    calls = []
    spans = []
    for match in _XML_CALL.finditer(body):
        name = match.group(1)
        inner = match.group(2)
        schema = parameters.get(name) or {}
        arguments = {}
        for param in _XML_PARAM.finditer(inner):
            key = param.group(1)
            arguments[key] = _coerce(
                _xml_param_value(param.group(2)), _property_schema(schema, key)
            )
        calls.append(_new_call(name, arguments))
        spans.append(match.span())
    content = []
    cursor = 0
    for start, end in spans:
        content.append(body[cursor:start])
        cursor = end
    content.append(body[cursor:])
    return "".join(content).strip(), calls


def _parse_hermes(body: str, tools) -> tuple[str, list[dict]]:
    calls = []
    spans = []
    for match in _HERMES_CALL.finditer(body):
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "name" in payload:
            name = payload.get("name")
            arguments = payload.get("arguments", {})
        elif isinstance(payload, dict) and "function" in payload:
            function = payload.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            continue
        if not isinstance(name, str) or not name:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"_value": arguments}
        calls.append(_new_call(name, arguments))
        spans.append(match.span())
    content = []
    cursor = 0
    for start, end in spans:
        content.append(body[cursor:start])
        cursor = end
    content.append(body[cursor:])
    return "".join(content).strip(), calls


def _parse_hermes_xml(body: str, tools) -> tuple[str, list[dict]]:
    """解析 ``<function=NAME><parameter=KEY>值</parameter>`` 形态的工具调用。"""
    parameters = _tool_parameters(tools)
    calls = []
    spans = []
    for match in _HX_CALL.finditer(body):
        name = match.group(1).strip()
        inner = match.group(2)
        schema = parameters.get(name) or {}
        arguments = {}
        for param in _HX_PARAM.finditer(inner):
            key = param.group(1).strip()
            raw = param.group(2)
            value = _xml_param_value(raw)
            if not _CDATA.search(raw):
                value = value.strip()
            arguments[key] = _coerce(value, _property_schema(schema, key))
        calls.append(_new_call(name, arguments))
        spans.append(match.span())
    content = []
    cursor = 0
    for start, end in spans:
        content.append(body[cursor:start])
        cursor = end
    content.append(body[cursor:])
    text = "".join(content).strip()
    return text, calls


def _parse_deepseek(body: str, tools) -> tuple[str, list[dict]]:
    calls = []
    spans = []
    for match in re.finditer(
        re.escape(_DS_CALL_BEGIN) + r"(.*?)" + re.escape(_DS_CALL_END), body, re.S
    ):
        chunk = match.group(1)
        head, _, rest = chunk.partition(_DS_SEP)
        if head.strip() != "function":
            continue
        name, _, payload = rest.partition("\n")
        name = name.strip()
        if not name:
            continue
        fence = re.search(r"```(?:json)?\s*(.*?)```", payload, re.S)
        raw = fence.group(1).strip() if fence else payload.strip()
        try:
            arguments = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            arguments = {"_raw": raw}
        if not isinstance(arguments, dict):
            arguments = {"_value": arguments}
        calls.append(_new_call(name, arguments))
        spans.append(match.span())
    content = []
    cursor = 0
    for start, end in spans:
        content.append(body[cursor:start])
        cursor = end
    content.append(body[cursor:])
    text = "".join(content)
    text = text.replace("<｜｜tool▁calls▁begin｜｜>", "")
    text = text.replace("<｜｜tool▁calls▁end｜｜>", "")
    return text.strip(), calls


def parse_output(text: str, protocol: str, *, tools=None) -> ParsedOutput:
    """把模型原始输出解析成 content / reasoning_content / tool_calls。"""
    reasoning, body = split_thinking(text or "")
    if protocol == "xml_function":
        content, calls = _parse_xml(body, tools)
    elif protocol == "deepseek_v3":
        content, calls = _parse_deepseek(body, tools)
    else:
        content, calls = _parse_hermes(body, tools)
    if not calls and "<function=" in (content or ""):
        # 兜底：任何协议下漏掉的 XML 形态调用都解析出来，避免整段 XML 当成正文回给客户端。
        content, calls = _parse_hermes_xml(content, tools)
    return ParsedOutput(
        content=content or None,
        reasoning_content=reasoning,
        tool_calls=calls,
    )
