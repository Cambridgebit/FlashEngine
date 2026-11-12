"""OpenAI 工具调用/结构化输出的端到端校验（纯 stdlib，任何机器都能对着 server 跑）。

它模拟 agent 客户端（DSH 等）的两种典型请求：

- ``tools`` + ``tool_choice=required``：要求服务返回 ``finish_reason="tool_calls"``，
  ``arguments`` 必须是可 ``json.loads`` 的对象，``function.name`` 必须是给定工具之一；
- ``response_format={"type":"json_schema",...}``：要求 ``content`` 本身是符合 schema 的 JSON。

另外校验流式（SSE）把 tool_calls 拆成 delta 时，分片拼接后仍是合法 JSON。

用法（先起 server）：:

    ~/vllm-env/bin/python3 -m scripts.toolcall_bench http://127.0.0.1:8000 --model MiniCPM5-2B
    ~/vllm-env/bin/python3 -m scripts.toolcall_bench http://127.0.0.1:8000 --mode tool --stream
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file from disk",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_lines": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
}


def _post(base_url, payload, api_key=None, timeout=600):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return body, time.perf_counter() - started


def _read_sse(response):
    """把 SSE 流读成 (chunks, full_text, first_delta_s)。"""
    chunks = []
    first = None
    started = time.perf_counter()
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        chunks.append(chunk)
        if first is None:
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content:
                first = time.perf_counter() - started
    return chunks, first


def _stream_post(base_url, payload, api_key=None, timeout=600):
    payload = dict(payload)
    payload["stream"] = True
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        chunks, first = _read_sse(response)
    return chunks, first, time.perf_counter() - started


def _merge_stream(chunks):
    """按 OpenAI 流式语义把 deltas 拼成一条完整 message。"""
    content = []
    reasoning = []
    calls = {}
    finish = None
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning.append(delta["reasoning_content"])
            for item in delta.get("tool_calls") or []:
                index = item.get("index", 0)
                slot = calls.setdefault(index, {
                    "id": None, "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if item.get("id"):
                    slot["id"] = item["id"]
                function = item.get("function") or {}
                if function.get("name"):
                    slot["function"]["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    slot["function"]["arguments"] += function["arguments"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {
        "content": "".join(content) or None,
        "reasoning_content": "".join(reasoning) or None,
        "tool_calls": [calls[i] for i in sorted(calls)],
        "finish_reason": finish,
    }


def _validate_calls(case, calls, tool_names):
    problems = []
    if not calls:
        problems.append("没有返回 tool_calls")
        return problems
    for call in calls:
        function = call.get("function") or {}
        name = function.get("name")
        if name not in tool_names:
            problems.append(f"函数名 {name!r} 不在 {tool_names} 里")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                problems.append(f"arguments 不是合法 JSON：{exc}：{arguments[:80]!r}")
                continue
        if not isinstance(arguments, dict):
            problems.append(f"arguments 不是对象：{arguments!r}")
            continue
        if not arguments:
            problems.append("arguments 为空对象")
    return problems


def _validate_schema(value, schema, path="$"):
    problems = []
    type_ = schema.get("type")
    if type_ == "object":
        if not isinstance(value, dict):
            return [f"{path} 不是对象"]
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                problems.append(f"{path} 缺必填键 {key}")
        for key, child in properties.items():
            if key in value:
                problems.extend(_validate_schema(value[key], child, f"{path}.{key}"))
    elif type_ == "array":
        if not isinstance(value, list):
            return [f"{path} 不是数组"]
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                problems.extend(_validate_schema(item, item_schema, f"{path}[{index}]"))
    elif type_ == "string":
        if not isinstance(value, str):
            problems.append(f"{path} 不是字符串")
    elif type_ == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append(f"{path} 不是整数")
    elif type_ == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append(f"{path} 不是数值")
    elif type_ == "boolean":
        if not isinstance(value, bool):
            problems.append(f"{path} 不是布尔")
    return problems


def _tool_case(base_url, args, label, tools, messages, tool_choice):
    payload = {
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
    }
    if args.stream:
        chunks, first, elapsed = _stream_post(base_url, payload, args.api_key)
        message = _merge_stream(chunks)
        ttft = first
    else:
        body, elapsed = _post(base_url, payload, args.api_key)
        data = json.loads(body)
        choice = data["choices"][0]
        message = choice["message"]
        message["finish_reason"] = choice.get("finish_reason")
        ttft = None
    names = [tool["function"]["name"] for tool in tools]
    problems = _validate_calls(label, message.get("tool_calls"), names)
    if message.get("finish_reason") != "tool_calls":
        problems.append(f"finish_reason={message.get('finish_reason')!r}，期望 tool_calls")
    if message.get("content"):
        problems.append(f"tool_calls 响应里还带正文：{message['content'][:60]!r}")
    rendered = [
        f"{call['function']['name']}({call['function']['arguments']})"
        for call in (message.get("tool_calls") or [])
    ]
    return problems, elapsed, ttft, rendered


def _json_case(base_url, args):
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "temperature_c": {"type": "number"},
            "condition": {"type": "string", "enum": ["sunny", "rainy", "cloudy"]},
        },
        "required": ["city", "temperature_c"],
    }
    payload = {
        "model": args.model,
        "messages": [{"role": "user",
                      "content": "Return the weather in Paris as JSON."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "weather", "schema": schema, "strict": True},
        },
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
    }
    if args.stream:
        chunks, first, elapsed = _stream_post(base_url, payload, args.api_key)
        message = _merge_stream(chunks)
        ttft = first
    else:
        body, elapsed = _post(base_url, payload, args.api_key)
        data = json.loads(body)
        choice = data["choices"][0]
        message = choice["message"]
        message["finish_reason"] = choice.get("finish_reason")
        ttft = None
    problems = []
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        problems.append("没有返回 JSON 正文")
        return problems, elapsed, ttft, []
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        problems.append(f"正文不是合法 JSON：{exc}：{content[:120]!r}")
        return problems, elapsed, ttft, [content[:120]]
    problems.extend(_validate_schema(value, schema))
    return problems, elapsed, ttft, [json.dumps(value, ensure_ascii=False)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="OpenAI 工具调用端到端校验")
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="flashengine")
    parser.add_argument("--mode", choices=["tool", "json", "all"], default="all")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--tool-choice", default="required",
                        choices=["required", "auto"])
    args = parser.parse_args(argv)

    failures = 0
    for repeat in range(args.repeat):
        if args.mode in ("tool", "all"):
            cases = [
                ("weather", [WEATHER_TOOL],
                 [{"role": "user",
                   "content": "What is the weather in Paris? Call the tool."}]),
                ("read_file", [READ_TOOL],
                 [{"role": "user",
                   "content": "Read the file src/main.py and show me the code."}]),
            ]
            for label, tools, messages in cases:
                try:
                    problems, elapsed, ttft, rendered = _tool_case(
                        args.base_url, args, label, tools, messages, args.tool_choice)
                except urllib.error.HTTPError as exc:
                    print("[%s] HTTP %d: %s" % (label, exc.code, exc.read()[:400]))
                    failures += 1
                    continue
                except Exception as exc:  # noqa: BLE001
                    print("[%s] 请求失败：%r" % (label, exc))
                    failures += 1
                    continue
                status = "ok  " if not problems else "FAIL"
                extra = "" if ttft is None else " ttft=%.0fms" % (ttft * 1000)
                print("[%s] %s %.0fms%s calls=%s%s" % (
                    label, status, elapsed * 1000, extra, rendered,
                    "" if not problems else " 问题=" + "; ".join(problems)))
                failures += bool(problems)
        if args.mode in ("json", "all"):
            try:
                problems, elapsed, ttft, rendered = _json_case(args.base_url, args)
            except urllib.error.HTTPError as exc:
                print("[json] HTTP %d: %s" % (exc.code, exc.read()[:400]))
                failures += 1
            except Exception as exc:  # noqa: BLE001
                print("[json] 请求失败：%r" % exc)
                failures += 1
            else:
                status = "ok  " if not problems else "FAIL"
                extra = "" if ttft is None else " ttft=%.0fms" % (ttft * 1000)
                print("[json] %s %.0fms%s value=%s%s" % (
                    status, elapsed * 1000, extra, rendered,
                    "" if not problems else " 问题=" + "; ".join(problems)))
                failures += bool(problems)
        if args.repeat > 1:
            print("-- repeat %d/%d done --" % (repeat + 1, args.repeat))
    print("\n%s" % ("全部通过" if not failures else "%d 项失败" % failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
