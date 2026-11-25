"""结构化约束解码自检（纯 CPU，不需要 GPU/模型权重，只要一个本地 tokenizer 目录）。

覆盖三件事：

1. 语法构建：JSON Schema 编译、三种工具协议的语法图能建出来；
2. 正向：把若干"本该合法"的输出逐 token 喂进语法机，合法 token 必须始终在掩码里，
   结束时语法必须可接受（EOS 可用）；
3. 反向：明显非法的续写必须被掩码挡住；不支持的 schema 必须在编译期报错。

用法（仓库根目录）：:

    ~/vllm-env/bin/python3 -m scripts.check_structured --model /path/to/Qwen3-1.7B

默认用 MiniCPM5-2B 的 tokenizer（本机路径，可用 --model 覆盖）。
"""

import argparse
import json
import sys
import time

from transformers import AutoTokenizer

import numpy as np
import torch

from flashengine.protocol import parse_output

from flashengine.structured import (
    PROTOCOLS,
    BitmaskApplier,
    json_constraint,
    pack_mask,
    tool_constraint,
    unpack_mask,
    vocab_table,
)

DEFAULT_MODEL = "/mnt/d/Materials/AVIC/models/OpenBMB--MiniCPM5-2B"

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
        }, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "encoding": {"type": "string", "enum": ["utf-8", "gbk"]},
            "lines": {"type": "integer"},
        }, "required": ["path"]}}},
]

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "score": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "mode": {"type": "string", "enum": ["fast", "safe"]},
        "ok": {"type": "boolean"},
        "extra": {"type": "null"},
    },
    "required": ["name", "score"],
}


class Checker:
    def __init__(self, tokenizer, vocab_size, eos_ids):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.eos_ids = tuple(eos_ids)
        self.vocab = vocab_table(tokenizer, vocab_size)
        self.failures = []

    def fail(self, message):
        self.failures.append(message)
        print("  FAIL %s" % message)

    # -- 公共检查 ---------------------------------------------------------

    def walk(self, name, constraint, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        state = constraint.initial_state()
        for token in tokens:
            mask = constraint.mask(state)
            if not bool(mask[token]):
                self.fail("%s：合法 token %d（%r）被掩码挡住"
                          % (name, token, self.vocab.bytes[token][:16]))
                return
            state = constraint.advance(state, token)
        if not constraint.is_accepting(state):
            self.fail("%s：结束时语法不可接受：%r" % (name, text[:80]))
            return
        final = constraint.mask(state)
        if not all(bool(final[eos]) for eos in self.eos_ids):
            self.fail("%s：结束时 EOS 不在掩码里" % name)
            return
        print("  ok   %s（%d tokens）" % (name, len(tokens)))

    def expect_blocked(self, name, constraint, samples):
        state = constraint.initial_state()
        mask = constraint.mask(state)
        blocked = 0
        for text in samples:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if tokens and not bool(mask[tokens[0]]):
                blocked += 1
        if blocked != len(samples):
            self.fail("%s：非法首 token 只挡住 %d/%d" % (name, blocked, len(samples)))
        else:
            print("  ok   %s 挡住 %d 个非法开头" % (name, blocked))

    def expect_error(self, name, factory, needle):
        try:
            factory()
        except ValueError as exc:
            if needle and needle not in str(exc):
                self.fail("%s：报错信息里没有 %r（实际：%s）"
                          % (name, needle, str(exc)[:120]))
            else:
                print("  ok   %s 如期报错" % name)
        except Exception as exc:  # noqa: BLE001
            self.fail("%s：抛出的不是 ValueError：%r" % (name, exc))
        else:
            self.fail("%s：不支持的 schema 没有报错" % name)


def main(argv=None):
    parser = argparse.ArgumentParser(description="结构化约束解码 CPU 自检")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="本地 tokenizer 目录（默认本机 MiniCPM5-2B）")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="词表宽度，默认取 tokenizer 长度")
    parser.add_argument("--eos", type=int, nargs="+", default=None,
                        help="结束 token id，默认取 tokenizer 的 eos/pad")
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    vocab_size = args.vocab_size or len(tokenizer)
    eos_ids = args.eos
    if not eos_ids:
        eos_ids = set()
        for attribute in ("eos_token_id", "pad_token_id"):
            value = getattr(tokenizer, attribute, None)
            for item in (value if isinstance(value, (list, tuple)) else (value,)):
                if isinstance(item, int) and 0 <= item < vocab_size:
                    eos_ids.add(item)
        eos_ids = tuple(sorted(eos_ids)) or (0,)

    print("tokenizer=%s vocab=%d eos=%r" % (args.model, vocab_size, eos_ids))
    t0 = time.perf_counter()
    checker = Checker(tokenizer, vocab_size, eos_ids)
    print("词表字节表构建 %.3fs（含首字节分桶）" % (time.perf_counter() - t0))

    # ---- XML（MiniCPM5）--------------------------------------------------
    print("\n[xml_function]")
    xml = tool_constraint(TOOLS, "xml_function", tokenizer, eos_ids, vocab_size=vocab_size)
    checker.walk("单调用", xml,
                 '<function name="get_weather"><param name="city">Paris</param></function>')
    checker.walk("全参数", xml,
                 '<function name="read_file"><param name="path">/tmp/a.txt</param>'
                 '<param name="encoding">utf-8</param>'
                 '<param name="lines">12</param></function>')
    checker.walk("空字符串参数", xml,
                 '<function name="get_weather"><param name="city"></param></function>')
    checker.walk("多调用", xml,
                 '<function name="get_weather"><param name="city">Paris</param></function>\n'
                 '<function name="get_weather"><param name="city">Berlin</param></function>')
    checker.expect_blocked("xml_function", xml,
                           ["The capital of France is Paris.", "hello", "<tool_call>"])

    # ---- Hermes（Qwen3）--------------------------------------------------
    print("\n[hermes]")
    hermes = tool_constraint(TOOLS, "hermes", tokenizer, eos_ids, vocab_size=vocab_size)
    checker.walk("单调用", hermes,
                 '<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>')
    checker.walk("带可选参数", hermes,
                 '<tool_call>\n{"name":"read_file","arguments":'
                 '{"path":"a.txt","encoding":"gbk","lines":3}}\n</tool_call>')
    checker.expect_blocked("hermes", hermes, ["The capital", "hello"])

    # ---- DeepSeek V3.1+ --------------------------------------------------
    print("\n[deepseek_v3]")
    deepseek = tool_constraint(TOOLS, "deepseek_v3", tokenizer, eos_ids,
                               vocab_size=vocab_size)
    checker.walk("单调用", deepseek,
                 '<｜｜tool▁calls▁begin｜｜><｜｜tool▁call▁begin｜｜>function'
                 '<｜｜tool▁sep｜｜>get_weather\n```json\n{"city":"Paris"}\n```'
                 '<｜｜tool▁call▁end｜｜><｜｜tool▁calls▁end｜｜>')

    # ---- 工具调用解析（Qwen3/Hermes 的 XML 形态）--------------------------
    print("\n[tool_parse]")
    qwen_call = ('<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/a.py\n'
                 '</parameter>\n<parameter=lines>\n7\n</parameter>\n</function>\n'
                 '</tool_call>')
    for protocol in PROTOCOLS:
        parsed = parse_output(qwen_call, protocol, tools=TOOLS)
        calls = parsed.tool_calls or []
        if len(calls) != 1:
            checker.fail("%s 未解析出 XML 形态工具调用（%d 个）" % (protocol, len(calls)))
            continue
        arguments = json.loads(calls[0]["function"]["arguments"])
        if calls[0]["function"]["name"] != "read_file":
            checker.fail("%s 解析出的函数名错误" % protocol)
        elif arguments.get("lines") != 7:
            checker.fail("%s 参数没有按 schema 转成整数：%r" % (protocol, arguments))
        elif parsed.content:
            checker.fail("%s 正文里残留了 XML：%r" % (protocol, parsed.content))
        else:
            print("  ok   %s：XML 形态调用可解析，参数按 schema 转型，正文不残留" % protocol)
    mixed = '先看一下文件' + qwen_call + '，然后总结'
    parsed = parse_output(mixed, "hermes", tools=TOOLS)
    if len(parsed.tool_calls or []) != 1 or "先看一下文件" not in (parsed.content or ""):
        checker.fail("混合正文与 XML 调用时切分错误：%r" % (parsed.content,))
    else:
        print("  ok   正文与 XML 调用混排时正确切分")

    # ---- response_format JSON Schema -------------------------------------
    print("\n[json_schema]")
    js = json_constraint(JSON_SCHEMA, tokenizer, eos_ids, vocab_size=vocab_size)
    checker.walk("全字段", js,
                 '{"name":"job","score":0.75,"tags":["a","b"],"mode":"fast",'
                 '"ok":true,"extra":null}')
    checker.walk("仅必填", js, '{"name":"","score":-1.5e3}')
    checker.walk("整数当 number", js, '{"name":"x","score":3}')
    checker.expect_blocked("json_schema", js, ["hello", "The answer is", "```json"])

    print("\n[不支持的 schema]")
    checker.expect_error(
        "additionalProperties=true",
        lambda: json_constraint({"type": "object", "additionalProperties": True},
                                tokenizer, eos_ids, vocab_size=vocab_size),
        "additionalProperties")
    checker.expect_error(
        "空 schema",
        lambda: json_constraint({}, tokenizer, eos_ids, vocab_size=vocab_size),
        "任意 JSON")
    checker.expect_error(
        "anyOf 首字节重叠",
        lambda: json_constraint(
            {"anyOf": [{"type": "string"}, {"type": "string", "enum": ["a"]}]},
            tokenizer, eos_ids, vocab_size=vocab_size),
        "首字节重叠")
    checker.expect_error(
        "required 不在 properties",
        lambda: json_constraint(
            {"type": "object", "properties": {"a": {"type": "string"}},
             "required": ["b"]},
            tokenizer, eos_ids, vocab_size=vocab_size),
        "required")

    # ---- 打包位图与投机多行状态（vLLM 口径）-------------------------------
    print("\n[bitmask]")
    rng = np.random.default_rng(0)
    synthetic = rng.random(vocab_size) > 0.5
    if not np.array_equal(unpack_mask(pack_mask(synthetic), vocab_size), synthetic):
        checker.fail("pack_mask / unpack_mask 往返不一致")
    else:
        print("  ok   pack/unpack 往返一致（%d token）" % vocab_size)

    state = xml.initial_state()
    allowed = unpack_mask(xml.bitmask(state), vocab_size)
    applier = BitmaskApplier(2, vocab_size, device="cpu")
    applier.fill([(xml, state), None], device="cpu")
    logits = torch.zeros(2, vocab_size)
    out = applier.apply(logits, 2)
    allowed_t = torch.from_numpy(allowed)
    if not bool(torch.isinf(out[0][~allowed_t]).all()):
        checker.fail("apply 没有把不允许的 token 写成 -inf")
    elif not bool(out[0][allowed_t].eq(0).all()):
        checker.fail("apply 改动了允许的 token")
    elif not bool(out[1].eq(0).all()):
        checker.fail("全允许行（None）被错误掩码")
    else:
        print("  ok   位图 apply：不允许位置 -inf、允许位不动、None 行全允许")

    valid = tokenizer.encode(
        '<function name="get_weather"><param name="city">Paris</param></function>',
        add_special_tokens=False)
    states = xml.spec_states(state, valid[:3])
    expected = [state]
    current = state
    for token in valid[:3]:
        current = xml.advance(current, token)
        expected.append(current)
    if list(states) != expected:
        checker.fail("spec_states 的行状态与逐 token 推进不一致")
    elif len(states) != 4:
        checker.fail("spec_states 行数应为 k+1")
    else:
        print("  ok   spec_states：k+1 行按乐观推进排列")

    invalid_token = tokenizer.encode("hello", add_special_tokens=False)[0]
    bad_states = xml.spec_states(state, [invalid_token, invalid_token])
    if bad_states[1] != state or bad_states[2] != state:
        checker.fail("非法草稿 token 没有保持上一个可推进状态")
    else:
        print("  ok   非法草稿保持上一个状态（无需 rollback）")

    # ---- 掩码缓存 --------------------------------------------------------
    print("\n[掩码缓存]")
    state = xml.initial_state()
    xml.mask(state)
    t0 = time.perf_counter()
    for _ in range(50):
        xml.mask(state)
    per_call = (time.perf_counter() - t0) / 50 * 1000
    print("  已缓存状态 %.3fms/次" % per_call)
    if per_call > 1.0:
        checker.fail("掩码缓存命中过慢：%.3fms/次" % per_call)

    if checker.failures:
        print("\n%d 项失败" % len(checker.failures))
        return 1
    print("\n全部通过（协议：%s）" % ", ".join(PROTOCOLS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
