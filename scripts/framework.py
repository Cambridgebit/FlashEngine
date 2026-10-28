"""统一框架入口：配置、离线生成、HTTP 服务与服务管理。

``python -m flashengine <子命令>`` 与 ``python -m scripts.framework <子命令>`` 等价。
离线生成、Python API 与服务共用同一份 ``RuntimeSettings``（``configs/*.json``），命令行
覆盖优先级最高；未知配置项直接报错，不做静默忽略。

本模块只依赖标准库与 ``flashengine.runtime``（后者 import 期不依赖 torch），因此
``config``/``status``/``stop`` 在没有设备运行时的机器上也能运行。
"""

import argparse
import json
from pathlib import Path
import sys
import time

from flashengine.runtime import BACKENDS, PRECISIONS, QUANTIZATIONS, RuntimeSettings, create_engine


ROOT = Path(__file__).resolve().parents[1]
# 这个版本的统一入口用环境变量传 API key（`--api-key-env`），避免密钥出现在 argv/进程表。
DEFAULT_API_KEY_ENV = "FLASHENGINE_API_KEY"


def add_settings(parser):
    """向子命令追加统一配置覆盖项（未显式给出时沿用配置/内置默认值）。"""
    parser.add_argument("--config", type=Path, default=None,
                        help=f"统一 JSON 配置（示例见 {ROOT.name}/configs/）；省略时用内置默认值")
    parser.add_argument("--model", default=None, help="目标模型本地目录，覆盖配置")
    parser.add_argument("--backend", default=None, choices=list(BACKENDS),
                        help="cuda/corex/ascend310p；auto 为运行时探测")
    parser.add_argument("--precision", default=None, choices=list(PRECISIONS),
                        help="计算精度；auto 跟随模型 config（float16/bfloat16/float32）")
    parser.add_argument("--quantization", default=None, choices=list(QUANTIZATIONS),
                        help="检查点量化方式的声明值（实际以检查点元数据为准）")
    parser.add_argument("--fp16-only", action=argparse.BooleanOptionalAction, default=None,
                        dest="fp16_only", help="启用 FP16 专用守卫（拒绝量化检查点 + 参数审计）")
    parser.add_argument("--spec-model", default=None, help="匹配的 EAGLE3/DFlash/DSpark 目录；空字符串关闭")
    parser.add_argument("--spec-num-tokens", type=int, default=None, help="草稿长度，块草稿沿用自动派生")
    parser.add_argument("--max-model-len", type=int, default=None, help="上下文窗口")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="同时调度序列数")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None, help="Prefill 每步预算")
    parser.add_argument("--max-num-partial-prefills", type=int, default=None, help="部分 Prefill 并发上限")
    parser.add_argument("--gpu-mem-util", type=float, default=None, dest="gpu_memory_utilization",
                        help="设备内存预算比例")
    parser.add_argument("--block-size", type=int, default=None, dest="kvcache_block_size",
                        help="KV 块大小；310P 仅 64/128，cuda 默认要求 256 的倍数")
    parser.add_argument("--kv-blocks", type=int, default=None, dest="num_kvcache_blocks",
                        help="固定 KV 页数；默认按显存预算计算（-1）")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="TP 规模（1..8）")
    parser.add_argument("--device-ids", default=None, help="逗号分隔的设备索引，如 0,1")
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=None,
                        dest="enable_prefix_cache", help="前缀缓存开关（按后端能力裁决）")
    parser.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, default=None,
                        dest="enable_chunked_prefill", help="分块 Prefill 开关")
    parser.add_argument("--eager", action=argparse.BooleanOptionalAction, default=None,
                        dest="enforce_eager", help="关闭图，使用 eager 回退")
    parser.add_argument("--decode-linear-backend", choices=("functional", "selective"), default=None,
                        help="单 token FP16 Linear 路径（仅 ascend310p 生效）")
    parser.add_argument("--greedy-backend", choices=("npu", "cpu"), default=None,
                        help="greedy 归约位置（仅 ascend310p 生效）")
    parser.add_argument("--cpu-threads", type=int, default=None, help="主机侧 CPU 线程数（LITEQ_CPU_THREADS）")
    parser.add_argument("--pa-graph-update", choices=("auto", "0", "1"), default=None,
                        help="FLASHENGINE_ACLGRAPH_PA_UPDATE 覆盖")
    parser.add_argument("--host", default=None, help="HTTP 监听地址")
    parser.add_argument("--port", type=int, default=None, help="HTTP 监听端口")
    parser.add_argument("--max-tokens", type=int, default=None, help="离线输出上限；HTTP 请求上限由请求体指定")
    parser.add_argument("--temperature", type=float, default=None, help="0 为 greedy")
    parser.add_argument("--top-p", type=float, default=None, help="随机采样 top-p")
    parser.add_argument("--top-k", type=int, default=None, help="随机采样 top-k，0 关闭")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                        dest="enable_thinking", help="Qwen3 模板思考开关")
    parser.add_argument("--reserve-output", type=int, default=None, help="为输出预留的上下文 token 数")
    parser.add_argument("--stats-interval", type=float, default=None, help="[INFO] 聚合周期秒数，0 关闭")
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=None, dest="web_ui",
                        help="内置 Web 控制台开关")
    parser.add_argument("--api-key-env", default=None,
                        help=f"API key 所在的环境变量名（默认 {DEFAULT_API_KEY_ENV}）")


def settings_from_args(args):
    """按 CLI 覆盖配置文件的优先级组装并校验统一设置。"""
    names = RuntimeSettings.__dataclass_fields__
    overrides = {key: value for key, value in vars(args).items() if key in names}
    if isinstance(overrides.get("device_ids"), str):
        overrides["device_ids"] = tuple(
            int(part) for part in overrides["device_ids"].split(",") if part.strip())
    return RuntimeSettings.load(args.config, **overrides)


def _require_model(settings, command):
    if not settings.model:
        raise ValueError(f"{command} 需要本地模型目录：请通过 --config 的 model 字段或 --model 提供")


def server_arguments(settings):
    """把统一设置翻译成既有 HTTP 服务的命令行参数（不复制其调度/流式/投机实现）。

    API key 只传环境变量名，绝不把明文放进 argv；主项目服务已有的 ``--raw-log``、
    ``--no-prefix-cache``、``--kv-blocks`` 等开关仍可直接使用。
    """
    arguments = [
        settings.model,
        "--backend", settings.backend,
        "--host", settings.host,
        "--port", str(settings.port),
        "--max-model-len", str(settings.max_model_len),
        "--max-num-seqs", str(settings.max_num_seqs),
        "--max-num-batched-tokens", str(settings.max_num_batched_tokens),
        "--max-num-partial-prefills", str(settings.max_num_partial_prefills),
        "--gpu-mem-util", str(settings.gpu_memory_utilization),
        "--block-size", str(settings.kvcache_block_size),
        "--spec-model", settings.spec_model,
        "--spec-num-tokens", str(settings.spec_num_tokens),
        "--decode-linear-backend", settings.decode_linear_backend,
        "--greedy-backend", settings.greedy_backend,
        "--temperature", str(settings.temperature),
        "--top-p", str(settings.top_p),
        "--top-k", str(settings.top_k),
        "--reserve-output", str(settings.reserve_output),
        "--stats-interval", str(settings.stats_interval),
        "--recurrent-prefix-cache-mb", str(settings.recurrent_prefix_cache_bytes // (1024 * 1024)),
        "--api-key-env", settings.api_key_env,
        "--prefix-cache" if settings.enable_prefix_cache else "--no-prefix-cache",
    ]
    if settings.precision != "auto":
        arguments += ["--dtype", settings.precision]
    if settings.num_kvcache_blocks >= 0:
        arguments += ["--kv-blocks", str(settings.num_kvcache_blocks)]
    if settings.fp16_only:
        arguments.append("--fp16-only")
    for enabled, flag in ((settings.enforce_eager, "--eager"),
                          (not settings.enable_chunked_prefill, "--disable-chunked-prefill"),
                          (not settings.enable_thinking, "--no-think"),
                          (not settings.web_ui, "--no-ui")):
        if enabled:
            arguments.append(flag)
    return arguments


def synchronize(engine):
    """压测/计时边界同步设备（主项目暂无 ``LLMEngine.synchronize()`` 的公开入口）。"""
    backend = getattr(getattr(engine, "model_runner", None), "backend", None)
    if backend is not None and hasattr(backend, "synchronize"):
        backend.synchronize()


def generate(settings, args):
    """离线生成：加载模型→生成→打印并可选保存文本/token/运行快照，结束后释放设备。"""
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    if not prompt:
        raise ValueError("请通过 --prompt 或 --prompt-file 提供非空输入")
    from flashengine import SamplingParams

    with create_engine(settings) as engine:
        messages = ([{"role": "system", "content": args.system}] if args.system else [])
        messages.append({"role": "user", "content": prompt})
        text = engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=settings.enable_thinking)
        ids = engine.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) + settings.max_tokens > engine.config.max_model_len:
            raise ValueError("输入加输出超过上下文窗口；请缩短输入或减小 --max-tokens")
        params = SamplingParams(temperature=settings.temperature, top_p=settings.top_p,
                                top_k=settings.top_k, max_tokens=settings.max_tokens)
        synchronize(engine)
        started = time.perf_counter()
        output = engine.generate([ids], params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - started
        runtime = engine.runtime_snapshot()
        result = {"text": output["text"], "token_ids": output["token_ids"], "wall_s": elapsed,
                  "completion_tokens": len(output["token_ids"]),
                  "precision": runtime["engine"]["dtype"],
                  "quantization": (runtime.get("precision") or {}).get("quantization", settings.quantization),
                  "runtime": runtime}
        print(output["text"], flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0


# 子命令 → 帮助文本；顺序即 ``--help`` 中的展示顺序。
_COMMANDS = (
    ("config", "展示最终配置，不加载设备"),
    ("generate", "离线生成"),
    ("serve", "前台运行 HTTP 与 Web UI"),
    ("start", "后台启动并等待就绪"),
    ("status", "读取本框架服务状态"),
    ("stop", "验证 PID 身份后优雅停止本框架"),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in _COMMANDS:
        child = sub.add_parser(name, help=help_text)
        add_settings(child)
        if name == "generate":
            group = child.add_mutually_exclusive_group(required=True)
            group.add_argument("--prompt", help="用户输入")
            group.add_argument("--prompt-file", type=Path, help="UTF-8 输入文件")
            child.add_argument("--system", default="", help="可选 system 提示词")
            child.add_argument("--output", type=Path, help="保存文本、token 与运行快照的 JSON 路径")
        if name in {"start", "stop", "status"}:
            child.add_argument("--timeout", type=float, default=240 if name == "start" else 60,
                               help="就绪/退出等待秒数")
        if name == "serve":
            child.add_argument("--raw-log", default=None, metavar="PATH",
                               help="把模型原始输出以 JSONL 追加写入该文件（透传既有服务开关）")
            child.add_argument("--webui", default=None, metavar="HTML",
                               help="Web 控制台 HTML 路径（透传既有服务开关）")
    args = parser.parse_args(argv)
    try:
        settings = settings_from_args(args)
        if args.command == "config":
            print(json.dumps(settings.as_dict(), ensure_ascii=False, indent=2))
            return 0
        settings.apply_environment()
        if args.command in {"start", "stop", "status"}:
            if args.command == "start":
                _require_model(settings, "start")
            from scripts.service_control import control
            return control(args.command, settings, args.timeout)
        if args.command == "generate":
            _require_model(settings, "generate")
            return generate(settings, args)
        if args.command == "serve":
            _require_model(settings, "serve")
            from scripts.openai_server import main as serve
            extra = []
            if getattr(args, "raw_log", None):
                extra += ["--raw-log", args.raw_log]
            if getattr(args, "webui", None):
                extra += ["--webui", args.webui]
            return serve(server_arguments(settings) + extra) or 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FRAMEWORK_ERROR {exc}", file=sys.stderr, flush=True)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
