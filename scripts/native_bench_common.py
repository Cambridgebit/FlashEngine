"""Python/C++ 基准共用的纯主机输入校验、统计和严格比较。"""
import subprocess
import time

from scripts.bench_utils import percentile


def validate_workload(workload):
    """固定输入，不允许在两端分别裁剪或重新分词。"""
    if workload.get("schema_version") != 1 or not workload.get("requests"):
        raise ValueError("workload 版本错误或请求为空")
    cfg = workload["config"]
    if cfg["dtype"] not in ("float16", "bfloat16"):
        raise ValueError("非法 dtype")
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "max_num_partial_prefills", "kvcache_block_size", "num_kvcache_blocks"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"非法配置：{key}")
    seen = set()
    for item in workload["requests"]:
        if not isinstance(item["id"], str) or item["id"] in seen:
            raise ValueError("workload id 非法/重复")
        seen.add(item["id"])
        ids, limit = item["prompt_token_ids"], item["max_tokens"]
        if not ids or type(limit) is not int or limit < 1 or len(ids) + limit > cfg["max_model_len"]:
            raise ValueError("请求超出上下文，禁止静默裁剪")
        if any(type(i) is not int or not 0 <= i < workload["vocab_size"] for i in ids):
            raise ValueError("token ID 越界")
        if type(item["ignore_eos"]) is not bool:
            raise ValueError("ignore_eos 必须为 bool")


def summary(rows, wall):
    """吞吐按整个主循环墙钟，TTFT 包含组内排队；不把 chunk 数当 token 数。"""
    if not rows or wall <= 0:
        raise ValueError("空基准或无效墙钟")
    count = sum(len(row["token_ids"]) for row in rows)
    ttft = [row["ttft_seconds"] * 1000 for row in rows]
    tpot = [(row.get("e2e_seconds", row["group_wall_seconds"]) - row["ttft_seconds"]) * 1000 / (len(row["token_ids"]) - 1) for row in rows if len(row["token_ids"]) > 1]
    return dict(output_tokens=count, output_tps=count / wall, ttft_p50_ms=percentile(ttft, 50), ttft_p95_ms=percentile(ttft, 95), tpot_p50_ms=percentile(tpot, 50), requests=len(rows))


def checked_greedy_round(draft, verify, committed, context_length):
    """trace 必须描述接受时的提案，不能读到 refill 覆盖后的下一轮首 token。"""
    if len(verify) != len(draft) + 1:
        raise ValueError("verify 行数与提案不匹配")
    accepted = 0
    while accepted < len(draft) and draft[accepted] == verify[accepted]:
        accepted += 1
    expected = draft[:accepted] + [verify[accepted]]
    if committed != expected:
        raise ValueError("trace 的提案/verify 与实际提交不自洽")
    return dict(draft_ids=draft, verify_ids=verify, accepted_count=accepted,
                committed_ids=committed, context_length=context_length)


def compare_rows(reference, observed, trace=False):
    """只允许请求/物理块顺序不同；按稳定输入 id 和重复轮次对齐。"""
    def index(rows):
        out = {}
        for row in rows:
            key = (row["repeat"], row["input_id"])
            if key in out:
                raise ValueError("重复结果 id")
            out[key] = row
        return out
    expected, actual = index(reference), index(observed)
    diagnostics = []
    for key in sorted(expected.keys() | actual.keys()):
        a, b = expected.get(key), actual.get(key)
        if a is None or b is None:
            diagnostics.append(dict(key=key, passed=False, reason="缺失请求"))
            continue
        x, y = a["token_ids"], b["token_ids"]
        prefix = next((j for j, values in enumerate(zip(x, y)) if values[0] != values[1]), min(len(x), len(y)))
        ok = x == y and a["finish_reason"] == b["finish_reason"]
        item = dict(key=key, passed=ok, common_prefix=prefix, lengths=[len(x), len(y)])
        if trace:
            if "rounds" not in a or "rounds" not in b:
                item.update(passed=False, reason="缺少逐轮 trace")
            else:
                ar, br = a["rounds"], b["rounds"]
                item["round_counts"] = [len(ar), len(br)]
                item["passed"] = ok and ar == br
                for step, (left, right) in enumerate(zip(ar, br)):
                    if left != right:
                        item["first_round_difference"] = dict(round=step, oracle=left, native=right)
                        break
        diagnostics.append(item)
    return dict(passed=bool(diagnostics) and all(row["passed"] for row in diagnostics), cases=diagnostics)


def gpu_sample():
    """热状态只采集证据，不更改设备功耗或时钟。"""
    query = "name,temperature.gpu,clocks.sm,memory.used,power.draw"
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        return dict(unix_time=time.time(), exit_code=p.returncode, values=p.stdout.strip(), error=p.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(unix_time=time.time(), error=repr(error))
