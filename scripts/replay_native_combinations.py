"""最终二进制重放六种投机组合的热前缀 Graph 压力结果，并逐轮比较旧报告。"""
import argparse
import json
from pathlib import Path
import subprocess
from flashengine.deployment import read_json,write_json,sha256
from scripts.native_bench_common import compare_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("native");p.add_argument("previous");p.add_argument("output");a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);previous=Path(a.previous);checks=[];artifacts={}
    try:
        for label in ("dense-dflash","dense-eagle","moe-dflash","moe-eagle","hybrid-dflash","hybrid-moe-dflash"):
            cmd=[str(Path(a.native).resolve()),"benchmark",str(previous/(label+"-bundle")),"--workload",str(previous/(label+"-on.workload.json")),"--batch-size","4","--warmup","0","--repeats","2","--trace-rounds","--graph"]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=180);write_json(root/(label+".process.json"),dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
            if r.returncode:raise RuntimeError(label+" 重放失败")
            value=json.loads(r.stdout);path=root/(label+".json");write_json(path,value);artifacts[path.name]=sha256(path)
            runtime=value["runtime_after"];s=value["scheduler"]
            if not value["resolved_config"]["enable_prefix_cache"] or not s["prefix_hit_requests"] or s["used_blocks"] or not runtime["draft_graph_replays"] or not runtime["verify_graph_replays"]:raise RuntimeError("前缀/Graph/回收门禁失败")
            if runtime.get("recurrent_active_requests",0) or runtime.get("eagle_active_requests",0) or runtime.get("eagle_kv_active_requests",0):raise RuntimeError("投机状态未释放")
            checks.append(dict(test=label,**compare_rows(read_json(previous/(label+"-on-graph.json"))["rows"],value["rows"],trace=True)));print("重放 "+label,flush=True)
    except Exception as e:checks.append(dict(test="execution",passed=False,error=repr(e)))
    passed=bool(checks) and all(c["passed"] for c in checks);write_json(root/"summary.json",dict(passed=passed,release_qualified=False,binary_sha256=sha256(a.native),checks=checks,results=artifacts));print(root/"summary.json")
    if not passed:raise SystemExit(2)


if __name__=="__main__":main()
