"""小 KV 池强制前缀驱逐/抢占，比较 ON/OFF 和 eager/Graph；Qwen3 真模型或夹具。"""
import argparse
import json
from pathlib import Path
import subprocess
from flashengine.deployment import write_json,sha256,read_json
from scripts.native_bench_common import compare_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("native");p.add_argument("bundle");p.add_argument("output");args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);results={};comparisons=[]
    vocab=read_json(Path(args.bundle)/"engine/config.json")["target"]["vocab_size"]
    try:
        for cache in (False,True):
            workload=dict(schema_version=1,bundle_sha256=sha256(Path(args.bundle)/"bundle.json"),vocab_size=vocab,config=dict(dtype="float16",max_model_len=64,max_num_seqs=4,max_num_batched_tokens=8,max_num_partial_prefills=2,kvcache_block_size=16,num_kvcache_blocks=4,enable_prefix_cache=cache),requests=[dict(id=str(i),prompt_token_ids=ids,max_tokens=8,ignore_eos=True) for i,ids in enumerate(([7]*31,[7]*32+[8],[7]*47,[8]*31,[7]*31))])
            path=out/("on.json" if cache else "off.json");write_json(path,workload)
            for graph in (False,True):
                tag=("on" if cache else "off")+("-graph" if graph else "-eager");cmd=[str(Path(args.native).resolve()),"benchmark",args.bundle,"--workload",str(path),"--batch-size","4","--repeats","2","--warmup","0","--graph" if graph else "--eager"]
                r=subprocess.run(cmd,capture_output=True,text=True,timeout=120);write_json(out/(tag+".process.json"),dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
                if r.returncode:raise RuntimeError(tag+" 执行失败")
                results[tag]=json.loads(r.stdout.strip().splitlines()[-1]);write_json(out/(tag+".result.json"),results[tag]);s=results[tag]["scheduler"]
                if not results[tag]["preemptions"] or s["used_blocks"]:raise RuntimeError("未触发抢占或未回收 KV")
                if cache and (not s["prefix_hit_requests"] or not s["prefix_evictions"]):raise RuntimeError("未覆盖缓存命中和驱逐")
        for left,right in (("off-eager","on-eager"),("off-graph","on-graph"),("on-eager","on-graph")):
            comparisons.append(dict(left=left,right=right,**compare_rows(results[left]["rows"],results[right]["rows"])))
    except Exception as error:comparisons.append(dict(passed=False,error=str(error)))
    passed=bool(comparisons) and all(x["passed"] for x in comparisons);write_json(out/"summary.json",dict(passed=passed,release_qualified=False,binary_sha256=sha256(args.native),bundle_sha256=sha256(Path(args.bundle)/"bundle.json"),results={name:sha256(out/(name+".result.json")) for name in results},comparisons=comparisons));print(out/"summary.json")
    if not passed:raise SystemExit(2)


if __name__=="__main__":main()
