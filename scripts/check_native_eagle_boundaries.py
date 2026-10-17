"""EAGLE3 上下文尾部和轮内输出上限检查；固定高 margin 夹具，期待 token=52。"""
import argparse
import json
from pathlib import Path
import subprocess
from flashengine.deployment import sha256,write_json
from scripts.native_bench_common import compare_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("native");p.add_argument("bundle");p.add_argument("output");args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False);results={};checks=[]
    workload=dict(schema_version=1,bundle_sha256=sha256(Path(args.bundle)/"bundle.json"),vocab_size=151936,config=dict(dtype="float16",max_model_len=64,max_num_seqs=4,max_num_batched_tokens=8,max_num_partial_prefills=2,kvcache_block_size=16,num_kvcache_blocks=16),requests=[dict(id=str(length),prompt_token_ids=[7]*length,max_tokens=64-length,ignore_eos=True) for length in (60,61,62,63)])
    path=root/"workload.json";write_json(path,workload)
    try:
        for graph in (False,True):
            mode="graph" if graph else "eager";cmd=[str(Path(args.native).resolve()),"benchmark",args.bundle,"--workload",str(path),"--batch-size","4","--repeats","1","--warmup","0","--trace-rounds","--graph" if graph else "--eager"]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=120);write_json(root/(mode+".process.json"),dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
            if r.returncode:raise RuntimeError("上下文尾部路径失败："+mode)
            value=json.loads(r.stdout.strip().splitlines()[-1]);write_json(root/(mode+".json"),value);results[mode]=value
            ok=all(row["token_ids"]==[52]*(64-int(row["input_id"])) and row["finish_reason"]=="length" for row in value["rows"])
            checks.append(dict(test=mode,passed=ok and value["runtime_after"]["eagle_active_requests"]==0 and value["scheduler"]["used_blocks"]==0))
        checks.append(dict(test="eager-graph",**compare_rows(results["eager"]["rows"],results["graph"]["rows"],trace=True)))
    except Exception as error:checks.append(dict(passed=False,error=str(error)))
    passed=bool(checks) and all(c["passed"] for c in checks);write_json(root/"summary.json",dict(passed=passed,release_qualified=False,checks=checks,binary_sha256=sha256(args.native)));print(root/"summary.json")
    if not passed:raise SystemExit(2)


if __name__=="__main__":main()
