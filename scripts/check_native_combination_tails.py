"""组合夹具的上下文尾部与 max_tokens=1/2 截断，验证单 token fallback 不遗留状态。"""
import argparse
import json
from pathlib import Path
import subprocess
from flashengine.deployment import read_json,write_json,sha256
from scripts.native_bench_common import compare_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("native");p.add_argument("bundles",help="check_native_combinations 生成的目录");p.add_argument("output");a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);checks=[];artifacts={}
    try:
        for label in ("dense-dflash","dense-eagle","moe-dflash","moe-eagle","hybrid-dflash","hybrid-moe-dflash"):
            bundle=Path(a.bundles)/(label+"-bundle");requests=[dict(id=str(i),prompt_token_ids=[7]*(length-1)+[52],max_tokens=limit,ignore_eos=True) for i,(length,limit) in enumerate([(59,5),(60,4),(61,3),(62,2),(63,1),(33,1),(33,2)])]
            w=root/(label+".workload.json");write_json(w,dict(schema_version=1,bundle_sha256=sha256(bundle/"bundle.json"),vocab_size=151936,config=dict(dtype="float16",max_model_len=64,max_num_seqs=4,max_num_batched_tokens=8,max_num_partial_prefills=2,kvcache_block_size=16,num_kvcache_blocks=32),requests=requests,synthetic_model=True))
            results={}
            for mode in ("eager","graph"):
                cmd=[str(Path(a.native).resolve()),"benchmark",str(bundle),"--workload",str(w),"--batch-size","4","--repeats","1","--warmup","0","--trace-rounds","--"+mode];r=subprocess.run(cmd,capture_output=True,text=True,timeout=180);write_json(root/(label+"-"+mode+".process.json"),dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
                if r.returncode:raise RuntimeError(label+" "+mode+" 尾部路径失败")
                value=json.loads(r.stdout);path=root/(label+"-"+mode+".json");write_json(path,value);artifacts[path.name]=sha256(path);results[mode]=value
                for row,request in zip(value["rows"],requests):
                    expected=[53 if j%2==0 else 52 for j in range(request["max_tokens"])]
                    if row["token_ids"]!=expected or row["finish_reason"]!="length":raise RuntimeError("尾部/输出限额不符合已知 token 转移")
                runtime=value["runtime_after"]
                if value["scheduler"]["used_blocks"] or runtime.get("recurrent_active_requests",0) or runtime.get("eagle_active_requests",0) or runtime.get("eagle_kv_active_requests",0):raise RuntimeError("尾部完成后状态未回收")
            checks.append(dict(test=label,**compare_rows(results["eager"]["rows"],results["graph"]["rows"],trace=True)));print("完成尾部 "+label,flush=True)
    except Exception as e:checks.append(dict(test="execution",passed=False,error=repr(e)))
    passed=bool(checks) and all(c["passed"] for c in checks);write_json(root/"summary.json",dict(passed=passed,release_qualified=False,binary_sha256=sha256(a.native),checks=checks,results=artifacts));print(root/"summary.json")
    if not passed:raise SystemExit(2)


if __name__=="__main__":main()
