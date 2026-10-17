"""原生目标×投机×默认 Prefix 组合矩阵；保留原始输出和失败，不伪装训练模型验收。"""
import argparse
import json
from pathlib import Path
import subprocess
from flashengine.deployment import export_bundle,read_json,write_json,sha256,source_state
from scripts.native_bench_common import compare_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("native");p.add_argument("fixtures");p.add_argument("output");p.add_argument("--pressure",action="store_true");p.add_argument("--profiles",help="逗号分隔标签，只回归指定组合");a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);fixtures=Path(a.fixtures);native=str(Path(a.native).resolve());checks=[];results={};artifacts={}
    specs=[(model+"-"+(draft or "plain"),model,draft) for model in ("dense","moe","hybrid","hybrid-moe") for draft in ((None,"dflash","eagle") if model in ("dense","moe") else (None,"dflash"))]
    if a.profiles:
        selected=set(a.profiles.split(","))
        if selected-{x[0] for x in specs}:raise ValueError("未知组合")
        specs=[s for s in specs if s[0] in selected]
    def execute(tag,cmd):
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
            record=dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except subprocess.TimeoutExpired as e:
            record=dict(command=cmd,timed_out=True,stdout=(e.stdout or b"").decode(errors="replace"),stderr=(e.stderr or b"").decode(errors="replace"));write_json(root/(tag+".process.json"),record);raise
        write_json(root/(tag+".process.json"),record)
        if r.returncode:raise RuntimeError(tag+" 执行失败")
        value=json.loads(r.stdout.strip().splitlines()[-1]);path=root/(tag+".json");write_json(path,value);artifacts[tag]=sha256(path);return value
    try:
        for label,model,draft in specs:
            b=root/(label+"-bundle");export_bundle(fixtures/model,b,draft=fixtures/draft if draft else None,spec_num_tokens=2 if draft else None,hardlink=True,verifier=native)
            for cache in (False,True):
                config=dict(dtype="float16",max_model_len=64,max_num_seqs=4,max_num_batched_tokens=8,max_num_partial_prefills=2,kvcache_block_size=16,num_kvcache_blocks=4 if a.pressure else 32)
                # ON 刻意不传开关，验证原生缺省为 True，而不是测试工具替用户打开。
                if not cache:config["enable_prefix_cache"]=False
                prompts=[[7]*32+[52],[7]*32+[53],[7]*16+[52],[7]*48+[53],[7]*32+[52]]
                if a.pressure:prompts += [[7]*30+[52],[7]*46+[53]]  # 生成过程中跨满页，验证已接受行的发布边界。
                requests=[dict(id=str(i),prompt_token_ids=ids,max_tokens=8,ignore_eos=True) for i,ids in enumerate(prompts)]
                w=root/(label+("-on" if cache else "-off")+".workload.json");write_json(w,dict(schema_version=1,bundle_sha256=sha256(b/"bundle.json"),vocab_size=151936,config=config,requests=requests,synthetic_model=True))
                for graph in (False,True):
                    tag=label+("-on" if cache else "-off")+("-graph" if graph else "-eager");value=execute(tag,[native,"benchmark",str(b),"--workload",str(w),"--batch-size","4","--warmup","0","--repeats","2","--trace-rounds","--graph" if graph else "--eager"]);results[tag]=value
                    runtime=value["runtime_after"];scheduler=value["scheduler"]
                    if value["resolved_config"]["enable_prefix_cache"]!=cache:raise RuntimeError("默认前缀配置错误")
                    if scheduler["used_blocks"] or runtime.get("eagle_active_requests",0) or runtime.get("eagle_kv_active_requests",0) or runtime.get("recurrent_active_requests",0):raise RuntimeError("请求结束后状态或 KV 未释放")
                    if cache and (not scheduler["prefix_hit_requests"] or ((draft=="eagle" or model.startswith("hybrid")) and not runtime["prefix_state"]["restored"])):raise RuntimeError("未覆盖真实前缀和附属状态恢复")
                    if graph and (runtime["draft_graph_replays"]==0 or runtime["verify_graph_replays"]==0) and draft:raise RuntimeError("投机测量期没有 draft/verify 图回放")
                    if graph and not draft and not runtime["decode_graph_replays"]:raise RuntimeError("decode 未命中图")
            for left,right in (("off-eager","on-eager"),("off-graph","on-graph"),("on-eager","on-graph")):
                checks.append(dict(test=label+":"+left+":"+right,**compare_rows(results[label+"-"+left]["rows"],results[label+"-"+right]["rows"],trace=bool(draft) and not a.pressure)))
            plain=model+"-plain-on-graph"
            if draft and plain in results:checks.append(dict(test=label+":versus-target-greedy",**compare_rows(results[plain]["rows"],results[label+"-on-graph"]["rows"])))
            print("完成 "+label,flush=True)
    except Exception as error:checks.append(dict(test="execution",passed=False,error=repr(error)))
    passed=bool(checks) and all(c["passed"] for c in checks);write_json(root/"summary.json",dict(passed=passed,release_qualified=False,synthetic=True,pressure=a.pressure,binary_sha256=sha256(native),oracle_source=source_state(),results=artifacts,checks=checks));print(root/"summary.json")
    if not passed:raise SystemExit(2)


if __name__=="__main__":main()
