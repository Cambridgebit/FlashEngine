"""顺序执行原生新增 profile 验收；保存命令、原始输出和严格比较，不掩盖失败。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from flashengine.deployment import export_bundle,read_json,sha256,source_state,write_json
from scripts.native_bench_common import compare_rows
from scripts.audit_native_benchmark import graph_progress


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("native");p.add_argument("fixtures");p.add_argument("output");p.add_argument("--python",action="store_true",help="同时运行 Python 引擎逐轮差分");p.add_argument("--profiles",help="仅运行逗号分隔的 profile；缺省运行全部");args=p.parse_args()
    root=Path(args.output)
    if root.exists():raise ValueError("拒绝覆盖验收目录")
    root.mkdir(parents=True);native=str(Path(args.native).resolve());fixtures=Path(args.fixtures).resolve();records=[]
    fixture_metadata=read_json(fixtures/"fixture.json")
    def execute(label,cmd):
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
        write_json(root/(label+".process.json"),dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise RuntimeError(f"{label} 失败，查看归档 stderr")
        return result
    def bundle(label,model,draft=None):
        dest=root/(label+"-bundle");export_bundle(model,dest,draft=draft,spec_num_tokens=2 if draft else None,hardlink=True,verifier=native);return dest
    def workload(label,b,cache=False):
        prefix=label.startswith("prefix")
        cfg=dict(dtype="float16",max_model_len=512 if prefix else 64,max_num_seqs=4,max_num_batched_tokens=64 if prefix else 8,max_num_partial_prefills=2,kvcache_block_size=256,num_kvcache_blocks=8,enable_prefix_cache=cache)
        prompts=[[1,2,3],[7]*16+[8],[7]*16+[9],[2,3,4,5,6],[7]*32+[10]]
        if prefix:prompts=[[7]*256+[8],[7]*256+[9],[7]*256+[10],[7]*255+[6,10],[7]*256+[11]]
        rows=[dict(id=str(i),prompt_token_ids=ids,max_tokens=8,ignore_eos=True) for i,ids in enumerate(prompts)]
        value=dict(schema_version=1,bundle_sha256=sha256(b/"bundle.json"),vocab_size=151936,config=cfg,requests=rows,synthetic_model=True);out=root/(label+".workload.json");write_json(out,value);return out
    results={}
    specs=[("dense","dense",None),("prefix","dense",None),("prefix-off","dense",None),("eagle","dense","eagle"),("quark","quark",None),("quark-reference","quark-reference",None),("hybrid","hybrid",None),("hybrid-moe","hybrid-moe",None)]
    if args.profiles:
        selected=set(args.profiles.split(","))
        if selected-{x[0] for x in specs}:raise ValueError("未知 profile")
        specs=[x for x in specs if x[0] in selected]
    try:
        for label,model,draft in specs:
            b=bundle(label,fixtures/model,fixtures/draft if draft else None);w=workload(label,b,label=="prefix")
            for graph in (False,True):
                tag=label+("-graph" if graph else "-eager");cmd=[native,"benchmark",str(b),"--workload",str(w),"--batch-size","4","--warmup","0","--repeats","2","--trace-rounds","--graph" if graph else "--eager"]
                process=execute(tag,cmd);result=json.loads(process.stdout.strip().splitlines()[-1]);write_json(root/(tag+".json"),result);results[tag]=result
                if graph:
                    metric="draft_graph_replays" if draft else "decode_graph_replays"
                    if result["runtime_after"][metric] <= result["runtime_before"][metric]:raise RuntimeError("测量期未命中 Graph")
                    if draft and result["runtime_after"]["verify_graph_replays"] <= result["runtime_before"]["verify_graph_replays"]:raise RuntimeError("target verify 未命中 Graph")
                if result["scheduler"]["used_blocks"] != 0:raise RuntimeError("结束后 KV 引用未归零")
                if label=="prefix" and result["scheduler"]["prefix_hit_requests"] <= 0:raise RuntimeError("前缀未命中")
                if label.startswith("hybrid") and result["runtime_after"]["recurrent_active_requests"] != 0:raise RuntimeError("递归状态未释放")
                if label.startswith("hybrid") and fixture_metadata.get("hybrid_nonconstant") and any(len(set(row["token_ids"]))<2 for row in result["rows"]):raise RuntimeError("hybrid 输出未随 token 改变，疑似读取陈旧 Graph 输出")
                if draft and result["runtime_after"]["eagle_active_requests"] != 0:raise RuntimeError("EAGLE3 状态未释放")
                if args.python and label!="prefix":
                    py=root/(tag+".python.json");command=[sys.executable,"-B","-m","scripts.bench_python_native",str(fixtures/model),str(b),str(w),str(py),"--batch-size","4","--warmup","0","--repeats","2","--trace-rounds"]
                    if draft:command += ["--draft",str(fixtures/draft)]
                    if graph:command += ["--graph"]
                    execute(tag+"-python",command);reference=read_json(py)
                    if graph:
                        progress=graph_progress(reference)
                        if progress["replays"]<=0 or progress.get("capture_failures",0) or progress.get("mode")!=("spec_graph" if draft else "decode_graph"):
                            raise RuntimeError("Python --graph 未实际命中稳态图，不能把 eager 回退作为图对照")
                    comparison=compare_rows(reference["rows"],result["rows"],trace=bool(draft));records.append(dict(test=tag+"-python",**comparison))
            records.append(dict(test=label+"-eager-graph",**compare_rows(results[label+"-eager"]["rows"],results[label+"-graph"]["rows"],trace=bool(draft))))
        for mode in ("eager","graph"):
            for label,left,right in (("prefix-output","prefix-off","prefix"),("quark-dequant","quark-reference","quark")):
                if left+"-"+mode in results and right+"-"+mode in results:
                    records.append(dict(test=label+"-"+mode,**compare_rows(results[left+"-"+mode]["rows"],results[right+"-"+mode]["rows"])))
    except Exception as error:
        records.append(dict(test="execution",passed=False,error=str(error)))
    write_json(root/"summary.json",dict(passed=bool(records) and all(r["passed"] for r in records),release_qualified=False,tests=records,binary_sha256=sha256(native),oracle=source_state()))
    print(root/"summary.json")
    if not records or not all(r["passed"] for r in records):raise SystemExit(2)


if __name__=="__main__":main()
