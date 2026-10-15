# Qwen3.5 系列文本推理

当前支持 Qwen3.5 dense 和 MoE 的文本生成，目标注册名为 `qwen3_5` / `qwen3_5_moe`。
读取多模态检查点的 `text_config` 或纯文本配置，跳过视觉塔和 MTP。
不支持图像/视频输入。普通推理和匹配的 DFlash 草稿适配 CUDA、CoreX、Ascend 310P。
当前 TP=1、fp16/bf16；EAGLE3、DSpark 和量化检查点明确拒绝。

## 实现

- 全注意力复用三后端 Attention/KV 接口，只为 full_attention 层分配 KV。
  Qwen3.5-2B 的结构为 18 个 GDN 层和 6 个 full_attention 层。
- 门控全注意力使用 partial NeoX RoPE（2B 为 64/256）、Gemma RMSNorm（weight+1）
  和逐头输出门控；decode 的 attention 输出先归形，避免广播膨胀。
- GDN 采用经过 Transformers 参考公式验证的 chunk/recurrent 实现。
  CUDA 默认融合递推、短卷积、Gemma/GDN Norm 与设备状态提交；
  FlashAttention 使用融合 prefill，FlashInfer auto 保留参考 prefill。
  conv/SSM 状态与请求生命周期绑定，分块 prefill 复用此前的状态。
- `RecurrentStateRuntime` 分配请求主槽、verify 行槽和 dummy 槽。
  DFlash 验证后把 `count-1` 行终态拷回主槽（CUDA 在图内提交，其他后端在读回后提交）；bonus 留给下一轮消费。
  完成、抢占、取消、预热收尾均释放槽映射。
- DFlash aux 为 `target_layer_ids` 指定的 0-based 层输出，不采用 Qwen3 的 +1。
  MoE 添加门控共享专家，路由专家复用原三后端 `fused_moe`。
- prefix cache 默认开启：完整块边界保存 conv/SSM 主机快照，命中后恢复末块状态。
  DFlash 同时恢复草稿上下文 KV；任何必要状态缺失都会缩短命中或重算。
  快照预算、生命周期、实机回归见 [Qwen3.5 前缀缓存](QWEN35_PREFIX_CACHE.md)。
- 接近上下文上限、剩余空间不足一整轮时走单 token target fallback，
  避免固定宽度 verify 越界访问 KV。计数为 `context_limited_rounds`。

## 各后端状态

| 后端 | 代码路径 | 验证边界 |
|---|---|---|
| CUDA / FlashAttention | 普通 decode Graph、DFlash 整轮 Graph | 本地真实权重已验收 |
| CUDA / FlashInfer | 普通 decode Graph、DFlash eager | 本地真实权重已验收 |
| CoreX | IXFormer 全注意力、共享 GDN、CUDA 兼容 Graph、设备 RNG | 无实机，待厂商环境验收 |
| Ascend 310P | fp16/NZ 全注意力、静态短卷积展开、共享 GDN、NPUGraph 状态绑定 | Qwen3.5-4B + DFlash 实机已验收（2026-10-14/15）；其余组合待测 |

310P 的 head_dim=256 自动选 KV block=64，以满足 block_size×head_dim≤16384。
短卷积不用数千组 depthwise Conv1d，改为固定 kernel 长度的 fp32 乘加展开。
GDN 采用独立 ND 状态，full_attention KV 仍由后端分配 FRACTAL_NZ。
所有图捕获仍按既有机制降级；代码放行不等于三种硬件均已实机验收。

## 310P 实机性能：4B DFlash k 甜点与 8B/4B 加速对比（2026-10-15）

口径（8B 与 4B 同命令、同板同日同会话）：

```bash
python3 -m scripts.benchmark <目标> --backend ascend310p --no-think --gpu-mem-util 0.5 \
  --max-model-len 2048 --max-num-seqs 1 --warmup 1 --num-prompts 3 --max-tokens 64 \
  --questions bench_data/question.jsonl --no-prefix-cache \
  [--spec-model <DFlash 草稿>] [--spec-num-tokens K]
```

指标 = Steady-state decode throughput（排除 TTFT/prefill，vLLM bench 同口径）；
batch=1、温度 0、spec 图 100% 回放（0 回退）。“轮时” = E / 吞吐。

| 配置 | tok/s | E | 加速比 | 轮时 | 备注 |
|---|---:|---:|---:|---:|---|
| Qwen3-8B 无投机 | 10.9 | — | 1× | 91.7ms | 16.4GB 权重流 @ ~179GB/s |
| Qwen3-8B + DFlash k15 | **38.0** | 4.878 | **3.49×** | 128.4ms | |
| Qwen3-8B + DFlash k2 | 21.4 | 2.494 | 1.96× | 116.5ms | |
| Qwen3.5-4B 无投机 | 15.7 | — | 1× | 63.7ms | |
| Qwen3.5-4B + DFlash k2 | 27.4 | 2.595 | 1.75× | 94.6ms | |
| Qwen3.5-4B + DFlash k7 | **36.6** | 4.304 | **2.33×** | 117.6ms | **甜点** |
| Qwen3.5-4B + DFlash k10 | 33.0 | 4.370 | 2.10× | 132.4ms | |
| Qwen3.5-4B + DFlash k15 | 30.2 | 4.705 | 1.92× | 155.8ms | 默认值，非最优 |

k 曲线（4B）：每行验证成本线性 ≈ **4.7ms/行**；E 在 k≈7 后饱和（k7→k15 多验 8 行只多 0.4 接受）。
**4B 建议显式 `--spec-num-tokens 7`（比默认 k=15 快 21%）**。
注意 CLI 的 k=3/4 视为未显式指定，本次扫描的 k=4 实际执行了默认 k=15（E 与 TPS 同 k15 逐位一致）。
2026-10-14 的“k=2 与 k=15 打平”结论在 GDN 下标修复后不再成立——修复主要提速 rows 路径，长 k 受益更多。

为什么 8B 加速 3.5× 而 4B 只有 2×：加速比 = E × T_单步 / T_轮；两边 E 相当（4.7~4.9），
差距全在每轮成本（轮/单步比：8B 1.40×、4B 2.45×）：

- 8B（纯注意力）验证 16 行 ≈ 同一份权重流一次（单步 91.7ms 已是带宽地板），额外行几乎免费（0.9ms/行）。
- 4B 的 24 层 GDN 状态递推逐行串行：310P 走参考路径（无 `supports_fused_gdn`），
  每层每行 ~6 个 2MB fp32 小算子；图内标定 rows 层 seg=8 = 1588us vs decode 层 927us
  （+94us/行/层 → 整模 +2.3ms/行，约占行边际一半）。CUDA 的 triton 融合核
  （状态驻留寄存器、row 循环在 kernel 内、每层一次启动）在 310P 不存在——
  CUDA DFlash 71.7→134.6 tok/s 的优化吃的正是这个杠杆。
- 另有固定开销：k2 轮 94.6ms vs 单步 63.7ms（+31ms = 草稿 1.18GiB 前向 + context 投影/分页写 +
  采样/接受/提交/读回），对小模型占比更大。

空间估算：GDN 行融合把行边际 4.7→~2.4ms，k7 轮时 117.6→~107ms ≈ **40 tok/s（2.6×）**；
4×（需轮≈单步）不可达。

### GDN 位等微优化与算子地板（2026-10-15）

已落地全位等微优化（`g_exp` 整段一次、rank-1 就地 `add_`、`has_initial` 零张量复用、
conv1d 输入/权重各一次 fp32 转换）：层内 rows seg=8 1574→1565us、decode 911→896us；
端到端无投机 15.7→15.8、k7 36.6→36.8（E=4.304 逐位不变）；三处位等校验（conv1d 三形状、
rows/decode 输出 + ssm/conv 双状态池）与 check_qwen35 四用例 5/5 全过。

图内逐算子目录（NPUGraph、链式、310P）：

| 算子 | 成本 |
|---|---:|
| 2MB fp32 elementwise（mul/add 链式） | ≈6.1us/op |
| 2MB `(x * k).sum(-2)` 组合 | ≈21.6us |
| 2MB `index_select` / `index_copy_` / 切片 int 写 | 9.1 / 44–49 / 8.6us |
| 2MB `index_put_` / `scatter_(expand)` / `torch.gather(expand)` | 578 / ~8550 / ~7970us |
| tiny 依赖链 | ≈1.1–2us/op |
| 40KB 级单发算子（cast/silu/where/cat/contiguous） | 6.4–11.9us |

分段（rows 层 seg=8 = 1589us）：投影+尾 534（权重主导）+ conv 段 198 + 拆头/l2norm/gating 73.5 +
行递推 783（97.9us/行）。行循环在环消融（713us / 8 行）：kv 收缩 19.8/行 + out 收缩 17.8 +
池写 16.8 + gexp 乘 10.8 + outer/add ≈24。
判决：行递推 ≈10 个算子 × ~9us/行，已贴 torch 算子地板；否决清单（addcmul/unfold/GQA 视图/
融合池写两式/index_put_/scatter 系/…）与仅剩的两条大杠杆（融合自定义算子、放宽位等契约）见 AGENTS.md。
标定与工件：板卡 `/data2/fe_gdn_calib5-10.py` + `q35_gdn_calib*.log`、A/B `fe_q35_ab_patch.sh`。

板卡工件：`/data2/fe_q35_repeat.sh`（A/B）、`/data2/q35_repeat_final.log`、`/data2/q35_kscan.log`（k 扫描）、
`/data2/fe_8b_df_cmp.sh`（8B 对照）、`/data2/fe_gdn_calib*.py`（行成本标定）。

## WSL 启动

先激活匹配 PyTorch 的环境。Qwen3.5 配置解析需要 Transformers 5.6 或更新版本。
安装清单已恢复为 `pyproject.toml` / `requirements.txt`，FlashInfer 可用 `.[flashinfer]` extra。

```bash
cd /mnt/d/Materials/AVIC/FlashEngine
source /home/yjq040828/vllm-env/bin/activate
python3 -m scripts.openai_server /home/yjq040828/vllm-lab/models/Qwen3.5-2B \
  --backend cuda --host 127.0.0.1 --no-think --max-model-len 512 \
  --max-num-seqs 1 --max-num-batched-tokens 128 --gpu-mem-util 0.85 \
  --spec-model /home/yjq040828/vllm-lab/models/Qwen3.5-2B-Dflash
```

去掉 `--spec-model` 为普通推理；设置 `FLASHENGINE_CUDA_ATTENTION=flashinfer` 切换注意力库。
310P 用 `--backend ascend310p`（自动选择 64-token 页），CoreX 用 `--backend corex`。
8GB 卡先使用 seqs=1；GDN 状态池随并发及 verify 宽度增长，不能直接使用默认大容量。

## 本轮验证

本节为初始适配记录；后续优化的 56/56 回归、长输出对照与已知精度边界见
[DFlash 性能修复](DFLASH_PERFORMANCE.md)，不能把初始的 66-token 冒烟当稳定性能结论。

本地环境：Ubuntu-22.04 / RTX 4070 Laptop 8GB，torch 2.11.0、transformers 5.15，
真实模型目录 Qwen3.5-2B 与 Qwen3.5-2B-Dflash，bf16。

```bash
# （unittest 套件未随 main 保留：python3 -m unittest discover -s tests -v）
python3 -m scripts.check_qwen35 /home/yjq040828/vllm-lab/models/Qwen3.5-2B \
  --draft /home/yjq040828/vllm-lab/models/Qwen3.5-2B-Dflash --max-tokens 16
```

HF 文本参考、base eager/Graph、DFlash eager/Graph、FlashInfer base Graph/DFlash eager
七条路径，五个提示词逐 token 一致。默认 k=15；DFlash eager 与 Graph 的接受统计守恒，
同一回归集 E=4.211，Graph 主跑 19 次回放、0 eager fallback。
另一次 coding 两条/输出32 的 eager 冒烟 E=5.333。

原完整 unittest 为 47/47 通过（未随 main 保留）。回归集串行五请求、66 个实际输出 token、各请求墙钟求和（包含 prefill）：

| 路径 | token/s |
|---|---:|
| base eager | 16.42 |
| base Graph | 28.26 |
| DFlash eager | 21.09 |
| DFlash Graph | 33.15 |

这组小样本中 DFlash Graph 比 base Graph 高约 17%；不是稳定性能承诺。
题集、输出长度、缓存、thinking、预热、温度与硬件频率须一致后才能对表。

附加回归覆盖普通 decode 的 3→4 桶 padding；DFlash 并发使用 k=2 控制 8GB 卡峰值显存，
三桶全部捕获，37 次回放、0 fallback。单 KV 块双请求触发 6 次真实抢占，完成后 0 状态槽残留。
上下文边界测试：500-token prompt、请求64输出，在512处硬停为12输出；末11步用单步 fallback，
没有越界且状态全回收。主机测试还覆盖 GDN 数学、跨 chunk 历史、共享 MoE/融合权重、
310P 短卷积和 KV 页大小。wheel 本地构建通过。

工件保存在 `bench_results/qwen35-check/`（Git 忽略）；`--compare-only` 可读取已有工件。
Qwen3.5 MoE 只有微型 CPU 结构/权重契约验证，未运行真实 MoE 权重。

参考：[Transformers Qwen3.5](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、
[Qwen3.5-2B 配置](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)。
