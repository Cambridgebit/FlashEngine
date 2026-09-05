import os
import pickle
import time
from itertools import accumulate

import torch
import torch.nn.functional as F
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.backends import get_backend
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.eagle3 import Eagle3DraftModel, load_eagle3_draft
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.backend = get_backend()
        # EAGLE3 投机解码（仅 TP=1，见 config 断言）。
        self.spec_enabled = bool(config.spec_model)
        # 每序列的 draft 种子隐层：fc(cat3 aux) 的输入，维度 [1, 3*hidden]。
        self.spec_hidden: dict[int, torch.Tensor] = {}
        # 每序列的 draft 流 carry：(row k 隐层, 首提案 token, 上轮接受数 m)。
        # 按 seq_id 存取（batch 顺序每轮可变，位置索引不安全）；冷启动轮由
        # eager 路径补齐。
        self.spec_carry: dict[int, tuple[torch.Tensor, int]] = {}
        self.spec_m: dict[int, int] = {}
        # 端到端 graph 采样：lm_head + sampler 捕获进 decode graph，replay 后
        # 直接读 token id（省每步图外 1 次 GEMV launch + 5 个 full-vocab kernel）。
        # TP>1 时 lm_head 的 gather 不可捕获，自动退回图外；排障可用
        # NANOVLLM_GRAPH_SAMPLE=0 关闭。
        self.graph_sample = (
            self.world_size == 1
            and os.environ.get("NANOVLLM_GRAPH_SAMPLE", "1") != "0"
        )
        # graph static buffer 的 padding 状态跟踪（见 run_model）。
        self._graph_bs_active = 0
        self._graph_cols_active = 0
        self.graphs = None
        self.graph_pool = None
        # 每步 / 每轮复用的 pinned 暂存：避免热路径反复 cudaHostAlloc。
        # decode 元数据并入单个 blob（dmeta/dmetaf），图内 scatter 分发，仅 2 次 H2D。
        self._max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        M = min(config.max_num_seqs, 512)
        W = self._max_num_blocks + 1
        self._pin_off = {'ids': 0, 'pos': M, 'slots': 2 * M, 'ctx': 3 * M,
                         'bt': 4 * M, 'topks': 4 * M + M * W}
        self._pin = dict(
            dmeta=torch.empty(M * (5 + W), dtype=torch.int64, pin_memory=True),
            dmetaf=torch.empty(2 * M, dtype=torch.float32, pin_memory=True),
            M=M, W=W,
        )

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        if self.spec_enabled:
            # aux 隐层捕获必须在 warmup 前打开（prefill 前向开始返回 (hidden, aux)）。
            self.model.model.aux_hidden_state_layers = tuple(config.spec_aux_layers)
            from transformers import AutoConfig
            self.draft_config = AutoConfig.from_pretrained(config.spec_model)
            self.draft_model = Eagle3DraftModel(self.draft_config, self.model.model.embed_tokens)
            load_eagle3_draft(self.draft_model, config.spec_model)
        if config.compile:
            # torch.compile：CoreX 上 qkv/rope/attention 区域的 Inductor 融合会产出
            # 错误结果，Qwen3Attention.forward 已 @torch.compiler.disable 保持 eager，
            # 此处只编译 embed + RMSNorm + MLP（正确性已验证 PASS）。
            self.model = torch.compile(self.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        self.spec_graphs = None
        if not self.enforce_eager:
            if not self.spec_enabled:
                self.capture_cudagraph()
            elif (self.world_size == 1 and self.backend.name == "cuda"
                  and os.environ.get("NANOVLLM_SPEC_GRAPH", "1") != "0"):
                # spec 解码路径 CUDA Graph（draft 链图 + verify 图，仅 TP=1 / CUDA）。
                self.capture_spec_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if self.graphs is not None:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        if self.spec_enabled:
            # spec warmup：target prefill（含 aux 捕获）+ draft prefill 一并预热；
            # 产生的 h_seed 属于 dummy 序列，预热后清理。
            self.run_prefill_spec(seqs)
            self.spec_hidden.clear()
        else:
            self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        if self.spec_enabled:
            # draft KV cache 与 target 共用同一 block_table/slot 布局，预算一并扣除。
            d = self.draft_config
            draft_kv_heads = d.num_key_value_heads // self.world_size
            draft_head_dim = getattr(d, "head_dim", d.hidden_size // d.num_attention_heads)
            block_bytes += 2 * d.num_hidden_layers * self.block_size * draft_kv_heads * draft_head_dim * hf_config.dtype.itemsize
        # WSL 下上一进程退出后驱动回收显存存在数秒延迟，`used` 可能虚高；
        # 预算不足时短暂等待重测，而非直接断言崩溃。
        for _ in range(6):
            free, total = torch.cuda.mem_get_info()
            used = total - free
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
            if config.num_kvcache_blocks > 0:
                break
            time.sleep(5)
        else:
            raise RuntimeError(
                f"kvcache 显存预算不足：需 {block_bytes / 2**20:.1f} MiB/block，"
                f"按 gpu_memory_utilization={config.gpu_memory_utilization} 计算可用块数为 0"
                "（可调低 max_model_len / block_size，或检查其他进程的显存占用）")
        shape = self.backend.get_kv_cache_shape(
            num_layers=hf_config.num_hidden_layers,
            num_blocks=config.num_kvcache_blocks,
            block_size=self.block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.kv_cache = torch.empty(shape)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        if self.spec_enabled:
            d = self.draft_config
            draft_kv_heads = d.num_key_value_heads // self.world_size
            draft_head_dim = getattr(d, "head_dim", d.hidden_size // d.num_attention_heads)
            draft_shape = self.backend.get_kv_cache_shape(
                num_layers=d.num_hidden_layers,
                num_blocks=config.num_kvcache_blocks,
                block_size=self.block_size,
                num_kv_heads=draft_kv_heads,
                head_dim=draft_head_dim,
            )
            self.draft_kv_cache = torch.empty(draft_shape)
            attn = self.draft_model.midlayer.self_attn.attn
            attn.k_cache = self.draft_kv_cache[0, 0]
            attn.v_cache = self.draft_kv_cache[1, 0]

    def drop_spec_state(self, seq_id: int):
        self.spec_hidden.pop(seq_id, None)
        self.spec_carry.pop(seq_id, None)
        self.spec_m.pop(seq_id, None)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 槽位 dtype 由后端决定（CUDA int32 / CoreX int64），cu_seqlens 两后端均为 int32。
        slot_mapping = torch.tensor(slot_mapping, dtype=self.backend.graph_slot_mapping_dtype(), pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        n = len(seqs)
        block_size = self.block_size
        pin, po = self._pin, self._pin_off
        M, W = pin['M'], pin['W']
        meta = pin['dmeta']
        for i, seq in enumerate(seqs):
            meta[po['ids'] + i] = seq.last_token
            meta[po['pos'] + i] = len(seq) - 1
            meta[po['ctx'] + i] = len(seq)
            meta[po['slots'] + i] = seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
            table = seq.block_table
            bt = meta[po['bt']:po['bt'] + M * W].view(M, W)
            bt[i, :len(table)] = torch.tensor(table, dtype=torch.int64)
            bt[i, len(table):] = -1
        # graph 直通：host 填 blob → run_model 2 次 H2D → 图内 scatter 分发。
        # gv buffer 是 capture 时（inference mode）创建的 inference tensor，
        # 相关写入须在 inference mode 内进行。
        if self.graphs is not None:
            gv = self.graph_vars
            bucket = next((x for x in self.graph_bs if x >= n), None)
            if bucket is not None:
                with torch.inference_mode():
                    if bucket < self._graph_bs_active:
                        meta[po['slots'] + n:po['slots'] + bucket].fill_(-1)
                        meta[po['ctx'] + n:po['ctx'] + bucket].zero_()
                        bt[n:, :].fill_(-1)
                    self._graph_bs_active, self._graph_cols_active = bucket, W
                    set_context(False,
                                slot_mapping=gv['slot_mapping'][:n],
                                context_lens=gv['context_lens'][:n],
                                block_tables=gv['block_tables'][:n],
                                max_context_len=self.config.max_model_len,
                                use_cuda_graph=True)
                return gv['input_ids'][:n], gv['positions'][:n]
        # 无图回退：直接 H2D 出 staging 张量（槽位 dtype 随后端；ctx/block_table 两后端均 int32）
        with torch.inference_mode():
            slot_dtype = self.backend.graph_slot_mapping_dtype()
            input_ids = meta[po['ids']:po['ids'] + n].cuda(non_blocking=True)
            positions = meta[po['pos']:po['pos'] + n].cuda(non_blocking=True)
            slot_mapping = meta[po['slots']:po['slots'] + n].cuda(non_blocking=True).to(slot_dtype)
            context_lens = meta[po['ctx']:po['ctx'] + n].cuda(non_blocking=True).to(torch.int32)
            block_tables = meta[po['bt']:po['bt'] + n * W].view(n, W).cuda(non_blocking=True).to(torch.int32)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, max_context_len=self.config.max_model_len)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], is_prefill: bool = False):
        """返回 (temperatures, top_ps, top_ks) 张量；top_k 不生效时传词表大小（掩码恒假）。"""
        vocab = self.config.hf_config.vocab_size
        n = len(seqs)
        pin, po = self._pin, self._pin_off
        M = pin['M']
        meta, metaf = pin['dmeta'], pin['dmetaf']
        for i, seq in enumerate(seqs):
            metaf[i] = seq.temperature
            metaf[M + i] = seq.top_p
            meta[po['topks'] + i] = seq.top_k if seq.top_k > 0 else vocab
        # graph 直通（decode）：run_model 的 scatter 分发，返回图内视图。
        if not is_prefill and self.graphs is not None:
            gv = self.graph_vars
            return gv['temperatures'][:n], gv['top_ps'][:n], gv['top_ks'][:n]
        with torch.inference_mode():
            return (metaf[:n].cuda(non_blocking=True),
                    metaf[M:M + n].cuda(non_blocking=True),
                    meta[po['topks']:po['topks'] + n].cuda(non_blocking=True))

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, bool]:
        """端到端 graph 路径返回 (token_ids, True)，其余返回 (logits, False)。"""
        # FULL_DECODE_ONLY：prefill 永远 eager；只有 decode 走 CUDA Graph。
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions)), False

        bs = input_ids.size(0)
        gv = self.graph_vars

        # 真实 bs 向上取到最近的 graph bucket（bs=3 → graph 4）。
        graph_bs = next((x for x in self.graph_bs if x >= bs), None)
        if graph_bs is None:
            return self.model.compute_logits(self.model(input_ids, positions)), False
        # 防御：仅接受 prepare_decode 直通准备的输入（blob 分发依赖固定地址）。
        if input_ids.data_ptr() != gv["input_ids"].data_ptr():
            return self.model.compute_logits(self.model(input_ids, positions)), False

        # 2 次 H2D + replay；元数据分发（int64→int32 转换）在图内 scatter 完成。
        gv["dmeta"].copy_(self._pin["dmeta"], non_blocking=True)
        gv["dmetaf"].copy_(self._pin["dmetaf"], non_blocking=True)
        self.graphs[graph_bs].replay()
        if self.graph_sample:
            return gv["token_ids"][:bs], True
        return self.model.compute_logits(gv["outputs"][:bs]), False

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        sample = self.prepare_sample(seqs, is_prefill) if self.rank == 0 else None
        result, sampled = self.run_model(input_ids, positions, is_prefill, sample)
        if self.rank == 0:
            token_ids = (result.tolist() if sampled
                         else self.sampler(result, *sample).tolist())
        else:
            token_ids = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_prefill_spec(self, seqs: list[Sequence]) -> list[int]:
        """spec 模式 prefill 步：target 前向（返回 aux 隐层）+ draft 同 chunk 前向。

        draft prefill 复用同一 context（token 布局/槽位完全一致），为 prompt 的
        每个位置写入 draft KV。已完成 prefill 的序列记录 h_seed（aux 最后一行，
        即预测首个采样 token 的隐层）。
        """
        assert self.rank == 0
        input_ids, positions = self.prepare_prefill(seqs)
        hidden, aux = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden)
        sample = self.prepare_sample(seqs)
        token_ids = self.sampler(logits, *sample).tolist()
        offsets = list(accumulate([0] + [seq.num_scheduled_tokens for seq in seqs]))
        # draft prefill：fc(3H -> H) 后逐位置过 midlayer；输出丢弃。
        # 口径与 decode 虚构窗口一致（vLLM set_inputs_first_pass）：rope p 的行
        # embed tok@p+1（末槽补本步首个采样 token），aux@p 前瞻一位。
        seed = F.linear(aux, self.draft_model.fc.weight)
        shifted = []
        for i, seq in enumerate(seqs):
            off0, off1 = offsets[i], offsets[i + 1]
            shifted.append(torch.cat([input_ids[off0 + 1:off1],
                                      input_ids.new_tensor(token_ids[i]).view(1)]))
        self.draft_model.forward_prefill(positions, seed, torch.cat(shifted))
        reset_context()
        for i, seq in enumerate(seqs):
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 该序列本步完成 prefill：h_seed 取其 chunk 最后一行 aux（预测首 token）。
                self.spec_hidden[seq.seq_id] = aux[offsets[i + 1] - 1:offsets[i + 1]].clone()
                seq.spec_cold = True
        return token_ids

    @torch.inference_mode()
    def spec_round(self, seqs: list[Sequence]) -> list[list[int]]:
        """一轮投机解码（入口）：有捕获图、桶足够且全部序列已有 carry 时走 CUDA
        Graph，否则 eager（冷启动轮在 eager 内补齐 carry）。"""
        if (self.spec_graphs is not None and seqs
                and not any(seq.spec_cold for seq in seqs)):
            bucket = next((x for x in self.spec_graph_bs if x >= len(seqs)), None)
            if bucket is not None:
                return self._spec_round_graph(seqs, bucket)
        return self._spec_round_eager(seqs)

    @torch.inference_mode()
    def _spec_round_eager(self, seqs: list[Sequence]) -> list[list[int]]:
        """一轮投机解码（与图轮同构的 eager 版本）。

        流程：采样行（rope l-2：fc(aux@l-2)+emb(last_token)，冷启动用 prompt
        末行 aux，热启动与 carry 同源可复算）→ 链步 1..k-1（流式续写）→ target
        verify → 链式接受 → 虚构窗口重放（verify 行位置/槽位/look-ahead aux，
        embed 右移一位、行 m 为 bonus）。
        返回每个序列新提交的 token 列表（接受链 + bonus token，含截断）。
        """
        assert self.rank == 0
        k = self.config.spec_num_tokens
        n = len(seqs)
        if n == 0:
            return []
        block_size = self.block_size
        lens = [len(seq) for seq in seqs]
        rows = k + 1

        vocab = self.config.hf_config.vocab_size
        draft_vocab = self.draft_config.draft_vocab_size
        temperatures = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ps = torch.tensor([seq.top_p for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        # draft 采样在 draft 词表上，top_k 上限相应取 draft 词表大小。
        top_ks_draft = torch.tensor(
            [seq.top_k if 0 < seq.top_k < draft_vocab else draft_vocab for seq in seqs],
            dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        def slot_of(seq, p):
            return seq.block_table[p // block_size] * block_size + p % block_size

        # ---- 采样行：draft1（冷启动即首提案；热启动与 carry 同源） ----
        h_seed = torch.cat([self.spec_hidden[seq.seq_id] for seq in seqs], dim=0)
        seed = F.linear(h_seed, self.draft_model.fc.weight)
        slot_dtype = self.backend.graph_slot_mapping_dtype()
        p = torch.tensor([l - 2 for l in lens], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ctx = torch.tensor([l - 1 for l in lens], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slots = torch.tensor([[slot_of(seq, l - 2)] for seq, l in zip(seqs, lens)],
                             dtype=slot_dtype, pin_memory=True).cuda(non_blocking=True)
        tokens = torch.tensor([seq.last_token for seq in seqs], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        set_context(False, slot_mapping=slots, context_lens=ctx,
                    block_tables=self.prepare_block_tables(seqs),
                    max_context_len=self.config.max_model_len)
        row_m_hidden, draft_logits = self.draft_model.forward_decode(p, seed, tokens)
        row_m_hidden = row_m_hidden.clone()
        draft_ids = self.sampler.forward_native(draft_logits, temperatures, top_ps, top_ks_draft)
        draft1 = self.draft_model.map_to_target_ids(draft_ids)
        drafted_tokens = [draft1.tolist()]

        # ---- 链步 1..k-1：流式续写（rope l-2+i，读窗 l+i-1） ----
        for i in range(1, k):
            p = torch.tensor([l - 2 + i for l in lens], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            ctx = torch.tensor([l + i - 1 for l in lens], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            slots = torch.tensor([[slot_of(seq, l - 2 + i)] for seq, l in zip(seqs, lens)],
                                 dtype=slot_dtype, pin_memory=True).cuda(non_blocking=True)
            set_context(False, slot_mapping=slots, context_lens=ctx,
                        block_tables=self.prepare_block_tables(seqs),
                        max_context_len=self.config.max_model_len)
            seed, draft_logits = self.draft_model.forward_decode(p, seed, draft1 if i == 1 else tokens)
            draft_ids = self.sampler.forward_native(draft_logits, temperatures, top_ps, top_ks_draft)
            tokens = self.draft_model.map_to_target_ids(draft_ids)
            drafted_tokens.append(tokens.tolist())
        drafted = [list(col) for col in zip(*drafted_tokens)]  # [bs][k]
        reset_context()

        # ---- verify：target 一次性前向 [last_token, d1..dk]，行 j 预测位置 L+j ----
        # 注意：临时追加不经过 append_token，num_tokens 保持为已提交长度 L；
        # 涉及总长度的计算一律用 len(seq.token_ids)。
        for seq, toks in zip(seqs, drafted):
            seq.token_ids.extend(toks)
            seq.num_cached_tokens = len(seq.token_ids) - 1 - k    # = L-1（verify 起点）
            seq.num_scheduled_tokens = k + 1
        input_ids, positions = self.prepare_prefill(seqs)
        hidden, aux = self.model(input_ids, positions)
        logits = F.linear(hidden, self.model.lm_head.weight)
        reset_context()

        rows = k + 1
        temperatures_r = temperatures.repeat_interleave(rows)
        top_ps_r = top_ps.repeat_interleave(rows)
        top_ks_target = torch.tensor(
            [seq.top_k if seq.top_k > 0 else vocab for seq in seqs for _ in range(rows)],
            dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        sampled = self.sampler.forward_native(logits, temperatures_r, top_ps_r, top_ks_target).tolist()

        results = []
        for i, seq in enumerate(seqs):
            base = i * rows
            t = sampled[base:base + rows]
            d = drafted[i]
            # 链式接受：draft 提案与 target 采样逐位相等的最长前缀。
            m = 0
            while m < k and t[m] == d[m]:
                m += 1
            committed = d[:m]
            bonus = t[m]
            # 行 m 的 aux 隐层 = 位置 L-1+m 的 target 隐层（预测 bonus 的状态），
            # 无论接受多少个都必然在 verify 覆盖范围内；eager 复算采样行与图轮
            # fallback 均依赖它。
            self.spec_hidden[seq.seq_id] = aux[base + m:base + m + 1].clone()
            # carry 交接：采样行隐层 + 首提案 + 上轮接受数（图轮据此续流）。
            self.spec_carry[seq.seq_id] = (row_m_hidden[i:i + 1], drafted[i][0])
            self.spec_m[seq.seq_id] = m
            # 回滚未提交的 drafted token（KV 残留无害：槽位会被后续写入覆盖）。
            del seq.token_ids[len(seq.token_ids) - k:]
            seq.num_scheduled_tokens = 0
            committed.append(bonus)
            results.append(committed)

        # 虚构窗口重放（与图轮 refill 同构）：行 j（= verify 行 j，位置 L-1+j）
        # 的 x = fc(v_aux[j])，embed = verify token[j+1]、行 m 覆盖为 bonus；
        # KV 写入 verify 槽位。输出丢弃（下一轮 eager 从 spec_hidden 复算采样行，
        # 图轮走 carry）。
        seed_r = F.linear(aux, self.draft_model.fc.weight)
        t_all = torch.tensor(
            [tok for i, seq in enumerate(seqs) for tok in sampled[i * rows:(i + 1) * rows]],
            dtype=torch.int64, pin_memory=True)
        m_idx = torch.tensor([len(results[i]) - 1 for i in range(n)], dtype=torch.int64)
        bonus = t_all.view(n, rows).gather(1, m_idx[:, None])
        ar = torch.arange(rows)
        t_ext = torch.cat([t_all.view(n, rows)[:, 1:], t_all.view(n, rows)[:, k:k + 1]], dim=1)
        ids_r = torch.where(ar[None, :] == m_idx[:, None], bonus, t_ext).reshape(-1).tolist()
        pos_r, slots_r = [], []
        for i, (seq, l) in enumerate(zip(seqs, lens)):
            for j in range(rows):
                p = l - 1 + j
                pos_r.append(p)
                slots_r.append(slot_of(seq, p))
        cu_q = [0]
        for _ in seqs:
            cu_q.append(cu_q[-1] + rows)
        cu_k = [0]
        for l in lens:
            cu_k.append(cu_k[-1] + l + k)
        set_context(True,
                    cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
                    cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
                    max_seqlen_q=rows,
                    max_seqlen_k=self.config.max_model_len,
                    slot_mapping=torch.tensor(slots_r, dtype=slot_dtype, pin_memory=True).cuda(non_blocking=True),
                    block_tables=self.prepare_block_tables(seqs))
        self.draft_model.forward_prefill(
            torch.tensor(pos_r, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            seed_r,
            torch.tensor(ids_r, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True))
        reset_context()
        for seq in seqs:
            seq.spec_cold = False
        return results

    # ---- spec 解码 CUDA Graph ----
    # 每桶一张图覆盖整轮：draft 链式起草 → target verify → 图内接受/提交/种子
    # → refill（用 verify 的 target aux 回填 draft KV）。每轮仅一次 D2H。

    def _alloc_spec_graph_vars(self, max_bs: int, rows: int, k: int) -> dict:
        hf_config = self.config.hf_config
        vocab = hf_config.vocab_size
        draft_vocab = self.draft_config.draft_vocab_size
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size + 1
        # 槽位 dtype 由后端决定（CUDA int32 / CoreX int64）；ctx/cu_seqlens/
        # block_table 两后端均为 int32；meta blob 统一存 int64，图内 copy_ 自动转型。
        slot_dtype = self.backend.graph_slot_mapping_dtype()
        # 元数据 blob 分段（host 填 pinned → 每类一次 H2D → 图内 scatter 分发）。
        # 段均按 max_bs 步进，任意 bs 只用前缀。
        M = max_bs
        W = max_num_blocks
        o = self._meta_off = {}
        pos = 0
        for name, size in (('chain_tokens', M), ('chain_pos', k * M), ('chain_ctxlen', k * M),
                           ('chain_slots', k * M), ('v_pos', rows * M), ('v_slots', rows * M),
                           ('v_cu_k', M + 1), ('ctopks', M),
                           ('v_topks', rows * M), ('btables', M * W)):
            o[name] = pos
            pos += size
        fo = self._metaf_off = {'ctemps': 0, 'ctopps': M,
                                'v_temps': 2 * M, 'v_topps': 2 * M + rows * M}
        self._meta_i64_end = pos
        self._meta_f32_end = fo['v_topps'] + rows * M
        return dict(
            meta=torch.zeros(pos, dtype=torch.int64),
            metaf=torch.zeros(fo['v_topps'] + rows * M, dtype=torch.float32),
            # draft 链静态 buffer；carry_seed 为上一轮 refill 采样行（row k）的
            # midlayer 输出（TTT 链种子），drafted_ids[0] 为其采样的首个提案。
            carry_seed=torch.zeros(max_bs, hf_config.hidden_size, dtype=hf_config.dtype),
            chain_tokens=torch.zeros(max_bs, dtype=torch.int64),
            drafted_ids=torch.zeros(k, max_bs, dtype=torch.int64),
            chain_pos=torch.zeros(k, max_bs, dtype=torch.int64),
            chain_slots=torch.full((k, max_bs), -1, dtype=slot_dtype),
            chain_ctxlen=torch.zeros(k, max_bs, dtype=torch.int32),
            chain_btables=torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32),
            chain_temps=torch.ones(max_bs, dtype=torch.float32),
            chain_topps=torch.ones(max_bs, dtype=torch.float32),
            chain_topks=torch.full((max_bs,), draft_vocab, dtype=torch.int64),
            # verify 静态 buffer
            v_input_ids=torch.zeros(max_bs * rows, dtype=torch.int64),
            v_positions=torch.zeros(max_bs * rows, dtype=torch.int64),
            # cu_seqlens_q 恒定：每序列恰好 k+1 行，capture 后无需更新
            v_cu_q=(torch.arange(max_bs + 1, dtype=torch.int32) * rows),
            v_cu_k=torch.zeros(max_bs + 1, dtype=torch.int32),
            v_slots=torch.full((max_bs * rows,), -1, dtype=slot_dtype),
            v_btables=torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32),
            v_temps=torch.ones(max_bs * rows, dtype=torch.float32),
            v_topps=torch.ones(max_bs * rows, dtype=torch.float32),
            v_topks=torch.full((max_bs * rows,), vocab, dtype=torch.int64),
            v_aux=torch.zeros(max_bs * rows, hf_config.hidden_size * 3, dtype=hf_config.dtype),
            v_token_ids=torch.zeros(max_bs * rows, dtype=torch.int64),
            # 接受/提交输出（cnt 打包在最后一列，每轮仅一次 D2H）
            commit_buf=torch.full((max_bs, rows + 1), -1, dtype=torch.int64),
        )

    @torch.inference_mode()
    def capture_spec_cudagraph(self):
        k = self.config.spec_num_tokens
        rows = k + 1
        max_bs = min(self.config.max_num_seqs, 512)
        self.spec_graph_bs = [x for x in ([1, 2, 4, 8] + list(range(16, max_bs + 1, 16))) if x <= max_bs]
        g = self._sg = self._alloc_spec_graph_vars(max_bs, rows, k)
        # spec 热路径 pinned 元数据 blob（每轮 2 次 H2D，零 cudaHostAlloc）。
        # 注意此时 default device 已是 cuda，须显式 device='cpu' 才能分配 pinned。
        self._spin = dict(
            M=max_bs,
            W=self._max_num_blocks + 1,
            meta=torch.empty(self._meta_i64_end, dtype=torch.int64, device='cpu', pin_memory=True),
            metaf=torch.empty(self._meta_f32_end, dtype=torch.float32, device='cpu', pin_memory=True),
        )
        self.spec_graphs = {}
        self.spec_graph_pool = None
        self._spec_graph_active = 0

        def draft_chain(bs):
            # 链从上一轮 refill 采样行（carry_seed）续写：TTT 流式种子 + 上轮
            # 首提案；写在流位置（host 按 l-m+k+i-2 填充），读窗仍是提交前缀。
            set_context(False,
                        slot_mapping=g['chain_slots'][1][:bs],
                        context_lens=g['chain_ctxlen'][1][:bs],
                        block_tables=g['chain_btables'][:bs],
                        max_context_len=self.config.max_model_len,
                        use_cuda_graph=True)
            seed = g['carry_seed'][:bs]
            for i in range(1, k):
                if i > 1:
                    set_context(False,
                                slot_mapping=g['chain_slots'][i][:bs],
                                context_lens=g['chain_ctxlen'][i][:bs],
                                block_tables=g['chain_btables'][:bs],
                                max_context_len=self.config.max_model_len,
                                use_cuda_graph=True)
                seed, logits = self.draft_model.forward_decode(
                    g['chain_pos'][i][:bs], seed, g['drafted_ids'][i - 1][:bs])
                ids = self.sampler.forward_native(
                    logits, g['chain_temps'][:bs], g['chain_topps'][:bs], g['chain_topks'][:bs])
                g['drafted_ids'][i][:bs] = self.draft_model.map_to_target_ids(ids)

        def verify(bs):
            # max_seqlen_k 用 max_model_len 上界（varlen kernel 只用于 grid sizing，
            # 正确性由 cu_seqlens 决定），避免 capture 烘焙随轮次变化的 python int。
            set_context(True,
                        cu_seqlens_q=g['v_cu_q'][:bs + 1],
                        cu_seqlens_k=g['v_cu_k'][:bs + 1],
                        max_seqlen_q=rows,
                        max_seqlen_k=self.config.max_model_len,
                        slot_mapping=g['v_slots'][:bs * rows],
                        block_tables=g['v_btables'][:bs])
            hidden, aux = self.model(g['v_input_ids'][:bs * rows], g['v_positions'][:bs * rows])
            logits = F.linear(hidden, self.model.lm_head.weight)
            g['v_token_ids'][:bs * rows] = self.sampler.forward_native(
                logits, g['v_temps'][:bs * rows], g['v_topps'][:bs * rows], g['v_topks'][:bs * rows])
            g['v_aux'][:bs * rows].copy_(aux)

        def acceptance(bs):
            """链式接受 + 提交组装 + 下一轮 draft 种子（全部图内静态算子）。"""
            t = g['v_token_ids'][:bs * rows].view(bs, rows)
            d = g['drafted_ids'][:k, :bs].t()
            match = t[:, :k] == d
            m = torch.where(match.all(-1),
                            torch.full((bs,), k, dtype=torch.int64, device=t.device),
                            match.int().argmin(-1))
            t_m = t.gather(1, m[:, None])
            ar = torch.arange(rows, device=t.device)
            d_pad = torch.cat([d, d.new_zeros(bs, 1)], dim=1)
            out = torch.where(ar < m[:, None], d_pad.gather(1, ar.clamp(max=k).expand(bs, rows)), t_m)
            # [0..m] 为提交 token，[rows] 列存 cnt=m+1；padding 行保持 -1
            g['commit_buf'][:bs, :rows] = torch.where(ar <= m[:, None], out, torch.full_like(out, -1))
            g['commit_buf'][:bs, rows] = m + 1

        def refill(bs):
            """虚构窗口重放（对齐 vLLM padded 分支）+ 下一轮 draft1 采样。

            行 j（= verify 行 j，位置 L-1+j）的 x = fc(v_aux[j])（look-ahead aux，
            目标消费过本行提案 token 后的隐层），embed 取 verify token[j+1]
            （vLLM 的 shift 口径：rope p 的行 embed tok@p+1，输出预测 tok@p+2），
            行 m 被覆盖为 bonus 并作为采样行（eagle_prepare_inputs_padded_kernel:
            index_to_sample = 段尾 - 被拒数）。KV 写入 verify 槽位，构成下一轮
            链步连续读取的 draft 流窗口；采样行隐层/提案经 host carry 交接。
            """
            seed_r = F.linear(g['v_aux'][:bs * rows], self.draft_model.fc.weight)
            t = g['v_token_ids'][:bs * rows].view(bs, rows)
            m_idx = g['commit_buf'][:bs, rows] - 1
            bonus = t.gather(1, m_idx[:, None])
            ar = torch.arange(rows, device=t.device)
            t_ext = torch.cat([t[:, 1:], t[:, k:k + 1]], dim=1)
            ids_r = torch.where(ar[None, :] == m_idx[:, None], bonus, t_ext).reshape(-1)
            set_context(True,
                        cu_seqlens_q=g['v_cu_q'][:bs + 1],
                        cu_seqlens_k=g['v_cu_k'][:bs + 1],
                        max_seqlen_q=rows,
                        max_seqlen_k=self.config.max_model_len,
                        slot_mapping=g['v_slots'][:bs * rows],
                        block_tables=g['v_btables'][:bs])
            h = self.draft_model.forward_prefill(
                g['v_positions'][:bs * rows], seed_r, ids_r)
            h_m = h.view(bs, rows, -1).gather(
                1, m_idx[:, None, None].expand(bs, 1, h.size(-1))).squeeze(1)
            logits = F.linear(self.draft_model.norm(h_m), self.draft_model.lm_head.weight)
            ids = self.sampler.forward_native(
                logits, g['chain_temps'][:bs], g['chain_topps'][:bs], g['chain_topks'][:bs])
            g['drafted_ids'][0][:bs] = self.draft_model.map_to_target_ids(ids)
            g['carry_seed'][:bs] = h_m

        def scatter(bs):
            """图内把 meta blob 分发到各静态 buffer（int64→图 buffer dtype 转换在 copy 内）。"""
            o, fo = self._meta_off, self._metaf_off
            M, W = self._spin['M'], self._spin['W']
            m, mf = g['meta'], g['metaf']
            g['chain_tokens'][:bs].copy_(m[o['chain_tokens']:o['chain_tokens'] + bs])
            cp = m[o['chain_pos']:o['chain_pos'] + k * M].view(k, M)
            cc = m[o['chain_ctxlen']:o['chain_ctxlen'] + k * M].view(k, M)
            cs = m[o['chain_slots']:o['chain_slots'] + k * M].view(k, M)
            for i in range(k):
                g['chain_pos'][i][:bs].copy_(cp[i, :bs])
                g['chain_ctxlen'][i][:bs].copy_(cc[i, :bs])
                g['chain_slots'][i][:bs].copy_(cs[i, :bs])
            g['v_positions'][:bs * rows].copy_(m[o['v_pos']:o['v_pos'] + bs * rows])
            g['v_slots'][:bs * rows].copy_(m[o['v_slots']:o['v_slots'] + bs * rows])
            g['v_cu_k'][:bs + 1].copy_(m[o['v_cu_k']:o['v_cu_k'] + bs + 1])
            g['chain_topks'][:bs].copy_(m[o['ctopks']:o['ctopks'] + bs])
            g['v_topks'][:bs * rows].copy_(m[o['v_topks']:o['v_topks'] + bs * rows])
            bt = m[o['btables']:o['btables'] + M * W].view(M, W)
            g['chain_btables'][:bs].copy_(bt[:bs])
            g['v_btables'][:bs].copy_(bt[:bs])
            g['chain_temps'][:bs].copy_(mf[fo['ctemps']:fo['ctemps'] + bs])
            g['chain_topps'][:bs].copy_(mf[fo['ctopps']:fo['ctopps'] + bs])
            g['v_temps'][:bs * rows].copy_(mf[fo['v_temps']:fo['v_temps'] + bs * rows])
            g['v_topps'][:bs * rows].copy_(mf[fo['v_topps']:fo['v_topps'] + bs * rows])

        def round_graph(bs):
            scatter(bs)
            draft_chain(bs)
            v = g['v_input_ids'][:bs * rows].view(bs, rows)
            v[:, 0] = g['chain_tokens'][:bs]
            v[:, 1:].copy_(g['drafted_ids'][:k, :bs].t())
            verify(bs)
            acceptance(bs)
            refill(bs)

        try:
            for bs in reversed(self.spec_graph_bs):
                # warmup：triton/flash kernel lazy init 必须在 capture 前完成；
                # warmup 数据用零/-1（slot=-1 跳过 KV 写，ctxlen=0 合法）。
                g['v_cu_k'][:bs + 1] = g['v_cu_q'][:bs + 1]
                round_graph(bs)
                reset_context()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.spec_graph_pool):
                    round_graph(bs)
                if self.spec_graph_pool is None:
                    self.spec_graph_pool = graph.pool()
                self.spec_graphs[bs] = graph
                reset_context()
                torch.cuda.synchronize()
        except Exception as e:
            # 后端不支持图捕获（如部分 vendor kernel 含 host 同步/动态 workspace）
            # 时优雅降级：spec_round 自动回落到同构 eager 轮，功能不受影响。
            print(f'[nano-vllm] spec CUDA Graph 捕获失败，降级为 eager spec 轮：'
                  f'{type(e).__name__}: {e}', flush=True)
            reset_context()
            torch.cuda.synchronize()
            self.spec_graphs = None
            self._sg = None
            self.spec_graph_pool = None
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def _spec_round_graph(self, seqs: list[Sequence], bucket: int) -> list[list[int]]:
        k = self.config.spec_num_tokens
        rows = k + 1
        n = len(seqs)
        block_size = self.block_size
        lens = [len(seq) for seq in seqs]
        g = self._sg
        M, W = self._spin['M'], self._spin['W']
        o, fo = self._meta_off, self._metaf_off
        meta, metaf = self._spin['meta'], self._spin['metaf']
        draft_vocab = self.draft_config.draft_vocab_size
        vocab = self.config.hf_config.vocab_size

        # ---- 填充 pinned 元数据 blob（图内 scatter 分发，仅 2 次 H2D）----
        for i, seq in enumerate(seqs):
            meta[o['chain_tokens'] + i] = seq.last_token
            meta[o['ctopks'] + i] = seq.top_k if 0 < seq.top_k < draft_vocab else draft_vocab
            metaf[fo['ctemps'] + i] = seq.temperature
            metaf[fo['ctopps'] + i] = seq.top_p
        cp = meta[o['chain_pos']:o['chain_pos'] + k * M].view(k, M)
        cc = meta[o['chain_ctxlen']:o['chain_ctxlen'] + k * M].view(k, M)
        cs = meta[o['chain_slots']:o['chain_slots'] + k * M].view(k, M)
        # 链步 1..k-1：采样行 rope 恒为 l-2（与 m 无关），流式续写 p = l-2+i，
        # 读窗 = 提交前缀 ctxlen = l+i-1（含本行流槽写）；行 0 已并入 refill
        # 采样行。
        for i in range(1, k):
            for j, (seq, l) in enumerate(zip(seqs, lens)):
                p = l + i - 2
                cp[i, j] = p
                cc[i, j] = l + i - 1
                cs[i, j] = seq.block_table[p // block_size] * block_size + p % block_size
        vp = meta[o['v_pos']:o['v_pos'] + rows * M].view(M, rows)
        vs = meta[o['v_slots']:o['v_slots'] + rows * M].view(M, rows)
        vt = metaf[fo['v_temps']:fo['v_temps'] + rows * M].view(M, rows)
        vpp = metaf[fo['v_topps']:fo['v_topps'] + rows * M].view(M, rows)
        cu_k = meta[o['v_cu_k']:o['v_cu_k'] + M + 1]
        cu_k[0] = 0
        acc = 0
        for i, (seq, l) in enumerate(zip(seqs, lens)):
            for j in range(rows):
                p = l - 1 + j
                vp[i, j] = p
                vs[i, j] = seq.block_table[p // block_size] * block_size + p % block_size
                vt[i, j] = seq.temperature
                vpp[i, j] = seq.top_p
                meta[o['v_topks'] + i * rows + j] = seq.top_k if seq.top_k > 0 else vocab
            acc += l + k
            cu_k[i + 1] = acc
        bt = meta[o['btables']:o['btables'] + M * W].view(M, W)
        for i, seq in enumerate(seqs):
            table = seq.block_table
            bt[i, :len(table)] = torch.tensor(table, dtype=torch.int64)
            bt[i, len(table):] = -1

        # padding 复位（桶缩小时 blob 段补 -1/0，scatter 会覆盖图 buffer）
        if bucket < self._spec_graph_active:
            cp[:, n:].fill_(-1)
            cc[:, n:].zero_()
            cs[:, n:].fill_(-1)
            vs[:, n:].fill_(-1)
            cu_k[n + 1:].copy_(torch.arange(n + 1, M + 1, dtype=torch.int64) * rows)
            bt[n:, :].fill_(-1)
        self._spec_graph_active = bucket

        # ---- H2D + 单图回放：scatter → draft 链 → verify → 图内接受 → refill ----
        g['carry_seed'][:n] = torch.cat([self.spec_carry[seq.seq_id][0] for seq in seqs], dim=0)
        g['drafted_ids'][0][:n] = torch.tensor(
            [self.spec_carry[seq.seq_id][1] for seq in seqs], dtype=torch.int64)
        g['meta'].copy_(meta, non_blocking=True)
        g['metaf'].copy_(metaf, non_blocking=True)
        self.spec_graphs[bucket].replay()

        # ---- 每轮唯一一次 D2H（cnt 打包在 commit_buf 最后一列）----
        rows_out = g['commit_buf'][:n].tolist()
        results = []
        for i, seq in enumerate(seqs):
            row = rows_out[i]
            results.append(row[:row[rows]])
            # 下一轮 carry：refill 行 k 隐层 + 其采样提案 + 上轮接受数。
            self.spec_carry[seq.seq_id] = (g['carry_seed'][i:i + 1].clone(),
                                           g['drafted_ids'][0][i].item())
            self.spec_m[seq.seq_id] = row[rows] - 1
            # aux 行 m（供 eager fallback 复算采样行）。
            base = i * rows + row[rows] - 1
            self.spec_hidden[seq.seq_id] = g['v_aux'][base:base + 1].clone()
            # 回滚未提交的 drafted token（KV 残留已被 refill / 后续写入覆盖）。
            seq.num_scheduled_tokens = 0
        return results

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        # + 1：runtime block_tables 宽度可能比 capture 时更宽
        # （nano-vLLM 公开 bug），static buffer 直接留出余量。
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size + 1
        # 同 dtype 的输入合并为一块连续 buffer，prepare 阶段可单次 H2D 直写。
        io = torch.zeros(2 * max_bs, dtype=torch.int64)
        sc = torch.zeros(2 * max_bs, dtype=self.backend.graph_slot_mapping_dtype())
        sc[:max_bs].fill_(-1)
        input_ids, positions = io[:max_bs], io[max_bs:]
        slot_mapping, context_lens = sc[:max_bs], sc[max_bs:]
        block_tables = torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size, dtype=hf_config.dtype)
        temperatures = torch.ones(max_bs, dtype=torch.float32)
        top_ps = torch.ones(max_bs, dtype=torch.float32)
        top_ks = torch.full((max_bs,), hf_config.vocab_size, dtype=torch.int64)
        token_ids = torch.zeros(max_bs, dtype=torch.int64)
        self.graph_bs = [x for x in ([1, 2, 4, 8] + list(range(16, max_bs + 1, 16))) if x <= max_bs]
        self.graphs = {}
        self.graph_pool = None
        # decode 元数据 blob 的设备暂存：host 只做 2 次 H2D，分发在图内 scatter。
        po = self._pin_off
        dmeta = torch.zeros(self._pin['dmeta'].numel(), dtype=torch.int64)
        dmetaf = torch.zeros(self._pin['dmetaf'].numel(), dtype=torch.float32)

        def scatter(bs):
            """图内把 dmeta blob 分发到各静态 buffer（int64→int32 转换在 copy 内）。"""
            input_ids[:bs].copy_(dmeta[po['ids']:po['ids'] + bs])
            positions[:bs].copy_(dmeta[po['pos']:po['pos'] + bs])
            slot_mapping[:bs].copy_(dmeta[po['slots']:po['slots'] + bs])
            context_lens[:bs].copy_(dmeta[po['ctx']:po['ctx'] + bs])
            bt = dmeta[po['bt']:po['bt'] + max_bs * max_num_blocks].view(max_bs, max_num_blocks)
            block_tables[:bs].copy_(bt[:bs])
            top_ks[:bs].copy_(dmeta[po['topks']:po['topks'] + bs])
            temperatures[:bs].copy_(dmetaf[:bs])
            top_ps[:bs].copy_(dmetaf[max_bs:2 * max_bs][:bs])

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # Graph Context 必须静态：max_context_len 用 config.max_model_len，
            # 不要用 context_lens.max().item()（capture 后不可变 python 整数）。
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                max_context_len=config.max_model_len,
                use_cuda_graph=True,
            )
            # Warmup：IXFormer kernel lazy init / workspace / torch.compile
            # 必须在正式 capture 前完成。graph_sample 路径把 lm_head + sampler
            # 一并捕获（含 exponential_ 随机采样——PyTorch CUDA RNG 是
            # graph-safe 的，replay 每次产生新随机数）。
            scatter(bs)
            hidden = self.model(input_ids[:bs], positions[:bs])    # warmup
            if self.graph_sample:
                token_ids[:bs] = self.sampler(
                    self.model.compute_logits(hidden), temperatures[:bs], top_ps[:bs], top_ks[:bs])
            else:
                outputs[:bs] = hidden
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                scatter(bs)
                hidden = self.model(input_ids[:bs], positions[:bs])    # capture
                if self.graph_sample:
                    token_ids[:bs] = self.sampler(
                        self.model.compute_logits(hidden), temperatures[:bs], top_ps[:bs], top_ks[:bs])
                else:
                    outputs[:bs] = hidden
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            M=max_bs,
            io=io,
            sc=sc,
            dmeta=dmeta,
            dmetaf=dmetaf,
            block_tables=block_tables,
            outputs=outputs,
            top_ks=top_ks,
            token_ids=token_ids,
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            temperatures=temperatures,
            top_ps=top_ps,
        )
