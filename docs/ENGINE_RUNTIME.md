# Engine runtime contracts

The engine resolves hardware and speculative-decoding choices once during
`Config` construction. Runtime code consumes capabilities and strategy
objects; backend and speculative method names are presentation metadata only.

## Construction order

The order is part of the memory and graph contract:

1. resolve backend, target adapter, speculative adapter, and derived flags;
2. initialize the process group and device;
3. load target and optional draft weights, enabling auxiliary-state capture
   before any warmup forward;
4. warm target and draft prefill paths;
5. calculate the KV budget, including draft paged KV where applicable;
6. allocate speculative metadata after KV sizing so it does not silently
   reduce the configured KV pool;
7. capture decode or whole-round speculative graphs, falling back to the
   isomorphic eager path if capture fails.

## Capability ownership

- `BackendBase` owns device, dtype, attention, KV layout, graph and config
  validation differences.
- `TargetModelAdapter` owns target matching, construction, weight loading and
  auxiliary-layer semantics.
- `SpecMethodAdapter` owns checkpoint normalization, draft construction and
  selection of a `SpecRuntime` family.
- `SpecRuntime` owns per-sequence speculative state and chooses eager versus
  graph execution. `BlockDraftRuntime` is shared by DFlash and DSpark; their
  sampling difference is implemented by the draft model.
- `Scheduler` receives already-derived cache policy and never resolves a
  backend or speculative method.

## Stable host-side observation

`LLMEngine.runtime_snapshot()` is the sole read interface for benchmark and
server metrics. It returns engine/configuration data, scheduler gauges, KV and
prefix-cache counters, prefill-path counters, graph counters, speculative
acceptance counters and draft-prefix snapshot counters.

`reset_measurement_stats()` starts a new measurement window without erasing
graph capture success/failure. `EngineObserver` reports scheduling before the
accelerator call and speculative commits after readback, allowing the HTTP
server to retain queue-wait, TTFT and per-request acceptance statistics without
monkey-patching engine methods.

## Graph and KV invariants

- Runtime batch size is rounded up to a captured bucket. Every padding row and
  cumulative-length tail must be reset whenever `n < bucket`, including first
  use and same-bucket reuse.
- Slot dtype is backend-defined; context lengths and block tables remain
  int32. Ascend's expanded per-query block table must not be passed to the
  CUDA/CoreX varlen FlashAttention path.
- Speculative capacity is reserved before a round and committed one token at a
  time so neither rejected drafts nor a `k+1` commit can cross
  `max_model_len`.
- Per-sequence draft state follows target KV lifetime and is dropped on both
  completion and preemption.

Qwen3.5 dense/MoE 使用注册模型的 recurrent state runtime，conv/SSM 状态按请求主槽管理。
DFlash verify 使用独立逐行状态槽，接受后把相应终态复制回主槽；分块 prefill 续接历史，
完成/抢占/取消均回收。prefix cache 在实现 GDN 快照前关闭。上下文剩余空间不足一整轮时
只执行单 token target fallback，避免固定宽度 verify 越界。详见 `QWEN35.md`。

Algorithm details and hardware-specific operator contracts remain in
`EAGLE3.md`, `DFlash.md`, and `ASCEND310P_GRAPH_SPEC.md`.

## Serving and benchmark contracts

The HTTP server has one engine-driving thread. Request threads only enqueue,
stream deltas, and wait for completion. The observer records the first
scheduling timestamp and speculative commit lengths without replacing engine
methods. Chat and completion endpoints share the same generation state and
response builders; stream and non-stream modes differ only in emission.

`/metrics` and `/v1/stats` are views of the same host counters. Reading either
endpoint must not synchronize a device. Resetting a benchmark window clears
replay, fallback, prefix, acceptance, and serving deltas while preserving graph
capture outcomes.

Both benchmark clients use `scripts.bench_utils`, which is deliberately
stdlib-only. Dataset filtering keeps records with a matching category or no
category; a non-negative seed shuffles deterministically, while a negative
seed preserves file order. Percentiles use linear interpolation and HTTP
metrics are compared as counter deltas.

## Speculative graph lifecycle

EAGLE3 and block-draft capture use one lifecycle: initialize valid padding
metadata, warm every bucket, capture with a shared pool, replay once to expose
deferred vendor errors, and discard the complete graph set on any failure.
Partial capture is never exposed to the hot path. Capture success/failure is a
startup fact; replay, eager fallback, sampling fallback, and bucket hits are
measurement-window counters.

Block-draft eager execution intentionally uses the same fixed-width buffers and
round body as graph execution. Masked rows make the static maximum-width draft
context numerically equivalent to the former per-batch allocation, while
avoiding a second append/draft/verify implementation.
