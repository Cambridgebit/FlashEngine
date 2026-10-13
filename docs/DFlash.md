# DFlash / DSpark block speculative decoding

The current engine supports DFlash and DSpark draft checkpoints for the dense
Qwen3 target model. Qwen3.5 and DFlash2 are intentionally not part of the
current support surface.

## Architecture boundary

Target models and speculative methods are registered independently:

- `flashengine/models/registry.py` owns target-model construction, weight loading,
  auxiliary-layer semantics, and the list of compatible speculative methods.
- `flashengine/speculative/registry.py` detects a draft checkpoint, normalizes its
  configuration, derives the proposal length, and constructs the draft model.
- `ModelRunner` operates on the resolved adapters and the common speculative
  commit contract. It does not select concrete model classes by name.

Adding a future target model therefore requires a target adapter plus its model
implementation. Adding a future speculative method requires a spec adapter plus
its draft implementation. Compatibility is declared by the target adapter.

## Qwen3 auxiliary-state contract

DFlash and DSpark configurations describe `target_layer_ids` as zero-based
target-layer outputs. Qwen3 captures layer inputs, so the Qwen3 adapter maps
each layer `l` to capture point `l + 1`. This translation belongs to the target
adapter and must not be duplicated in draft models or the scheduler.

## One speculative round

A block-draft round has four phases:

1. Append target auxiliary states committed by the preceding verification.
2. Let the draft model propose a token block.
3. Verify the anchor and proposal tokens with one target-model forward pass.
4. Commit the longest accepted prefix and one verifier bonus token.

The scheduler sees only committed tokens. Rejected proposal KV entries may stay
in reserved slots because all later attention calls are bounded by the committed
context length and overwrite those slots before reading them.

## Chunked prefill

Chunked prefill is supported for both DFlash and DSpark. Every prompt chunk is
immediately projected into draft context K/V; projection is not delayed until
the final chunk. The final chunk additionally initializes the first speculative
round. This makes the draft context equivalent to an unchunked prefill while
keeping the accelerator batch bounded by `max_num_batched_tokens`.

For EAGLE3, a non-final chunk uses the actual next prompt token for its shifted
draft stream. Only the final prompt chunk uses the target model's sampled first
completion token.

## Prefix cache

Prefix hits require both target KV and corresponding draft state:

- CUDA and CoreX use dense draft K/V plus per-block draft-state snapshots.
- Ascend 310P uses paged FRACTAL_NZ draft K/V indexed by the target block table.

The final prompt block stays unhashed for paged block-draft models because the
draft writes the anchor and proposal positions into that region. A target KV hit
is accepted only when the backend-specific draft state can also be restored.

## Graph execution

Decode/spec graph support is a backend capability. Ascend 310P owns its
NPUGraph implementation under `flashengine/backends/ascend310p/`; CUDA and CoreX
continue to use their graph APIs through the common runner path. Capture failure
falls back to the equivalent eager speculative round rather than changing the
commit semantics.

## Current constraints

- Target model: dense Qwen3.
- Draft methods: EAGLE3, DFlash, and DSpark.
- Speculative decoding currently requires tensor parallel size 1.
- DFlash2 is rejected explicitly; it is not silently interpreted as DFlash.
- Multi-card DP/TP for speculative state is deferred until it can be validated
  on each hardware backend.
