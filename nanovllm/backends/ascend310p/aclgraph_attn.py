# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project

"""ACLGraph helpers for 310P `_npu_paged_attention` dynamic metadata.

Ascend full-graph replay does not automatically pick up new `context_lens` /
`block_table` values from device tensors alone. vLLM-Ascend records a
graph-task handle during capture and refreshes the operator via
`graph_task_update_{begin,end}` before each replay. This module mirrors that
pattern for nano-vllm 310P decode graphs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# 主仓无独立 logger；捕获/降级走 print。

_CAPTURING = False
_PENDING: list["PagedAttnCapture"] = []
_SUPPORT_CACHE: bool | None = None


@dataclass
class PagedAttnCapture:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    num_kv_heads: int
    num_heads: int
    scale: float
    block_table: torch.Tensor
    context_lens: torch.Tensor
    output: torch.Tensor
    handle: object
    event: object
    workspace: torch.Tensor | None = None


def supports_pa_graph_update() -> bool:
    """Whether to wrap PA in graph_task_group and update before replay.

    Default on 310P is False: many HDK/driver builds reject
    ``graph_task_update`` for ``_npu_paged_attention`` with
    "current driver version does not support to update this op".
    vLLM-Ascend 310P decode also calls PA directly without task-update.

    Override with env ``NANOVLLM_ACLGRAPH_PA_UPDATE``:
      0/false/off -> force disable
      1/true/on   -> force enable if APIs exist
      auto/unset  -> disable on 310P, enable elsewhere when APIs exist
    """
    global _SUPPORT_CACHE
    if _SUPPORT_CACHE is not None:
        return _SUPPORT_CACHE

    import os
    import torch_npu

    flag = os.getenv("NANOVLLM_ACLGRAPH_PA_UPDATE", "auto").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        _SUPPORT_CACHE = False
        print("[nano-vllm] Paged-attention graph-task updates disabled by environment.")
        return _SUPPORT_CACHE

    apis_ok = (
        hasattr(torch.npu, "graph_task_group_begin")
        and hasattr(torch.npu, "graph_task_group_end")
        and hasattr(torch.npu, "graph_task_update_begin")
        and hasattr(torch.npu, "graph_task_update_end")
        and hasattr(torch_npu, "_npu_paged_attention_get_workspace")
    )
    if not apis_ok:
        _SUPPORT_CACHE = False
        print(
            "[nano-vllm] Paged-attention graph-update APIs are unavailable; "
            "using copy-only graph capture."
        )
        return _SUPPORT_CACHE

    if flag in {"1", "true", "on", "yes"}:
        _SUPPORT_CACHE = True
        print("[nano-vllm] Paged-attention graph-task updates force-enabled by environment.")
        return _SUPPORT_CACHE

    # 本 runner 仅服务 310P：auto = 关闭 task-update。
    _SUPPORT_CACHE = False
    return _SUPPORT_CACHE


def begin_attn_capture() -> None:
    global _CAPTURING, _PENDING
    _CAPTURING = True
    _PENDING = []


def finish_attn_capture() -> list[PagedAttnCapture]:
    global _CAPTURING, _PENDING
    _CAPTURING = False
    captures = _PENDING
    _PENDING = []
    return captures


def is_attn_capturing() -> bool:
    return _CAPTURING


def capture_paged_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    scale: float,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run `_npu_paged_attention` inside a graph-task group and record update state."""
    import torch_npu
    stream = torch.npu.current_stream()
    workspace = torch_npu._npu_paged_attention_get_workspace(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale_value=scale,
        block_table=block_table,
        context_lens=context_lens,
        out=out,
    )
    # Do NOT call ExternalEvent.wait/reset on the capture stream: on some
    # torch_npu builds this raises "stream is not registered with any allocator"
    # and corrupts subsequent NPU allocations.
    event = None
    if hasattr(torch.npu, "ExternalEvent"):
        try:
            event = torch.npu.ExternalEvent()
        except Exception:  # pragma: no cover - env dependent
            event = None

    torch.npu.graph_task_group_begin(stream)
    torch_npu._npu_paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale_value=scale,
        block_table=block_table,
        context_lens=context_lens,
        out=out,
        workspace=workspace,
    )
    handle = torch.npu.graph_task_group_end(stream)

    _PENDING.append(
        PagedAttnCapture(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            block_table=block_table,
            context_lens=context_lens,
            output=out,
            handle=handle,
            event=event,
            workspace=workspace,
        )
    )
    return out


def update_paged_attention_params(
    captures: list[PagedAttnCapture],
    *,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    update_stream: torch.npu.Stream,
) -> bool:
    """Patch every captured PA node with the current block_table / context_lens.

    Returns False if update is unsupported/failed so caller can fall back.
    """
    if not captures:
        return False
    import torch_npu
    try:
        with torch.npu.stream(update_stream):
            workspace = None
            for cap in captures:
                if workspace is None:
                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=cap.query,
                        key_cache=cap.key_cache,
                        value_cache=cap.value_cache,
                        num_kv_heads=cap.num_kv_heads,
                        num_heads=cap.num_heads,
                        scale_value=cap.scale,
                        block_table=block_table,
                        context_lens=context_lens,
                        out=cap.output,
                    )
                torch.npu.graph_task_update_begin(update_stream, cap.handle)
                torch_npu._npu_paged_attention(
                    query=cap.query,
                    key_cache=cap.key_cache,
                    value_cache=cap.value_cache,
                    num_kv_heads=cap.num_kv_heads,
                    num_heads=cap.num_heads,
                    scale_value=cap.scale,
                    block_table=block_table,
                    context_lens=context_lens,
                    out=cap.output,
                    workspace=workspace,
                )
                torch.npu.graph_task_update_end(update_stream)
                if cap.event is not None:
                    try:
                        cap.event.record(update_stream)
                    except Exception:
                        pass
        torch.npu.current_stream().wait_stream(update_stream)
        return True
    except Exception as exc:
        print(
            f"[nano-vllm] Paged-attention graph-task update failed "
            f"({type(exc).__name__}: {exc}); falling back to eager or copy-only replay."
        )
        return False
