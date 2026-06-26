"""Stream utilities for Inductor codegen."""

from __future__ import annotations

import functools

from torch._inductor.stream_constants import (
    DEFAULT_STREAM,
    DEFAULT_STREAM_IDX,
    STREAM_NAME_TEMPLATE,
)


__all__ = [
    "DEFAULT_STREAM",
    "DEFAULT_STREAM_IDX",
    "STREAM_NAME_TEMPLATE",
    "get_raw_stream_name",
    "get_stream_name",
]


@functools.lru_cache
def get_stream_name(stream_idx: int) -> str:
    """Generate CUDA Stream name from stream index number.

    Args:
        stream_idx: Non-negative index number. 0 refers to the default stream, others refer to side
            streams.
    """
    if stream_idx == 0:
        return DEFAULT_STREAM
    else:
        return STREAM_NAME_TEMPLATE.format(stream_idx=stream_idx)


@functools.lru_cache
def _raw_stream_name_for_device(device_idx: int) -> str:
    return f"raw_stream{device_idx}"


def get_raw_stream_name(device_idx: int) -> str:
    """Generate variable name for a raw stream handle on the given device."""
    # Under compile-on-one-rank the wrapper must be byte-identical across ranks, so the
    # stream variable name cannot carry a rank-specific device index.
    from torch.fx.experimental.proxy_tensor import _coor_enabled

    if _coor_enabled():
        return "raw_stream"
    return _raw_stream_name_for_device(device_idx)
