"""Reusable GPU vs CPU baseline for the RX 6900 XT ROCm torch wheel.

Run from pixi so the installed wheel is imported, not this source tree:

    pixi run bench
    pixi run bench -- --quick
    pixi run bench -- --json-out agent_space/bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections.abc import Callable


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)

import torch
import torch.nn.functional as F

# RX 6900 XT paper peaks: 80 CU * 64 ALU * 2 FMA * 2.25 GHz, packed FP16 = 2x.
# HIP reports 40 WGPs and a 2075 MHz clock; burst GEMM still reaches ~2.25 GHz.
PAPER_PEAK_TFLOPS = {"float32": 23.04, "float16": 46.08}
PAPER_DRAM_GBPS = 512.0


def _reject_source_checkout() -> None:
    source = os.environ.get("RX6900_SOURCE_TORCH")
    torch_file = pathlib.Path(torch.__file__).resolve()
    if source:
        source_path = pathlib.Path(source).resolve()
        if torch_file == source_path or source_path in torch_file.parents:
            raise SystemExit(
                f"Imported torch from source checkout instead of installed wheel: {torch_file}"
            )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench(fn: Callable[[], None], device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / iters


def _matmul_flops(n: int) -> float:
    return 2.0 * n * n * n


def _conv_flops(batch: int, channels: int, spatial: int, kernel: int) -> float:
    return float(batch * channels * channels * kernel * kernel * spatial * spatial * 2)


def _record(
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    ms: float,
    flops: float | None,
    extra: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "op": name,
        "device": device.type,
        "dtype": str(dtype).replace("torch.", ""),
        "ms": round(ms, 3),
        **extra,
    }
    if flops is not None and ms > 0:
        tflops = flops / (ms / 1000.0 * 1e12)
        row["tflops"] = round(tflops, 2)
        peak = PAPER_PEAK_TFLOPS.get(row["dtype"]) if device.type == "cuda" else None
        if peak:
            row["paper_peak_tflops"] = peak
            row["pct_paper_peak"] = round(100.0 * tflops / peak, 1)
    return row


def _run_matmul(
    device: torch.device,
    n: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    mode: str = "sustained",
) -> dict[str, object]:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    ms = _bench(lambda: a @ b, device, warmup, iters) * 1000
    extra: dict[str, object] = {"n": n, "mode": mode}
    return _record("matmul", device, dtype, ms, _matmul_flops(n), extra)


def _run_conv(
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    batch, channels, spatial, kernel = 8, 64, 56, 3
    x = torch.randn(batch, channels, spatial, spatial, device=device, dtype=dtype)
    w = torch.randn(channels, channels, kernel, kernel, device=device, dtype=dtype)
    ms = _bench(lambda: F.conv2d(x, w, padding=1), device, warmup, iters) * 1000
    return _record(
        "conv2d_3x3",
        device,
        dtype,
        ms,
        _conv_flops(batch, channels, spatial, kernel),
        {"batch": batch, "channels": channels, "spatial": spatial},
    )


def _run_vit_block(
    device: torch.device,
    seq: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    batch, dim, heads = 8, 768, 12
    head_dim = dim // heads
    q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    w1 = torch.randn(dim * 4, dim, device=device, dtype=dtype)
    w2 = torch.randn(dim, dim * 4, device=device, dtype=dtype)

    def block() -> None:
        y = F.scaled_dot_product_attention(q, k, v)
        hidden = F.linear(x, w1)
        F.linear(hidden, w2)
        y.transpose(1, 2).reshape(batch, seq, dim)

    ms = _bench(block, device, warmup, iters) * 1000
    return _record(
        "vit_block_sdpa_mlp",
        device,
        dtype,
        ms,
        None,
        {"seq": seq, "dim": dim, "batch": batch, "heads": heads},
    )


def _run_linear(
    device: torch.device,
    batch: int,
    seq: int,
    din: int,
    dout: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    name: str,
) -> dict[str, object]:
    x = torch.randn(batch, seq, din, device=device, dtype=dtype)
    w = torch.randn(dout, din, device=device, dtype=dtype)
    ms = _bench(lambda: F.linear(x, w), device, warmup, iters) * 1000
    return _record(
        "linear",
        device,
        dtype,
        ms,
        2.0 * batch * seq * din * dout,
        {"name": name, "batch": batch, "seq": seq, "din": din, "dout": dout},
    )


def _run_copy(
    device: torch.device,
    n: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    src = torch.randn(n, device=device, dtype=dtype)
    dst = torch.empty_like(src)
    ms = _bench(lambda: dst.copy_(src), device, warmup, iters) * 1000
    bytes_moved = 2.0 * n * src.element_size()
    row = _record("copy", device, dtype, ms, None, {"n": n})
    if ms > 0:
        gbps = bytes_moved / (ms / 1000.0) / 1e9
        row["gbps"] = round(gbps, 1)
        if device.type == "cuda":
            row["paper_peak_gbps"] = PAPER_DRAM_GBPS
            row["pct_paper_peak"] = round(100.0 * gbps / PAPER_DRAM_GBPS, 1)
    return row


def _pair_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (
            row["op"],
            row["dtype"],
            row.get("n"),
            row.get("seq"),
            row.get("spatial"),
            row.get("mode"),
            row.get("name"),
        )
        grouped.setdefault(key, {})[str(row["device"])] = row

    comparisons = []
    for key, by_device in grouped.items():
        cpu = by_device.get("cpu")
        gpu = by_device.get("cuda")
        if not cpu or not gpu:
            continue
        cpu_ms = float(cpu["ms"])
        gpu_ms = float(gpu["ms"])
        comparisons.append(
            {
                "op": key[0],
                "dtype": key[1],
                "shape": {
                    k: v
                    for k, v in (
                        ("n", key[2]),
                        ("seq", key[3]),
                        ("spatial", key[4]),
                        ("mode", key[5]),
                        ("name", key[6]),
                    )
                    if v is not None
                },
                "cpu_ms": cpu["ms"],
                "cuda_ms": gpu["ms"],
                "cuda_vs_cpu": round(cpu_ms / gpu_ms, 2) if gpu_ms else None,
                "cpu_tflops": cpu.get("tflops"),
                "cuda_tflops": gpu.get("tflops"),
            }
        )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="RX 6900 XT torch CPU vs GPU baseline")
    parser.add_argument("--quick", action="store_true", help="Smaller shapes and fewer iterations")
    parser.add_argument("--json-out", type=pathlib.Path, help="Write the full JSON report")
    args = parser.parse_args()

    _reject_source_checkout()
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda is not available")

    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    cpu = torch.device("cpu")
    gpu = torch.device("cuda")

    if args.quick:
        gemm_sizes = (1024,)
        vit_seqs = (49,)
        dtypes = (torch.float32, torch.float16)
        gpu_warmup, gpu_iters = 3, 10
        cpu_warmup, cpu_iters = 1, 2
    else:
        gemm_sizes = (1024, 2048, 4096)
        vit_seqs = (49, 64)
        dtypes = (torch.float32, torch.float16)
        gpu_warmup, gpu_iters = 5, 20
        cpu_warmup, cpu_iters = 1, 3

    rows: list[dict[str, object]] = []
    for n in gemm_sizes:
        for dtype in dtypes:
            _log(f"matmul n={n} {dtype} cuda")
            rows.append(_run_matmul(gpu, n, dtype, gpu_warmup, gpu_iters))
            # Skip CPU FP16: oneDNN is FP32-native and the ratio is not a GPU score.
            # Skip CPU 4096: one GEMM is many minutes and does not change the GPU scoreboard.
            if n < 4096 and dtype == torch.float32:
                _log(f"matmul n={n} {dtype} cpu")
                cpu_n_iters = 1 if n >= 2048 else cpu_iters
                rows.append(_run_matmul(cpu, n, dtype, cpu_warmup, cpu_n_iters))

    for dtype in dtypes:
        _log(f"conv2d {dtype} cuda")
        rows.append(_run_conv(gpu, dtype, gpu_warmup, gpu_iters))
        if dtype == torch.float32:
            _log(f"conv2d {dtype} cpu")
            rows.append(_run_conv(cpu, dtype, cpu_warmup, cpu_iters))

    for seq in vit_seqs:
        for dtype in dtypes:
            _log(f"vit seq={seq} {dtype} cuda")
            rows.append(_run_vit_block(gpu, seq, dtype, gpu_warmup, gpu_iters))
            _log(f"vit seq={seq} {dtype} cpu")
            rows.append(_run_vit_block(cpu, seq, dtype, cpu_warmup, cpu_iters))

    if not args.quick:
        for dtype in dtypes:
            _log(f"matmul n=4096 {dtype} cuda burst")
            rows.append(_run_matmul(gpu, 4096, dtype, 2, 5, mode="burst"))
        for dtype in dtypes:
            _log(f"linear mlp_up {dtype} cuda")
            rows.append(
                _run_linear(gpu, 8, 49, 768, 3072, dtype, gpu_warmup, gpu_iters, "vit_mlp_up")
            )
            rows.append(
                _run_linear(gpu, 32, 256, 768, 3072, dtype, gpu_warmup, gpu_iters, "fat_mlp_up")
            )
        _log("copy bandwidth cuda")
        rows.append(_run_copy(gpu, 64 * 1024 * 1024, torch.float32, gpu_warmup, gpu_iters))
        rows.append(_run_copy(gpu, 64 * 1024 * 1024, torch.float16, gpu_warmup, gpu_iters))

    props = torch.cuda.get_device_properties(0)
    report = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0),
        "torch_file": str(pathlib.Path(torch.__file__).resolve()),
        "quick": args.quick,
        "paper_peak_tflops": PAPER_PEAK_TFLOPS,
        "paper_dram_gbps": PAPER_DRAM_GBPS,
        "hip_clock_khz": getattr(props, "clock_rate", None),
        "hip_mp_count": props.multi_processor_count,
        "cases": rows,
        "cuda_vs_cpu": _pair_rows(rows),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
