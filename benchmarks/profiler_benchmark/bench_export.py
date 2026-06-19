"""CUPTI monitor export-latency benchmark.

Measures ``torch.profiler.export_chrome_trace`` latency as a function of trace size,
comparing stock Kineto (trunk) against the cupti_monitor backend.

The cupti_monitor backend exports ASYNCHRONOUSLY: ``export_chrome_trace`` only captures
the Kineto CPU-side trace + registers the output path (the hot-path cost that stalls the
calling/training thread), while the GPU-record merge + gzip happen off-thread, drained by
``wait_for_exports``. Stock Kineto does the whole thing synchronously inside
``export_chrome_trace`` (``wait_for_exports`` is a no-op). So this reports:

  export_ms        : the synchronous hot-path cost (per profiled step, the part that
                     stalls the training thread)
  wait_exports_ms  : the deferred merge + gzip (monitor only; off-thread in real use)
  total_ms         : export_ms + wait_exports_ms (the total export work)

Each ``--mode`` runs in its own process so the monitor gets a clean CUPTI subscriber
(Kineto holds the single CUPTI subscriber for the process once a CUDA window runs).
``--kernels`` scales the eager workload's kernel count -> the trace's event count
(~6 traceEvents per kernel here; pass e.g. --kernels 57000 for a ~340k-event trace).

Run on a host with libcupti >= 13.3 visible to the monitor, e.g.
``LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat python benchmarks/profiler_benchmark/bench_export.py --mode monitor --kernels 40000``.
"""

import argparse
import gzip
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import profile, ProfilerActivity


def run_export(mode: str, kernels: int, iters: int) -> dict:
    use_monitor = mode == "monitor"
    cfg = _ExperimentalConfig(
        custom_profiler_config='{"backend":"cupti_monitor"}' if use_monitor else ""
    )
    x = torch.randn(512, 512, device="cuda")
    for _ in range(50):  # warm up kernels / allocator
        x = torch.relu(x)
    torch.cuda.synchronize()

    export_ms: list[float] = []
    wait_ms: list[float] = []
    n_events = 0
    out_mb = 0.0
    with tempfile.TemporaryDirectory() as td:
        trace_path = str(Path(td) / "trace.json.gz")
        for _ in range(iters):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                experimental_config=cfg,
            ) as prof:
                for _ in range(kernels):
                    x = torch.relu(x)
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            prof.export_chrome_trace(trace_path)
            export_ms.append((time.perf_counter() - t0) * 1e3)
            t1 = time.perf_counter()
            prof.wait_for_exports()  # no-op for stock Kineto
            wait_ms.append((time.perf_counter() - t1) * 1e3)

            out_mb = os.path.getsize(trace_path) / 1e6
            with gzip.open(trace_path, "rt") as f:
                n_events = len(json.load(f)["traceEvents"])
            os.remove(trace_path)

    em = statistics.median(export_ms)
    wm = statistics.median(wait_ms)
    return {
        "mode": mode,
        "kernels": kernels,
        "iters": iters,
        "trace_events": n_events,
        "out_mb": round(out_mb, 1),
        "export_ms": round(em, 1),
        "wait_exports_ms": round(wm, 1),
        "total_ms": round(em + wm, 1),
        "us_per_event": round((em + wm) * 1e3 / max(n_events, 1), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["trunk", "monitor"], required=True)
    parser.add_argument("--kernels", type=int, default=40000)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    torch.cuda.init()
    print(
        json.dumps(
            run_export(args.mode, args.kernels, args.iters), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
