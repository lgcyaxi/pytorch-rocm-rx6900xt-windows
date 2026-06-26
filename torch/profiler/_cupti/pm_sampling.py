# mypy: allow-untyped-defs
"""CUPTI PM-sampling: continuous GPU performance-monitor sampling (SM-active %, DRAM-throughput
%) that runs concurrently with the activity monitor.

PM sampling reads dedicated on-chip performance-monitor units, so it has negligible GPU-side cost
and coexists with the activity subscriber, but it locks the GPU clocks while active (which can
shift absolute kernel durations) -- so it is opt-in via custom_profiler_config
{"enable_pm_sampling": true}, not always-on like the environment counters.

The HW units sample autonomously into a device-side ring (KEEP_LATEST) between start and decode,
so there is no background thread: the session starts at window open and the ring is drained once
at window close -- crucially *before* stop(), since a wrapped ring is only decodable while
sampling is active. A window that fits the ring yields all its samples; one that exceeds it
keeps the most recent (the trace's tail) rather than erroring. ``decode`` drains the whole ring
in one call and caps at its ``max_samples`` (a second call does not resume), so the host image is
sized above the ring's sample capacity. Samples are HW-timestamped in the CUPTI clock domain, the
same base the monitor's record timestamps use, so running them through the monitor's clock
conversion aligns them with the activity records; they surface as GPU counter tracks (siblings of
the environment counters).

One session per in-use CUDA device, so a multi-GPU process gets PM counters on every device. The
profiler init/deinit the per-device collectors share is process-global and *not* refcounted (a
second ``profiler_deinitialize`` crashes), so teardown disables every device's sampling while the
profiler is still alive and then deinitializes exactly once (see :meth:`_teardown`)."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)

# (counter_id, display name, cupti metric). counter_ids continue after the environment counters
# (1-5, see monitor_trace._ENV_COUNTERS) so GpuCounterDescriptor ids stay unique across all GPU
# counter tracks. The full sm__throughput.* metric needs many passes (not single-pass, so not
# PM-sampleable); sm__cycles_active (SM-active %) is the single-pass SM-utilization metric.
# NVLink is bidirectional, so RX/TX are tracked separately (no single aggregate metric exists);
# all of these fit in one PM-sampling pass.
PM_METRICS: tuple[tuple[int, str, str], ...] = (
    (6, "SM Active (%)", "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"),
    (7, "DRAM BW (%)", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
    (8, "NVLink RX (%)", "nvlrx__bytes.avg.pct_of_peak_sustained_elapsed"),
    (9, "NVLink TX (%)", "nvltx__bytes.avg.pct_of_peak_sustained_elapsed"),
)

_SAMPLING_INTERVAL_NS = (
    1_000_000  # 1 ms HW sampling interval (GPU_TIME_INTERVAL units = ns)
)
# Ring size: bounds the retained window (~24 s at 1 ms). A window within this decodes whole; a
# longer one keeps the most recent ~24 s. Allocated only while PM sampling is enabled and freed
# at disable.
_HW_BUFFER_SIZE = 256 * 1024 * 1024
# decode() drains the whole HW buffer in one call and caps at max_samples (it does not resume on a
# second call), so size the host image above the buffer's sample capacity. ~4 KiB/sample is a
# conservative lower bound (measured ~10 KiB), so a window that fit the buffer never caps.
_MAX_SAMPLES = _HW_BUFFER_SIZE // 4096


def is_available() -> bool:
    try:
        import cupti.pm_sampling  # noqa: F401  # pyrefly: ignore[missing-import]
    except Exception:
        return False
    return True


def _active_cuda_devices() -> list[int]:
    """CUDA devices with a live primary context -- the devices actually in use. PM sampling only
    touches these: enabling on a context-less device fails the chip-name query, and that failure
    triggers the collector's rollback, which deinitializes the *shared* profiler and would break
    the other devices' sessions. torch uses primary contexts, so this is the right set."""
    import torch

    try:
        from cuda.bindings import driver as drv  # pyrefly: ignore[missing-import]
    except Exception:
        return [torch.cuda.current_device()]
    active = []
    for d in range(torch.cuda.device_count()):
        try:
            dev = drv.cuDeviceGet(d)[1]
            # cuDevicePrimaryCtxGetState returns (result, flags, active).
            rc, _, in_use = drv.cuDevicePrimaryCtxGetState(dev)
            if int(rc) == 0 and in_use:
                active.append(d)
        except Exception:
            pass
    return active or [torch.cuda.current_device()]


class PmSampler:
    """PM-sampling sessions -- one per in-use CUDA device -- started at window open and drained at
    window close, both on the calling (foreground) thread, no background polling. Decoded samples
    are converted to unix-epoch ns and handed to ``sink`` as a named-column frame (``start_ns``/
    ``device_id``/``c<counter_id>`` per metric) for the observer to bucket into its window. The
    devices are those with a live primary context at :meth:`start`."""

    def __init__(
        self,
        sink: Callable[[dict[str, Any]], None],
        convert_ns: Callable[[np.ndarray], np.ndarray],
        *,
        sampling_interval_ns: int = _SAMPLING_INTERVAL_NS,
    ) -> None:
        self._sink = sink
        self._convert_ns = convert_ns
        self._sampling_interval_ns = sampling_interval_ns
        self._metric_names = [m for _, _, m in PM_METRICS]
        self._cols: dict[int, Any] = {}

    def start(self) -> None:
        """Enable + configure + start a collector on each in-use device. No-op if already started
        or PM sampling is unavailable; a per-device failure tears the whole set down (a failed
        enable rolls back the shared profiler, so the others can no longer be trusted)."""
        if self._cols or not is_available():
            return
        col = None
        try:
            from cupti import pm_sampling as pm  # pyrefly: ignore[missing-import]

            for device in _active_cuda_devices():
                col = pm.Collector(device_index=device)
                col.enable()
                col.configure(
                    metrics=self._metric_names,
                    hardware_buffer_size=_HW_BUFFER_SIZE,
                    sampling_interval=self._sampling_interval_ns,
                    trigger_mode=pm.TriggerMode.GPU_TIME_INTERVAL,
                    # KEEP_LATEST = ring buffer: an over-capacity window keeps the most recent
                    # samples (the trace's tail) instead of erroring. Requires decoding before
                    # stop() (see :meth:`stop`); a wrapped ring is not decodable after stop.
                    hw_buffer_append_mode=pm.HardwareBuffer_AppendMode.KEEP_LATEST,
                )
                col.start()
                self._cols[device] = col
                col = None
        except Exception as e:
            logger.warning("PM sampling could not start: %s", e)
            # If enable() succeeded but configure()/start() failed, this collector holds a live PM
            # object + finalizer but is not yet in _cols; fold it in so teardown disables it and
            # detaches its finalizer (otherwise the finalizer re-deinitializes the profiler at GC).
            if (
                col is not None
                and getattr(col, "_pm_sampling_object", None) is not None
            ):
                self._cols[col._device_index] = col
            self._teardown()

    def stop(self) -> None:
        """Drain each device's HW buffer into ``sink`` while still running, then stop + tear down.
        Idempotent. The drain MUST precede ``col.stop()``: a wrapped KEEP_LATEST ring is only
        decodable while sampling is active (decoding a wrapped ring after stop fails)."""
        cols = self._cols
        if not cols:
            return
        self._cols = {}
        try:
            for device, col in cols.items():
                self._drain(col, device)
            for col in cols.values():
                col.stop()
        except Exception:
            logger.exception("PM sampling decode error")
        finally:
            self._teardown(cols)

    def _teardown(self, cols: dict[int, Any] | None = None) -> None:
        """Disable every collector's sampling, then deinitialize the profiler exactly once. The
        collectors share a process-global profiler whose init/deinit is not refcounted, so we
        cannot call ``Collector.disable`` per device (each would deinit, and the second crashes):
        instead disable each session's sampling directly (profiler still alive), detach the
        collector finalizers that would re-deinit at GC, then deinitialize once at the end."""
        cols = self._cols if cols is None else cols
        if not cols:
            return
        from cupti import cupti as c  # pyrefly: ignore[missing-import]

        for col in cols.values():
            obj = getattr(col, "_pm_sampling_object", None)
            if obj is not None:
                try:
                    p = c.PmSampling_Disable_Params()
                    p.struct_size = c.PM_SAMPLING_DISABLE_PARAMS_STRUCT_SIZE
                    p.p_pm_sampling_object = obj
                    c.pm_sampling_disable(p.ptr)
                except Exception:
                    logger.exception("PM sampling disable error")
            fin = getattr(col, "_finalizer", None)
            if fin is not None:
                fin.detach()
            col._pm_sampling_object = None
        try:
            dp = c.Profiler_DeInitialize_Params()
            dp.struct_size = c.PROFILER_DEINITIALIZE_PARAMS_STRUCT_SIZE
            c.profiler_deinitialize(dp.ptr)
        except Exception:
            logger.exception("PM sampling profiler deinitialize error")
        self._cols = {}

    def _drain(self, col: Any, device: int) -> None:
        # A single decode drains the whole ring (see module docstring). With KEEP_LATEST a
        # wrapped ring just yields its most-recent samples, so no overflow error; the MemoryError
        # guard is a defensive backstop only.
        try:
            cd = col.decode(max_samples=_MAX_SAMPLES)
        except MemoryError:
            logger.warning(
                "PM sampling HW buffer overflow during decode; samples dropped."
            )
            return
        n = cd.num_completed_samples
        if not n:
            return
        if n >= _MAX_SAMPLES:
            # The image filled before the buffer drained; the newest samples past the cap were
            # dropped. Sized not to happen for a window that fit the buffer, so this is a backstop.
            logger.warning(
                "PM sampling decoded the maximum %d samples; some were dropped.",
                _MAX_SAMPLES,
            )
        # Iterating CounterData evaluates the metrics host-side (the decode cost).
        samples = [cd[i] for i in range(n)]
        # Stamp at the sample's interval START. The value is the average over [start, end] (~1 ms),
        # and the viewer draws a counter as a step from each sample's ts to the next -- so start
        # makes the step span exactly its measurement window, lining the high-counter region up
        # with the activity span (end would lag a full interval; midpoint would offset the edges).
        ts = np.fromiter((s.start_timestamp for s in samples), dtype=np.int64, count=n)
        # The very first sample of a session has an uninitialized (0) start timestamp; drop it.
        keep = ts > 0
        if not keep.any():
            return
        ts = ts[keep]
        vals = np.array([s.metric_values for s in samples], dtype=np.float64)[keep]
        frame: dict[str, Any] = {
            "start_ns": self._convert_ns(ts),
            "device_id": np.full(len(ts), device, dtype=np.int64),
        }
        for j, (cid, _, _) in enumerate(PM_METRICS):
            frame[f"c{cid}"] = vals[:, j]
        self._sink(frame)
