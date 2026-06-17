# mypy: allow-untyped-defs
"""Chrome-trace observer for the CUPTI activity multiplexer.

``ProfilerObserver`` is the multiplexer-backed form of the old monitor trace
path: it wants the trace activity kinds, accumulates the decoded GPU/API records
the monitor delivers, tracks the user-annotation and thread metadata the chrome
trace needs, and on ``drain()`` hands them back as a trace-window dict -- the
exact shape ``monitor_trace.merge_trace_window_into_chrome_trace`` consumes to
splice CUPTI activity into a stock Kineto chrome trace. The trace assembly is
entirely ``monitor_trace``'s existing logic; this class is the collection,
annotation join, and windowing around it.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading
import time
from typing import Any, TYPE_CHECKING

from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

import torch
from torch.profiler._cupti.cupti_python import OVERHEAD_KIND_NAMES
from torch.profiler._cupti.monitor_trace import merge_trace_window_into_chrome_trace
from torch.profiler._cupti.observers.base import (
    CuptiMonitorObserver,
    ObserverAnnotationSettings,
)
from torch.profiler._cupti.observers.windowing import WindowFinalizerMixin
from torch.profiler._cupti.records import (
    Api,
    CudaEvent,
    ExternalCorrelation,
    Field,
    Kernel,
    Memcpy,
    Memset,
    Overhead,
    Sync,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.profiler._cupti.observers.base import AnnotationResolver


def _current_thread_resource_tuple() -> tuple[int, int, int]:
    # (pid, opaque 32-bit thread id, system thread id) for trace lane naming.
    opaque_tid = ctypes.c_int32(threading.get_ident() & 0xFFFFFFFF).value
    return (os.getpid(), opaque_tid, threading.get_native_id())


_DEMANGLE_CACHE: dict[str, str] = {}


def _demangle_symbol(name: str) -> str:
    cached = _DEMANGLE_CACHE.get(name)
    if cached is None:
        cached = _DEMANGLE_CACHE[name] = torch._C._demangle(name)
    return cached


# The CUPTI fields ProfilerObserver requests: GPU work plus the CPU-side
# runtime/driver/overhead records and external correlation for the annotation join.
# (Omits fields the chrome trace never consumes, e.g. memcpy runtime_correlation_id
# / overhead object_kind -- which also have no v2 user-defined-record id.)
PROFILER_FIELDS: dict[ActivityKind, set[Field]] = {
    ActivityKind.CONCURRENT_KERNEL: {
        Kernel.START,
        Kernel.END,
        Kernel.DEVICE_ID,
        Kernel.CONTEXT_ID,
        Kernel.STREAM_ID,
        Kernel.CORRELATION_ID,
        Kernel.GRAPH_NODE_ID,
        Kernel.GRAPH_ID,
        Kernel.NAME,
        # Launch config (kineto reports these; for eager kernels this is the only
        # source -- graphs can also recover grid/block from kernel annotations).
        Kernel.GRID_X,
        Kernel.GRID_Y,
        Kernel.GRID_Z,
        Kernel.BLOCK_X,
        Kernel.BLOCK_Y,
        Kernel.BLOCK_Z,
        Kernel.REGISTERS_PER_THREAD,
        Kernel.STATIC_SHARED_MEMORY,
        Kernel.DYNAMIC_SHARED_MEMORY,
        Kernel.LAUNCH_PRIORITY,
        Kernel.QUEUED,
        Kernel.CHANNEL_ID,
        Kernel.CHANNEL_TYPE,
    },
    ActivityKind.MEMCPY: {
        Memcpy.START,
        Memcpy.END,
        Memcpy.DEVICE_ID,
        Memcpy.CONTEXT_ID,
        Memcpy.STREAM_ID,
        Memcpy.CORRELATION_ID,
        Memcpy.GRAPH_NODE_ID,
        Memcpy.GRAPH_ID,
        Memcpy.BYTES,
        Memcpy.COPY_KIND,
        Memcpy.SRC_KIND,
        Memcpy.DST_KIND,
        Memcpy.FLAGS,
    },
    ActivityKind.MEMSET: {
        Memset.START,
        Memset.END,
        Memset.DEVICE_ID,
        Memset.CONTEXT_ID,
        Memset.STREAM_ID,
        Memset.CORRELATION_ID,
        Memset.GRAPH_NODE_ID,
        Memset.GRAPH_ID,
        Memset.BYTES,
        Memset.VALUE,
        Memset.MEMORY_KIND,
        Memset.FLAGS,
    },
    ActivityKind.RUNTIME: {
        Api.CBID,
        Api.START,
        Api.END,
        Api.PROCESS_ID,
        Api.THREAD_ID,
        Api.CORRELATION_ID,
    },
    ActivityKind.DRIVER: {
        Api.CBID,
        Api.START,
        Api.END,
        Api.PROCESS_ID,
        Api.THREAD_ID,
        Api.CORRELATION_ID,
    },
    ActivityKind.EXTERNAL_CORRELATION: {
        ExternalCorrelation.EXTERNAL_KIND,
        ExternalCorrelation.EXTERNAL_ID,
        ExternalCorrelation.CORRELATION_ID,
    },
    ActivityKind.OVERHEAD: {
        Overhead.OVERHEAD_KIND,
        Overhead.START,
        Overhead.END,
        Overhead.CORRELATION_ID,
    },
}


# CUDA synchronization + event fields, selected only when the caller opts in via
# enable_cuda_sync_events -- matching kineto, which emits cuda_sync events only under
# that flag. SYNCHRONIZATION carries the sync spans; CUDA_EVENT records are join inputs
# that resolve which cudaEventRecord a wait refers to (the wait_on join in monitor_trace).
SYNC_FIELDS: dict[ActivityKind, set[Field]] = {
    ActivityKind.SYNCHRONIZATION: {
        Sync.TYPE,
        Sync.START,
        Sync.END,
        Sync.CORRELATION_ID,
        Sync.CONTEXT_ID,
        Sync.STREAM_ID,
        Sync.CUDA_EVENT_ID,
        Sync.CUDA_EVENT_SYNC_ID,
    },
    ActivityKind.CUDA_EVENT: {
        CudaEvent.CORRELATION_ID,
        CudaEvent.CONTEXT_ID,
        CudaEvent.STREAM_ID,
        CudaEvent.EVENT_ID,
        CudaEvent.DEVICE_ID,
        CudaEvent.CUDA_EVENT_SYNC_ID,
    },
}


class ProfilerObserver(WindowFinalizerMixin, CuptiMonitorObserver):
    """Accumulates decoded activity records and exports them as chrome-trace windows
    *asynchronously*. A window is opened at trace start (:meth:`open_window`) and
    closed at trace stop (:meth:`close_window`), but the merge + file write is
    deferred until the records the window covers have been NATURALLY delivered -- no
    device sync on the measured timeline (see :class:`WindowFinalizerMixin`). The
    caller supplies the output path via :meth:`set_export`; :meth:`join` blocks until
    every pending window is written (for teardown / ``wait_for_exports``).

    It requests the chrome-trace field selection (``PROFILER_FIELDS``) -- GPU work
    plus the CPU-side runtime/driver/overhead records and external correlation for
    the annotation join -- and the monitor hands it those fields as columns.

    Bucketing is in the converted (unix) clock to match the events' ``start_ns``; the
    boundary is ``convert_time(native)`` so the comparison stays order-equivalent
    (convert_time is monotonic)."""

    def __init__(
        self,
        annotation_resolver: AnnotationResolver | None = None,
        metadata_resolver: Callable[[int], str | None] | None = None,
        enable_cuda_sync: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        # Timestamped events (have "start_ns"), bucketed into windows by start time.
        self._events: list[dict[str, Any]] = []
        # Untimestamped events -- EXTERNAL_CORRELATION, the correlation_id ->
        # external_id metadata for the eager annotation join. They carry no time to
        # bucket by, so every finalized window includes whatever is currently buffered
        # (a harmless superset for the join) and clears them.
        self._ext_events: list[dict[str, Any]] = []
        # {external_id: opaque blob} drained from the metadata store alongside the
        # records (see CuptiMonitor.take_external_metadata). Joined onto events by
        # correlation_id -> external_id -> blob at finalize, the same chain as the
        # name join. Consumed per window like _ext_events. (Single-consumer: the
        # store is process-global, so the first observer to take() drains it.)
        self._ext_metadata: dict[int, str] = {}
        # graph_node_id -> blob for CUDA-graph-captured collectives, whose replay
        # kernels have no external-correlation link. Resolved the same way as
        # graph-node annotation names (a registry keyed by the stable exec-graph node
        # id); None when no graph metadata is wired. NOTE: like the kernel-annotation
        # registry it mirrors, the backing store is NOT reclaimed per graph -- entries
        # for a destroyed graph persist until a wholesale clear (acceptable: bounded
        # by distinct captured graph nodes; reset per session). A CUDA user-object
        # destructor could reclaim per graph if this ever grows unbounded.
        self._metadata_resolver = metadata_resolver
        # pid -> {opaque_tid: system_tid}, for naming GPU/CPU lanes in the trace.
        self._thread_resource_map: dict[int, dict[int, int]] = {}
        # Start boundary of the currently-open window (None when no window is open).
        self._open_start: int | None = None
        # window_id -> pending state: start, captured annotations/thread map, the
        # cpu-trace + output paths (set via set_export), and the built window dict
        # (set on coverage). Written + dropped once built AND paths are present.
        self._windows: dict[int, dict[str, Any]] = {}
        # Graph-node naming via the base resolver (self._resolver; custom or default).
        # PROFILER_FIELDS already selects RUNTIME + EXTERNAL_CORRELATION + correlation
        # ids and the eager join happens in monitor_trace, so this doesn't opt into the
        # base's eager augmentation.
        selection = {k: set(v) for k, v in PROFILER_FIELDS.items()}
        if enable_cuda_sync:
            selection.update({k: set(v) for k, v in SYNC_FIELDS.items()})
        super().__init__(
            selection,
            annotations=ObserverAnnotationSettings(
                graph=True, custom_graph_annotation_resolver=annotation_resolver
            ),
        )
        if self.available:
            self._init_windowing(
                poll_interval_ms=20, thread_name="cupti-profiler-export"
            )

    def _boundary_clock_ns(self) -> int:
        # Events carry convert_time(native) start_ns, so stamp the boundary the same
        # way: convert_time is monotonic, so comparing in the converted clock is
        # order-equivalent to comparing the raw native timestamps.
        return self.convert_time(self.now_native_ns())

    def _on_activities(self, columns: dict[Any, dict[int, Any]]) -> None:
        # Worker thread: build chrome-trace event dicts and accumulate them. Only while
        # a window is open or still pending export -- otherwise drop the columns rather
        # than pay the O(records) build for activity no window will ever consume.
        if not columns:
            return
        with self._lock:
            active = self._open_start is not None or bool(self._windows)
        if not active:
            return
        events: list[dict[str, Any]] = []
        for kind, cols in columns.items():
            events.extend(
                events_from_columns(
                    int(kind),
                    cols,
                    convert_time=self.convert_time,
                    annotation_resolver=self._resolver,
                )
            )
        if not events:
            return
        timed = [e for e in events if "start_ns" in e]
        ext = [e for e in events if "start_ns" not in e]
        self._annotate_user_external_ids(ext)
        meta = (
            self._monitor.take_external_metadata() if self._monitor is not None else {}
        )
        with self._lock:
            self._events.extend(timed)
            self._ext_events.extend(ext)
            if meta:
                self._ext_metadata.update(meta)

    def _annotate_user_external_ids(self, ext_events: list[dict[str, Any]]) -> None:
        """Stamp each EXTERNAL_CORRELATION event with ``user_external_id``: the
        innermost ENCLOSING id that names a region, via the monitor's active-id chain,
        so a kernel nested below a named region (e.g. a collective inside it) still
        gets that region's name in the trace -- the single-kind record only carries
        the innermost id. Resolved here at dispatch while the chain is live; the
        metadata join keeps using the raw innermost ``external_id`` (a collective's id
        keys its blob). No-op without a monitor or any named region active."""
        if self._monitor is None:
            return
        names = set(self.annotation_names())
        if not names:
            return
        chain = self._monitor.external_id_chain
        for e in ext_events:
            if e.get("kind") != "external_correlation":
                continue
            eid = e["external_id"]
            e["user_external_id"] = next(
                (c for c in reversed(chain(eid)) if c in names), eid
            )

    def push_annotation(self, name: str) -> int | None:
        # Record the calling thread (for trace-lane naming) on top of the base
        # external-correlation push (which owns the id -> name mapping).
        self._record_calling_thread()
        return super().push_annotation(name)

    def _record_calling_thread(self) -> None:
        pid, opaque_tid, sys_tid = _current_thread_resource_tuple()
        with self._lock:
            self._thread_resource_map.setdefault(pid, {})[opaque_tid] = sys_tid

    # --- async window API (the cupti_monitor profiler backend drives these) ----

    def open_window(self) -> None:
        """Mark the start of a trace window. Records before this point are excluded
        from it (so prepare-phase activity doesn't leak into the trace)."""
        with self._lock:
            self._open_start = self._boundary_clock_ns()

    def close_window(self) -> int | None:
        """Mark the end of the open window and queue it for deferred export. Snapshots
        the window's annotations + thread map now; pair with :meth:`set_export` to
        supply the paths. Returns the window id (None when unavailable)."""
        if not self.available:
            return None
        with self._lock:
            start = self._open_start if self._open_start is not None else 0
            self._open_start = None
            annotations = self.annotation_names(reset=True)
            thread_map = {
                pid: dict(mapping) for pid, mapping in self._thread_resource_map.items()
            }
        window_id = self.mark_boundary()
        with self._lock:
            self._windows[window_id] = {
                "start": start,
                "annotations": annotations,
                "thread_map": thread_map,
                "cpu": None,
                "out": None,
                "built": None,
            }
        return window_id

    def set_export(
        self,
        window_id: int,
        cpu_trace_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
    ) -> None:
        """Supply the (already-captured) Kineto CPU trace and the desired output path
        for a closed window. The merged file is written once the window's records are
        covered -- now if they already are, else by the background poller."""
        with self._lock:
            w = self._windows.get(window_id)
            if w is None:
                return
            w["cpu"] = os.fspath(cpu_trace_path)
            # CUPTI monitor traces are always written gzipped; ensure the .gz suffix so
            # the file is both compressed (the writer keys gzip off it) and named
            # correctly, regardless of what the caller passed.
            out = os.fspath(output_path)
            w["out"] = out if out.endswith(".gz") else out + ".gz"
        self._maybe_write(window_id)

    def join(self, *, force: bool = True, timeout_s: float = 30.0) -> None:
        """Finalize + write every pending window, then unregister. Idempotent.

        ``force`` (default) sync-flushes CUPTI to deliver the tail immediately -- use
        on the training thread (where it serializes with any other monitor flusher)
        when the file is needed now. ``force=False`` is for an off-thread finalize
        while collection continues: it must NOT flush (that would race the monitor /
        the SubgraphTimer's flushing), so it waits up to ``timeout_s`` for the poll
        thread to cover the windows from naturally-delivered records, then finalizes
        whatever is in hand (force-draining only if coverage stalled past the
        deadline)."""
        if getattr(self, "_boundaries", None) is not None:
            sync = force
            if not force:
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline:
                    with self._win_lock:
                        if not self._boundaries:
                            break
                    time.sleep(self._poll_interval_s)
                with self._win_lock:
                    sync = bool(
                        self._boundaries
                    )  # stalled -> fall back to a forced drain
            self._stop_windowing(sync=sync)
        # Write every window whose output path was set. Writing happens here (and in
        # set_export) on the foreground -- never the poll thread -- so it stays inside
        # the caller's temp-dir lifetime.
        for window_id in list(self._windows):
            self._maybe_write(window_id)
        super().close()

    # --- WindowFinalizerMixin hooks -------------------------------------------

    def _collect_delivered(self, *, sync: bool) -> None:
        # Events are built in _on_activities as buffers arrive; only nudge a flush at
        # teardown (sync) to deliver the tail. The periodic cycle never flushes.
        if sync and self._monitor is not None:
            self._monitor.flush(sync=True)

    def _window_watermark_ns(self) -> int:
        with self._lock:
            return max((e["start_ns"] for e in self._events), default=-1)

    def _finalize_window(self, window_id: int, boundary_ns: int) -> None:
        # Runs on the poll thread: the window's records are all in hand, so BUILD its
        # trace-window dict (events in [start, boundary)) in memory and drop the
        # consumed events. Does NOT write -- writing is a foreground concern
        # (set_export / join), so a deferred write can never outlive the caller's
        # (temporary) output directory.
        with self._lock:
            w = self._windows.get(window_id)
            if w is None:
                return
            start = w["start"]
            timed = [e for e in self._events if start <= e["start_ns"] < boundary_ns]
            self._events = [e for e in self._events if e["start_ns"] >= boundary_ns]
            # Untimestamped external-correlation events ride along (the join needs
            # them); consume the buffered ones with this window.
            ext, self._ext_events = self._ext_events, []
            meta, self._ext_metadata = self._ext_metadata, {}
        # Attach the opaque metadata blob onto the events it annotates (no lock: the
        # timed/ext lists are this thread's now). correlation_id -> external_id (from
        # the window's EXTERNAL_CORRELATION records) -> blob.
        _attach_metadata(timed, ext, meta, self._metadata_resolver)
        with self._lock:
            w = self._windows.get(window_id)
            if w is None:
                return
            w["built"] = {
                "events": timed + ext,
                "user_annotations": w["annotations"],
                "thread_resource_map": w["thread_map"],
                "start_ns": start,
            }

    def _maybe_write(self, window_id: int) -> None:
        # Write once the window is both built (covered) and has its paths (set_export).
        # Only ever called on the foreground (set_export / join), never the poll thread.
        # The del-under-lock makes the writer single.
        with self._lock:
            w = self._windows.get(window_id)
            if w is None or w["built"] is None or w["out"] is None:
                return
            built, cpu, out = w["built"], w["cpu"], w["out"]
            del self._windows[window_id]
        try:
            merge_trace_window_into_chrome_trace(cpu, out, built, trace_name=out)
        finally:
            # The CPU trace is a throwaway snapshot the caller handed us; drop it.
            with contextlib.suppress(OSError):
                os.remove(cpu)


# --- columns -> chrome-trace event dicts -------------------------------------
# Turn the per-kind columns the monitor delivers into the event dicts
# monitor_trace splices into the chrome trace. Owns the per-kind event shape, the
# symbol demangling, the clock conversion, and the graph-annotation resolution --
# all ProfilerObserver/presentation concerns (the monitor only produces columns).


def events_from_columns(
    kind: int,
    cols: dict[int, Any],
    *,
    convert_time: Callable[[int], int],
    annotation_resolver: AnnotationResolver | None,
) -> list[dict[str, Any]]:
    """Turn one kind's columns (``{field_id: column}``) into chrome-trace event
    dicts. Returns [] for kinds this builder doesn't render. A None resolver means
    no graph-node naming (every annotation resolves to None)."""
    builder = _BUILDERS.get(kind)
    if builder is None:
        return []
    resolver = annotation_resolver or (lambda *_: None)
    return builder(cols, convert_time, resolver)


def _col_len(cols: dict[int, Any]) -> int:
    return len(next(iter(cols.values()))) if cols else 0


def _attach_metadata(
    timed: list[dict[str, Any]],
    ext_events: list[dict[str, Any]],
    external_metadata: dict[int, str],
    metadata_resolver: Callable[[int], str | None] | None,
) -> None:
    """Attach the opaque per-annotation blob onto each timed event, by two routes:

    1. Eager: ``correlation_id -> external_id`` (from the window's
       EXTERNAL_CORRELATION records) ``-> blob``, the same chain the eager name
       join uses. This is the path for eager (non-graph) collectives, where the
       push produced an EXTERNAL_CORRELATION record linking the id to the launched
       kernel. (The monitor is the sole CUPTI subscriber and sole external-
       correlation pusher, so every such record is ours -- no kind filtering.)
    2. Graph: a CUDA-graph-captured collective has no external-correlation link on
       replay -- the kernel runs with fresh correlation ids and no host push -- so
       it is resolved by ``graph_node_id`` through ``metadata_resolver``. That is
       the same mechanism (and stack-managed registry) that resolves graph-node
       annotation NAMES: it already carries the stable executable-graph node ids
       across replays and owns their lifecycle, so no separate cache or reclamation
       is needed here.

    The blob rides as ``event["metadata"]`` (the producer's opaque string -- the
    consumer parses it); events without a match are left untouched. No-op when
    neither route is available, so the non-collective path pays nothing."""
    if not external_metadata and metadata_resolver is None:
        return
    # Scope to ids we actually have a blob for: an op may carry one
    # EXTERNAL_CORRELATION record per active kind (e.g. a tracer region annotation
    # too), and only the comms collective ids are in external_metadata -- so this
    # both picks the right record and avoids a region id shadowing the collective.
    corr_to_ext = (
        {
            e["correlation_id"]: e["external_id"]
            for e in ext_events
            if e["external_id"] in external_metadata
        }
        if external_metadata
        else {}
    )
    for ev in timed:
        if external_metadata:
            ext_id = corr_to_ext.get(ev.get("correlation_id"))
            if ext_id is not None:
                blob = external_metadata.get(ext_id)
                if blob is not None:
                    ev["metadata"] = blob
                    continue  # eager hit; don't also consult the graph resolver
        if metadata_resolver is not None:
            node = ev.get("graph_node_id") or 0
            if node:
                blob = metadata_resolver(node)
                if blob is not None:
                    ev["metadata"] = blob


def _kernel_events(cols, convert_time, resolver):
    events = []
    for i in range(_col_len(cols)):
        graph_node_id = int(cols[Kernel.GRAPH_NODE_ID.id][i])
        correlation_id = int(cols[Kernel.CORRELATION_ID.id][i])
        events.append(
            {
                "kind": "kernel",
                "device_id": int(cols[Kernel.DEVICE_ID.id][i]),
                "context_id": int(cols[Kernel.CONTEXT_ID.id][i]),
                "stream_id": int(cols[Kernel.STREAM_ID.id][i]),
                "correlation_id": correlation_id,
                "graph_node_id": graph_node_id,
                "graph_id": int(cols[Kernel.GRAPH_ID.id][i]),
                "start_ns": convert_time(int(cols[Kernel.START.id][i])),
                "end_ns": convert_time(int(cols[Kernel.END.id][i])),
                "annotation": resolver(
                    graph_node_id, ActivityKind.CONCURRENT_KERNEL, correlation_id
                ),
                "name": _demangle_symbol(cols[Kernel.NAME.id][i]),
                # Launch config (kineto-parity fields): grid/block dims, registers, shared
                # memory and priority -- per-kernel info not recoverable elsewhere for eager.
                "grid": [
                    int(cols[Kernel.GRID_X.id][i]),
                    int(cols[Kernel.GRID_Y.id][i]),
                    int(cols[Kernel.GRID_Z.id][i]),
                ],
                "block": [
                    int(cols[Kernel.BLOCK_X.id][i]),
                    int(cols[Kernel.BLOCK_Y.id][i]),
                    int(cols[Kernel.BLOCK_Z.id][i]),
                ],
                "registers_per_thread": int(cols[Kernel.REGISTERS_PER_THREAD.id][i]),
                "static_shared_memory": int(cols[Kernel.STATIC_SHARED_MEMORY.id][i]),
                "dynamic_shared_memory": int(cols[Kernel.DYNAMIC_SHARED_MEMORY.id][i]),
                "priority": int(cols[Kernel.LAUNCH_PRIORITY.id][i]),
                # queued is CUPTI's command-buffer enqueue time (needs the
                # subscriber latency-timestamp attr); 0 when unavailable.
                "queued": convert_time(int(cols[Kernel.QUEUED.id][i])),
                "channel": int(cols[Kernel.CHANNEL_ID.id][i]),
                "channel_type": int(cols[Kernel.CHANNEL_TYPE.id][i]),
            }
        )
    return events


def _memcpy_events(cols, convert_time, resolver):
    events = []
    for i in range(_col_len(cols)):
        graph_node_id = int(cols[Memcpy.GRAPH_NODE_ID.id][i])
        correlation_id = int(cols[Memcpy.CORRELATION_ID.id][i])
        events.append(
            {
                "kind": "gpu_memcpy",
                "device_id": int(cols[Memcpy.DEVICE_ID.id][i]),
                "context_id": int(cols[Memcpy.CONTEXT_ID.id][i]),
                "stream_id": int(cols[Memcpy.STREAM_ID.id][i]),
                "correlation_id": correlation_id,
                "graph_node_id": graph_node_id,
                "graph_id": int(cols[Memcpy.GRAPH_ID.id][i]),
                "start_ns": convert_time(int(cols[Memcpy.START.id][i])),
                "end_ns": convert_time(int(cols[Memcpy.END.id][i])),
                "bytes": int(cols[Memcpy.BYTES.id][i]),
                "copy_kind": int(cols[Memcpy.COPY_KIND.id][i]),
                "src_kind": int(cols[Memcpy.SRC_KIND.id][i]),
                "dst_kind": int(cols[Memcpy.DST_KIND.id][i]),
                "flags": int(cols[Memcpy.FLAGS.id][i]),
                "annotation": resolver(
                    graph_node_id, ActivityKind.MEMCPY, correlation_id
                ),
                "name": "Memcpy",
            }
        )
    return events


def _memset_events(cols, convert_time, resolver):
    events = []
    for i in range(_col_len(cols)):
        graph_node_id = int(cols[Memset.GRAPH_NODE_ID.id][i])
        correlation_id = int(cols[Memset.CORRELATION_ID.id][i])
        events.append(
            {
                "kind": "gpu_memset",
                "device_id": int(cols[Memset.DEVICE_ID.id][i]),
                "context_id": int(cols[Memset.CONTEXT_ID.id][i]),
                "stream_id": int(cols[Memset.STREAM_ID.id][i]),
                "correlation_id": correlation_id,
                "graph_node_id": graph_node_id,
                "graph_id": int(cols[Memset.GRAPH_ID.id][i]),
                "start_ns": convert_time(int(cols[Memset.START.id][i])),
                "end_ns": convert_time(int(cols[Memset.END.id][i])),
                "bytes": int(cols[Memset.BYTES.id][i]),
                "value": int(cols[Memset.VALUE.id][i]),
                "memory_kind": int(cols[Memset.MEMORY_KIND.id][i]),
                "flags": int(cols[Memset.FLAGS.id][i]),
                "annotation": resolver(
                    graph_node_id, ActivityKind.MEMSET, correlation_id
                ),
                "name": "Memset",
            }
        )
    return events


def _api_events(kind_name):
    def build(cols, convert_time, resolver):
        del resolver
        events = []
        for i in range(_col_len(cols)):
            cbid = int(cols[Api.CBID.id][i])
            events.append(
                {
                    "kind": kind_name,
                    "cbid": cbid,
                    "start_ns": convert_time(int(cols[Api.START.id][i])),
                    "end_ns": convert_time(int(cols[Api.END.id][i])),
                    "process_id": int(cols[Api.PROCESS_ID.id][i]),
                    "thread_id": int(cols[Api.THREAD_ID.id][i]),
                    "correlation_id": int(cols[Api.CORRELATION_ID.id][i]),
                    "name": f"cbid_{cbid}",
                }
            )
        return events

    return build


def _external_correlation_events(cols, convert_time, resolver):
    del convert_time, resolver
    events = []
    for i in range(_col_len(cols)):
        events.append(
            {
                "kind": "external_correlation",
                "external_kind": int(cols[ExternalCorrelation.EXTERNAL_KIND.id][i]),
                "external_id": int(cols[ExternalCorrelation.EXTERNAL_ID.id][i]),
                "correlation_id": int(cols[ExternalCorrelation.CORRELATION_ID.id][i]),
                "name": "external_correlation",
            }
        )
    return events


def _overhead_events(cols, convert_time, resolver):
    del resolver
    events = []
    for i in range(_col_len(cols)):
        overhead_kind = int(cols[Overhead.OVERHEAD_KIND.id][i])
        events.append(
            {
                "kind": "overhead",
                "object_id": 0,
                "start_ns": convert_time(int(cols[Overhead.START.id][i])),
                "end_ns": convert_time(int(cols[Overhead.END.id][i])),
                "correlation_id": int(cols[Overhead.CORRELATION_ID.id][i]),
                "name": OVERHEAD_KIND_NAMES.get(
                    overhead_kind, f"overhead_{overhead_kind}"
                ),
            }
        )
    return events


def _sync_events(cols, convert_time, resolver):
    del resolver
    events = []
    for i in range(_col_len(cols)):
        events.append(
            {
                "kind": "cuda_sync",
                "sync_type": int(cols[Sync.TYPE.id][i]),
                "start_ns": convert_time(int(cols[Sync.START.id][i])),
                "end_ns": convert_time(int(cols[Sync.END.id][i])),
                "context_id": int(cols[Sync.CONTEXT_ID.id][i]),
                "stream_id": int(cols[Sync.STREAM_ID.id][i]),
                "correlation_id": int(cols[Sync.CORRELATION_ID.id][i]),
                "cuda_event_id": int(cols[Sync.CUDA_EVENT_ID.id][i]),
                "cuda_event_sync_id": int(cols[Sync.CUDA_EVENT_SYNC_ID.id][i]),
                "name": "cuda_sync",
            }
        )
    return events


def _cuda_event_events(cols, convert_time, resolver):
    # No start_ns -> routed to the untimestamped join-input buffer (like
    # external_correlation). Resolves a Sync's cuda_event_sync_id to the cudaEventRecord
    # correlation id for the wait_on join in monitor_trace.
    del convert_time, resolver
    events = []
    for i in range(_col_len(cols)):
        events.append(
            {
                "kind": "cuda_event",
                "cuda_event_sync_id": int(cols[CudaEvent.CUDA_EVENT_SYNC_ID.id][i]),
                "correlation_id": int(cols[CudaEvent.CORRELATION_ID.id][i]),
                "device_id": int(cols[CudaEvent.DEVICE_ID.id][i]),
                "context_id": int(cols[CudaEvent.CONTEXT_ID.id][i]),
                "stream_id": int(cols[CudaEvent.STREAM_ID.id][i]),
                "event_id": int(cols[CudaEvent.EVENT_ID.id][i]),
                "name": "cuda_event",
            }
        )
    return events


_BUILDERS: dict[int, Callable[..., list[dict[str, Any]]]] = {
    ActivityKind.CONCURRENT_KERNEL: _kernel_events,
    ActivityKind.MEMCPY: _memcpy_events,
    ActivityKind.MEMSET: _memset_events,
    ActivityKind.RUNTIME: _api_events("cuda_runtime"),
    ActivityKind.DRIVER: _api_events("cuda_driver"),
    ActivityKind.EXTERNAL_CORRELATION: _external_correlation_events,
    ActivityKind.OVERHEAD: _overhead_events,
    ActivityKind.SYNCHRONIZATION: _sync_events,
    ActivityKind.CUDA_EVENT: _cuda_event_events,
}


# The active ProfilerObserver (if any) that record_function user annotations route
# to. The CUPTI external-correlation push is global (the monitor owns the stack);
# this just names which observer records the per-push id->name metadata. Set by the
# torch.profiler backend around a profiling session. This is a ProfilerObserver
# concern, not the monitor engine's, so it lives here.
_active_observer: ProfilerObserver | None = None


def set_active_profiler_observer(observer: ProfilerObserver | None) -> None:
    """Set (or clear, with None) the observer that push_user_annotation routes to."""
    global _active_observer
    _active_observer = observer


def push_user_annotation(name: str) -> int | None:
    """Push a record_function user annotation onto the active ProfilerObserver (if
    any). No-op returning None when no observer is active."""
    observer = _active_observer
    return observer.push_annotation(name) if observer is not None else None


def pop_user_annotation() -> int | None:
    """Pop the most recent user annotation off the active ProfilerObserver."""
    observer = _active_observer
    return observer.pop_annotation() if observer is not None else None
