# mypy: allow-untyped-defs
"""Shared window-finalization machinery for CUPTI monitor observers.

An observer that publishes per *window* -- a span of activity ended by a boundary
(a training step for the SubgraphTimer, a profiling window for the
ProfilerObserver) -- stamps each boundary in CUPTI's native record clock and
finalizes that window only once the records it covers have been *naturally*
delivered. No device sync touches the measured timeline: CUPTI delivers buffers
in fill order, so once a delivered record starts at/after a boundary, every
record before that boundary is already in hand ("covered").

This mixin owns the boundary queue, the background poll thread, the
cover-detection loop, and teardown. The subclass supplies what differs between
consumers (their buffer shape and bucketing):

  * ``_collect_delivered(sync)`` -- pull the records the monitor has delivered
    into the subclass's own buffer. ``sync`` is True only at teardown, where a
    synchronous flush is harmless (collection is over, nothing left to perturb).
  * ``_window_watermark_ns()`` -- the max delivered record start (native clock);
    a boundary at/below this is covered.
  * ``_finalize_window(window_id, boundary_ns)`` -- select that window's records,
    emit/publish them, and drop them from the buffer.

The clock source is ``now_native_ns()`` from :class:`CuptiMonitorObserver`, so
boundaries are in the same unconverted timebase as the records' START/END.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable  # noqa: TC003


logger = logging.getLogger(__name__)


class _PollThread(threading.Thread):
    """Daemon thread that calls ``fn`` every ``interval_s`` until stopped."""

    def __init__(self, fn: Callable[[], None], interval_s: float, name: str) -> None:
        super().__init__(name=name, daemon=True)
        self._fn = fn
        self._interval_s = interval_s
        # NB: not `_stop` -- that shadows threading.Thread's internal `_stop()`.
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self._fn()
            except Exception:
                # A finalize failure must not kill the poller -- log and keep going so
                # later windows still publish.
                logger.exception("CUPTI window poll cycle failed")

    def stop(self) -> None:
        self._stop_event.set()


class WindowFinalizerMixin:
    """Boundary queue + background poller + cover-and-finalize loop. See the module
    docstring. Subclasses call :meth:`_init_windowing` in ``__init__`` and implement
    ``_collect_delivered`` / ``_window_watermark_ns`` / ``_finalize_window``."""

    def _init_windowing(
        self, *, poll_interval_ms: int = 50, thread_name: str = "cupti-window-poller"
    ) -> None:
        self._win_lock = threading.Lock()
        # (window_id, boundary_ns) per ended window, oldest first; the poller pops one
        # once its boundary is covered by delivered records.
        self._boundaries: deque[tuple[int, int]] = deque()
        self._next_window_id = 0
        self._poll_interval_s = max(1, int(poll_interval_ms)) / 1000.0
        self._poll_thread: _PollThread | None = None
        self._poll_thread_name = thread_name

    def mark_boundary(self) -> int:
        """Stamp the current native record clock as a window boundary and queue it for
        finalization once delivered records cover it. Returns the window id. Lazily
        starts the background poller on the first call. Does NOT flush -- the caller
        decides whether to nudge a (non-forced) flush so the poller sees the records."""
        boundary = self._boundary_clock_ns()
        with self._win_lock:
            window_id = self._next_window_id
            self._next_window_id += 1
            self._boundaries.append((window_id, boundary))
            if self._poll_thread is None:
                self._poll_thread = _PollThread(
                    self._poll_once, self._poll_interval_s, self._poll_thread_name
                )
                self._poll_thread.start()
        return window_id

    def _poll_once(self, drain_all: bool = False, *, sync: bool = False) -> None:
        """One poll cycle: collect delivered records, then finalize every queued
        boundary the delivered records now cover. ``drain_all`` (teardown) finalizes
        every remaining boundary regardless of watermark. ``sync`` sync-flushes CUPTI
        in the collect (only safe on the same thread as any other monitor flusher);
        leave it False to rely on naturally-delivered records."""
        self._collect_delivered(sync=sync)
        watermark = self._window_watermark_ns()
        ready: list[tuple[int, int]] = []
        with self._win_lock:
            while self._boundaries and (
                drain_all or self._boundaries[0][1] <= watermark
            ):
                ready.append(self._boundaries.popleft())
        for window_id, boundary in ready:
            self._finalize_window(window_id, boundary)

    def _stop_windowing(self, *, sync: bool = True) -> None:
        """Stop the poller and finalize whatever remains. Idempotent. ``sync`` (the
        default) sync-flushes CUPTI in the final drain -- correct when this runs on
        the same thread as any other monitor flusher (e.g. teardown on the training
        thread). Pass ``sync=False`` to finalize only what's already been naturally
        delivered, when running off-thread where a flush would race the monitor."""
        thread = self._poll_thread
        if thread is not None:
            thread.stop()
            thread.join(timeout=5.0)
            self._poll_thread = None
        self._poll_once(drain_all=True, sync=sync)

    # --- subclass hooks ----------------------------------------------------

    def _boundary_clock_ns(self) -> int:
        """Clock a boundary is stamped in -- must match the clock
        :meth:`_window_watermark_ns` reports. Default: CUPTI's native record clock
        (matches raw record START/END). Override to a converted clock if the subclass
        stores records converted -- then convert the boundary the same way, so the
        comparison stays order-equivalent (convert_time is monotonic)."""
        return self.now_native_ns()

    def now_native_ns(self) -> int:
        raise NotImplementedError

    def _collect_delivered(self, *, sync: bool) -> None:
        raise NotImplementedError

    def _window_watermark_ns(self) -> int:
        raise NotImplementedError

    def _finalize_window(self, window_id: int, boundary_ns: int) -> None:
        raise NotImplementedError
