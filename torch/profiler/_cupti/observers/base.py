# mypy: allow-untyped-defs
"""Base class for CUPTI activity-monitor observers.

An observer registers the activity kinds it wants with the shared CUPTI monitor and, on
the monitor's worker thread, is handed the demuxed columns (``{ActivityKind: {field_id:
column}}``) sliced to its selection -- what it does with them is the subclass's
``_on_activities`` hook. This base handles registration, availability, teardown, the
clock passthroughs, and the user-annotation push/pop for naming regions.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


# (graph_node_id, activity_kind, correlation_id) -> annotation, or None. The graph
# naming mechanism shared by observers: map a CUDA-graph node id to its registered
# region name (survives graph replay, needs no extra record kinds).
AnnotationResolver = Callable[[int, int, int], "Any | None"]


def default_graph_annotation_resolver(
    graph_node_id: int, activity_kind: int, correlation_id: int
) -> Any | None:
    """Default resolver: map a CUDA-graph node id to its registered annotation."""
    del activity_kind, correlation_id
    if graph_node_id == 0:
        return None
    try:
        from torch.cuda._graph_annotations import get_kernel_annotations

        annotations = get_kernel_annotations()
    except Exception:
        return None
    return annotations.get(graph_node_id)


@dataclass(frozen=True)
class ObserverAnnotationSettings:
    """How an observer attributes activity to named regions. With neither ``graph``
    nor ``eager`` set (the default), no naming is applied and every span falls into
    the ``""`` bucket in :meth:`NodeTimerObserver.drain_annotated`.

    - ``graph`` -- enable graph-node naming (``graph_node_id -> name``, vectorized,
      no extra record kinds, survives graph replay). Uses
      ``custom_graph_annotation_resolver`` if given, else the CUDA-graph annotation
      registry.
    - ``eager`` -- enable eager ``record_function`` naming via the
      ``correlation_id -> external_id -> name`` external-correlation join. Folds
      EXTERNAL_CORRELATION + RUNTIME into the selection (CUPTI only emits the former
      when the latter is enabled), which drops decode onto the slower per-record walk.
    - ``custom_graph_annotation_resolver`` -- override the graph resolver (only used
      when ``graph`` is set; falls back to the default registry resolver if omitted).
    """

    eager: bool = False
    graph: bool = False
    custom_graph_annotation_resolver: AnnotationResolver | None = None


class CuptiMonitorObserver:
    """Base for observers backed by the shared CUPTI monitor.

    Subclasses set up state, then call ``super().__init__(activities)`` with the kinds they
    want (a set, or a field map ``{kind: field ids | "all"}``) -- registration last, so
    state is ready before the worker thread delivers buffers. They implement
    ``_on_activities(columns)`` (``{ActivityKind: {field_id: column}}`` sliced to their
    selection) and typically a ``drain()``.

    Observers can also bracket regions with ``push_annotation``/``pop_annotation`` (or
    ``annotate``): each push registers a global external-correlation id mapped here to a
    name, attributing activity until the pop via ``correlation_id -> external_id -> name``
    (eager only -- external ids don't survive graph capture; under graphs use
    ``graph_node_id``)."""

    def __init__(
        self,
        activities: Any,
        *,
        annotations: ObserverAnnotationSettings | None = None,
    ) -> None:
        # Region naming (see ObserverAnnotationSettings): a graph-node resolver and/or the
        # eager external-correlation join. Eager folds extra record kinds in -> per-record
        # walk; graph naming is just a resolver, staying on the vectorized path.
        if annotations is None:
            self._resolver: AnnotationResolver | None = None
            self._eager = False
        else:
            self._eager = annotations.eager
            self._resolver = (
                annotations.custom_graph_annotation_resolver
                or default_graph_annotation_resolver
                if annotations.graph
                else None
            )
        if self._eager:
            activities = self._with_eager_fields(activities)
        # frozenset of requested kinds (a field map collapses to keys) for the observer's
        # own "is this kind mine?" checks; the full request goes to the monitor singleton.
        self._activities: frozenset[int] = frozenset(activities)
        self._monitor: Any = None
        self._obs = None
        # external_id -> annotation name for the monitor's global pushes. Guarded by
        # _ann_lock (push on the caller's thread; a drain may read/reset from another).
        self._ann_lock = threading.Lock()
        self._ext_names: dict[int, str] = {}
        # Degrade gracefully (available == False) if the monitor can't be reached or
        # registration fails (CUPTI subscribe rejected, libcupti lacks v2): the profiler
        # must not crash because the optional monitor couldn't start.
        try:
            from torch.profiler._cupti.monitor import instance

            self._monitor = instance()
            self._obs = self._monitor.register(activities, self._on_activities)
        except Exception:
            self._obs = None

    @property
    def available(self) -> bool:
        """True when the monitor was available and this observer registered."""
        return self._obs is not None

    def _on_activities(self, columns: dict[Any, dict[int, Any]]) -> None:
        """Worker-thread hook: ``{ActivityKind: {field_id: column}}`` demuxed by the
        monitor and sliced to this observer's selection. Implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def _with_eager_fields(activities: Any) -> dict[int, set[int]]:
        """Augment a field map for the eager join: add each kind's CORRELATION_ID plus the
        EXTERNAL_CORRELATION and RUNTIME records (CUPTI only emits the former when RUNTIME
        is enabled; RUNTIME is just the carrier). Expects a ``{kind: fields}`` map."""
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.records import CORRELATION_FIELD, ExternalCorrelation

        aug: dict[int, set[int]] = {}
        for kind, sel in dict(activities).items():
            k = int(kind)
            fields = {int(f) for f in sel}
            if k in CORRELATION_FIELD:
                fields.add(CORRELATION_FIELD[k])
            aug[k] = fields
        aug[int(ActivityKind.EXTERNAL_CORRELATION)] = {
            int(ExternalCorrelation.EXTERNAL_ID),
            int(ExternalCorrelation.CORRELATION_ID),
        }
        aug[int(ActivityKind.RUNTIME)] = {CORRELATION_FIELD[int(ActivityKind.RUNTIME)]}
        return aug

    # --- user annotations (external-correlation) ---------------------------

    def push_annotation(self, name: str) -> int | None:
        """Push a global external-correlation id (mapped here to ``name``) so activity
        until the pop is attributed via ``correlation_id -> external_id -> name``. Eager
        only. No-op returning None when the monitor is unavailable."""
        if not self.available or self._monitor is None:
            return None
        ext_id = self._monitor.push_external_correlation_id()
        if ext_id is not None:
            with self._ann_lock:
                self._ext_names[ext_id] = name
        return ext_id

    def pop_annotation(self) -> int | None:
        """Pop the most recent external-correlation id (balances push_annotation)."""
        if not self.available or self._monitor is None:
            return None
        return self._monitor.pop_external_correlation_id()

    @contextlib.contextmanager
    def annotate(self, name: str) -> Iterator[int | None]:
        """Context-manager form of push_annotation/pop_annotation."""
        ext_id = self.push_annotation(name)
        try:
            yield ext_id
        finally:
            self.pop_annotation()

    def annotation_names(self, *, reset: bool = False) -> dict[int, str]:
        """Snapshot of the ``external_id -> name`` map pushed so far; pass
        ``reset=True`` to also clear it (e.g. when closing a window)."""
        with self._ann_lock:
            snapshot = dict(self._ext_names)
            if reset:
                self._ext_names = {}
        return snapshot

    # --- clock passthroughs -------------------------------------------------

    def now_ns(self) -> int:
        """Current time on the same unix-epoch clock as record timestamps --
        passthrough to the monitor."""
        return self._monitor.now_unix_ns() if self._monitor is not None else 0

    def now_native_ns(self) -> int:
        """Current value of CUPTI's native record clock -- the unconverted timebase of
        decoded record START/END. Use this to stamp a window boundary compared against
        raw record timestamps (see NodeTimerObserver bucketing). 0 if unavailable."""
        return self._monitor.now_native_ns() if self._monitor is not None else 0

    def convert_time(self, value: int) -> int:
        """Convert a CUPTI-clock timestamp to unix-epoch ns -- passthrough to the
        monitor (identity if clock alignment is unavailable)."""
        return self._monitor.convert_time(value) if self._monitor is not None else value

    def convert_time_array(self, values: Any) -> Any:
        """Vectorized :meth:`convert_time` over a whole column -- passthrough to the
        monitor (identity if clock alignment is unavailable)."""
        if self._monitor is None:
            return values
        return self._monitor.convert_time_array(values)

    def close(self) -> None:
        """Unregister from the monitor. Idempotent."""
        if self._obs is not None and self._monitor is not None:
            self._monitor.unregister(self._obs)
            self._obs = None
