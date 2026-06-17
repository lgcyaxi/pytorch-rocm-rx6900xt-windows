# Owner(s): ["oncall: profiler"]
"""Tests for the CUPTI activity monitor and its v2/user-defined-record codec.

These are the non-profiler-specific monitor tests: the ``records`` field schema +
``decode`` codec (pure, no CUDA), and collection through ``CuptiMonitor``
directly (CUDA). Tests that exercise the monitor *through* ``torch.profiler.profile``
(trace shape, op/kernel-name parity, record_shapes, multithread, ...) live in
``test_profiler.py``.
"""

import ctypes
import json
import threading
import time
import unittest

import torch
from torch.profiler._cupti.observers.windowing import WindowFinalizerMixin
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.utils._import_utils import _check_module_exists


TEST_CUDA = torch.cuda.is_available()
# cupti-python is pip-installable on ROCm hosts too, but CUPTI itself is a no-op
# there, so gate the monitor tests off ROCm as well.
TEST_CUPTI_PYTHON = _check_module_exists("cupti") and not TEST_WITH_ROCM


def _cupti_version() -> int:
    if not TEST_CUPTI_PYTHON:
        return 0
    try:
        from torch.profiler._cupti.cupti_python import pylibcupti

        return pylibcupti().get_version()
    except Exception:
        return 0


# The CUPTI monitor needs libcupti >= 13.3: it uses the v2 user-defined-record API
# (>= 13.2) AND decodes against pBufferCompleteInfo->ppRecordLayouts (CUPTI's own
# per-kind record layout), which 13.2 leaves null. So a single >= 13.3 gate covers
# the whole monitor (it implies v2).
TEST_CUPTI_V13_3 = TEST_CUPTI_PYTHON and _cupti_version() >= 130300


@unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
class TestCuptiRecords(TestCase):
    """Pure tests of the field schema + columnar codec -- no CUDA."""

    def test_field_catalog(self):
        # A Field carries (id, string); int(field) is its id. The per-kind catalogs
        # define the supported field ids; FIELD_REGISTRY and STRING_FIELDS derive
        # from them. No byte sizes -- those come from CUPTI's captured layout.
        from torch.profiler._cupti.cupti_python import ActivityKind
        from torch.profiler._cupti.records import FIELD_REGISTRY, Kernel, STRING_FIELDS

        self.assertEqual(int(Kernel.START), 7)
        self.assertEqual(Kernel.NAME.id, 24)
        self.assertTrue(Kernel.NAME.string)
        self.assertFalse(Kernel.START.string)
        kernel = int(ActivityKind.CONCURRENT_KERNEL)
        self.assertEqual(
            FIELD_REGISTRY[kernel],
            frozenset(
                {
                    0,
                    4,
                    7,
                    8,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    22,
                    24,
                    31,
                    33,
                    45,
                    25,
                    35,
                    36,
                }
            ),
        )
        self.assertEqual(STRING_FIELDS[kernel], frozenset({24}))

    def test_decode_columns(self):
        # records.decode demuxes a whole buffer (possibly interleaving
        # kinds) into {kind: {field_id: column}} against CUPTI's captured layout list
        # [(kind, record_size, [(field_id, offset, size), ...])] -- every field in
        # the layout. Three strategies: single kind (stride), uniform size (stride +
        # dispatch by KIND), variable size (per-record walk); plus a bounds guard
        # dropping a trailing record that overruns valid_size, and const char* fields
        # dereferenced in place. Synthetic layouts + buffers, no CUDA.
        import numpy as np

        from torch.profiler._cupti.cupti_python import ActivityKind
        from torch.profiler._cupti.monitor import CuptiMonitorBuffer

        kernel = int(ActivityKind.CONCURRENT_KERNEL)
        memcpy = int(ActivityKind.MEMCPY)

        def decode(addr, valid_size, record_layouts):
            # Decode a synthetic buffer. _returned=True so the RAII __del__ doesn't
            # hand this (non-pool) pointer back to the native buffer pool.
            buf = CuptiMonitorBuffer((addr, valid_size, 0, 0, record_layouts))
            buf._returned = True
            return buf.decode()

        def build(records, layouts):
            # records: [(kind, {field_id: int | bytes})]; a bytes value is written as
            # a real C string whose pointer is packed. layouts: {kind: (rsz, [(fid,
            # off, sz), ...])}. Returns (keepalive, addr, n).
            keep: list = []
            blob = bytearray()
            for kind, vals in records:
                rsz, fields = layouts[kind]
                rec = bytearray(rsz)
                values = {0: kind, **vals}
                for fid, off, sz in fields:
                    if fid not in values:
                        continue
                    v = values[fid]
                    if isinstance(v, bytes):
                        s = ctypes.create_string_buffer(v)
                        keep.append(s)
                        v = ctypes.addressof(s)
                    rec[off : off + sz] = int(v).to_bytes(sz, "little")
                blob += rec
            buf = (ctypes.c_uint8 * max(len(blob), 1)).from_buffer_copy(bytes(blob))
            keep.append(buf)
            return keep, ctypes.addressof(buf), len(blob)

        def as_list(layouts):
            return [(k, rsz, fields) for k, (rsz, fields) in layouts.items()]

        # --- single kind: homogeneous stride ---
        ker = {kernel: (24, [(0, 0, 4), (7, 8, 8), (8, 16, 8)])}
        keep, addr, n = build(
            [(kernel, {7: 100, 8: 150}), (kernel, {7: 200, 8: 275})], ker
        )
        out = decode(addr, n, as_list(ker))
        np.testing.assert_array_equal(out[kernel][7], [100, 200])
        np.testing.assert_array_equal(out[kernel][8], [150, 275])
        # Columns are independent copies of the buffer: scribbling over the source
        # bytes must not change already-decoded columns. This is what lets the worker
        # return the buffer to the pool immediately after decode().
        ctypes.memset(addr, 0, n)
        np.testing.assert_array_equal(out[kernel][7], [100, 200])
        np.testing.assert_array_equal(out[kernel][8], [150, 275])

        # --- uniform size: stride + dispatch by KIND (both kinds 24B) ---
        uni = {
            kernel: (24, [(0, 0, 4), (7, 8, 8), (8, 16, 8)]),
            memcpy: (24, [(0, 0, 4), (5, 8, 8), (6, 16, 8)]),
        }
        keepu, addru, nu = build(
            [
                (kernel, {7: 10, 8: 11}),
                (memcpy, {6: 20, 5: 999}),
                (kernel, {7: 30, 8: 33}),
            ],
            uni,
        )
        outu = decode(addru, nu, as_list(uni))
        np.testing.assert_array_equal(outu[kernel][7], [10, 30])
        np.testing.assert_array_equal(outu[memcpy][6], [20])
        np.testing.assert_array_equal(outu[memcpy][5], [999])

        # --- variable size + bounds guard: per-record walk ---
        # kernel (24B) and memcpy (16B) differ -> walk. A third kernel record's KIND
        # header fits in valid_size but its body runs past it -> dropped.
        var = {
            kernel: (24, [(0, 0, 4), (7, 8, 8)]),
            memcpy: (16, [(0, 0, 4), (6, 8, 8)]),
        }
        keepv, addrv, _ = build(
            [(kernel, {7: 1}), (memcpy, {6: 3}), (kernel, {7: 4})], var
        )
        valid = 24 + 16 + 4  # third kernel: KIND header only, body cut off
        outv = decode(addrv, valid, as_list(var))
        np.testing.assert_array_equal(outv[kernel][7], [1])  # truncated 3rd dropped
        np.testing.assert_array_equal(outv[memcpy][6], [3])

        # --- const char* string field dereferenced in place (NAME id 24) ---
        ks = {kernel: (40, [(0, 0, 4), (7, 8, 8), (24, 32, 8)])}
        keeps, addrs, ns = build([(kernel, {7: 7, 24: b"my_kernel"})], ks)
        outs = decode(addrs, ns, as_list(ks))
        self.assertEqual(list(outs[kernel][24]), ["my_kernel"])
        np.testing.assert_array_equal(outs[kernel][7], [7])

    def test_monitor_normalize_activities(self):
        # A registration request resolves to (kinds, per-kind field selection): a
        # bare kind iterable means "all fields"; a field map selects fields, with
        # "all"/None expanding; *_FIELD_KIND (0) is always included.
        from torch.profiler._cupti.cupti_python import ActivityKind
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import FIELD_REGISTRY, Kernel

        m = CuptiMonitor()
        kernel = ActivityKind.CONCURRENT_KERNEL
        memcpy = ActivityKind.MEMCPY
        all_kernel = frozenset(FIELD_REGISTRY[kernel]) | {0}
        all_memcpy = frozenset(FIELD_REGISTRY[memcpy]) | {0}

        kinds, fields = m._normalize_activities([kernel, memcpy])
        self.assertEqual(kinds, frozenset({kernel, memcpy}))
        self.assertEqual(fields[kernel], all_kernel)
        self.assertEqual(fields[memcpy], all_memcpy)

        kinds, fields = m._normalize_activities({kernel: {Kernel.START}, memcpy: "all"})
        self.assertEqual(fields[kernel], frozenset({0, int(Kernel.START)}))
        self.assertEqual(fields[memcpy], all_memcpy)

    def test_monitor_buffer_size_from_env(self):
        # The per-buffer pool size is user-configurable: an explicit buffer_size
        # arg wins, otherwise TORCH_CUPTI_MONITOR_BUFFER_SIZE is honored, else the
        # 4 MiB default. No CUDA -- this only reads the constructor config.
        import unittest.mock

        from torch.profiler._cupti.monitor import _DEFAULT_BUFFER_SIZE, CuptiMonitor

        self.assertEqual(CuptiMonitor().buffer_size, _DEFAULT_BUFFER_SIZE)
        with unittest.mock.patch.dict(
            "os.environ", {"TORCH_CUPTI_MONITOR_BUFFER_SIZE": "1048576"}
        ):
            self.assertEqual(CuptiMonitor().buffer_size, 1048576)
            # An explicit arg overrides the env var.
            self.assertEqual(CuptiMonitor(buffer_size=2048).buffer_size, 2048)

    def test_monitor_external_correlation_not_started(self):
        # External-correlation push/pop are no-ops until the monitor is started (no
        # subscriber yet), returning None rather than touching CUPTI's global stack.
        from torch.profiler._cupti.monitor import CuptiMonitor

        m = CuptiMonitor()
        self.assertFalse(m._started)
        self.assertIsNone(m.push_external_correlation_id())
        self.assertIsNone(m.pop_external_correlation_id())

    def test_external_correlation_id_mirror(self):
        # The native per-thread mirror of CUPTI's external-correlation stack lets
        # the current id be read (CUPTI has push/pop but no peek). Pure: the mirror
        # is host-side, no CUDA/CUPTI.
        m = torch._C._profiler._cupti_monitor
        while m.current_external_id():  # drain residue from a prior test (same thread)
            m.note_external_pop()
        self.assertEqual(m.current_external_id(), 0)
        m.note_external_push(7)
        self.assertEqual(m.current_external_id(), 7)
        m.note_external_push(9)
        self.assertEqual(m.current_external_id(), 9)  # top == innermost push
        self.assertEqual(m.note_external_pop(), 9)
        self.assertEqual(m.current_external_id(), 7)
        self.assertEqual(m.note_external_pop(), 7)
        self.assertEqual(m.current_external_id(), 0)
        self.assertEqual(m.note_external_pop(), 0)  # empty -> 0

    def test_external_id_chain_and_gc(self):
        # The push-time active-id chain: with one kind CUPTI tags a kernel with only
        # the innermost id, so the monitor snapshots the full active stack per id and
        # exposes external_id_chain(innermost) -> (outermost..innermost) for consumers
        # to recover enclosing contexts. Popped ids' chains are GC'd one generation
        # late (their records arrive after the pop). Drive the state directly -- no
        # CUDA/CUPTI.
        from torch.profiler._cupti.monitor import CuptiMonitor

        m = CuptiMonitor()
        # region id 1 encloses collective id 2 (as push_external_correlation_id would
        # record snapshotting this thread's live stack).
        m._id_chains[1] = (1,)
        m._id_chains[2] = (1, 2)
        # The collective's innermost id -> the full active set (region + collective).
        self.assertEqual(m.external_id_chain(2), (1, 2))
        self.assertEqual(m.external_id_chain(1), (1,))
        # An id with no snapshot falls back to itself (already GC'd, or not ours).
        self.assertEqual(m.external_id_chain(9), (9,))
        # GC keeps a popped id's chain through one drain, then drops it; a still-active
        # id (never popped) is retained.
        m._chains_gc_pending = [2]
        m._gc_external_chains()  # 2: pending -> ready, still present
        self.assertIn(2, m._id_chains)
        m._gc_external_chains()  # ready dropped
        self.assertNotIn(2, m._id_chains)
        self.assertIn(1, m._id_chains)

    def test_metadata_store_roundtrip(self):
        # The CollTrace-replacement metadata store: producers put a JSON object keyed
        # by the CURRENT external-correlation id on this thread's stack (pushed around
        # the op), so the put takes no id. Repeated puts under the same id MERGE. It
        # drains alongside the decoded records (folded into drain_decoded's return) so
        # the metadata and the records it annotates come from one snapshot. Pure
        # host-side, no CUDA/CUPTI.
        m = torch._C._profiler._cupti_monitor
        m.drain_decoded()  # clear any residue
        # No id on the stack -> put is a no-op.
        m.metadata_put_external(json.dumps({"func": "AllReduce"}))
        self.assertEqual(m.drain_decoded(), ([], {}))
        # Pushed id keys the object; puts merge recursively (mirrors several
        # producers each contributing fields, incl. nested objects, for one op).
        m.note_external_push(42)
        m.metadata_put_external(json.dumps({"func": "AllReduce", "nested": {"a": 1}}))
        m.metadata_put_external(json.dumps({"count": 4096, "nested": {"b": 2}}))
        m.note_external_pop()
        groups, ext_meta = m.drain_decoded()
        self.assertEqual(groups, [])  # no decode activity
        self.assertEqual(set(ext_meta), {42})
        # Top-level union + nested objects combined (recursive merge).
        self.assertEqual(
            json.loads(ext_meta[42]),
            {"func": "AllReduce", "count": 4096, "nested": {"a": 1, "b": 2}},
        )
        # drained -> empty
        self.assertEqual(m.drain_decoded(), ([], {}))

    def test_attach_metadata_join(self):
        # The CollTrace-replacement join: a blob keyed by external_id is attached
        # onto the timed events whose correlation_id maps (via the window's
        # EXTERNAL_CORRELATION records -- all ours, the monitor is the sole pusher)
        # to that external_id. Pure host-side.
        from torch.profiler._cupti.observers.profiler import _attach_metadata

        timed = [
            {"correlation_id": 100},  # -> ext 42 -> blob
            {"correlation_id": 200},  # -> ext 43, no blob
            {"correlation_id": 300},  # no ext mapping
        ]
        ext_events = [
            {"external_id": 42, "correlation_id": 100},
            {"external_id": 43, "correlation_id": 200},
        ]
        blob = json.dumps({"func": "AllReduce", "count": 4096})
        _attach_metadata(timed, ext_events, {42: blob}, None)
        self.assertEqual(
            json.loads(timed[0]["metadata"]), {"func": "AllReduce", "count": 4096}
        )
        self.assertNotIn("metadata", timed[1])  # external id has no blob
        self.assertNotIn("metadata", timed[2])  # no correlation -> external mapping
        # No metadata + no resolver is a no-op (the non-collective path pays nothing).
        clean = [{"correlation_id": 100}]
        _attach_metadata(clean, ext_events, {}, None)
        self.assertNotIn("metadata", clean[0])

    def test_attach_metadata_graph_resolver(self):
        # CUDA-graph-captured collectives have no replay-time external correlation;
        # their blob is resolved by graph_node_id via the metadata resolver (the same
        # mechanism as graph annotation names). Pure host-side.
        from torch.profiler._cupti.observers.profiler import _attach_metadata

        blob = json.dumps({"func": "AllReduce"})
        registry = {7: blob}  # stack-managed graph_node_id -> blob
        resolver = registry.get

        # Replay event: stable node 7, fresh correlation id, no EXTERNAL_CORRELATION
        # -> the eager join misses but the resolver hits.
        replay = [{"correlation_id": 999, "graph_node_id": 7}]
        _attach_metadata(replay, [], {}, resolver)
        self.assertEqual(json.loads(replay[0]["metadata"]), {"func": "AllReduce"})

        # Unknown node -> untouched.
        other = [{"correlation_id": 1000, "graph_node_id": 8}]
        _attach_metadata(other, [], {}, resolver)
        self.assertNotIn("metadata", other[0])

        # Eager wins over the resolver when both could resolve (and avoids calling it).
        both = [{"correlation_id": 100, "graph_node_id": 7}]
        ext_events = [{"external_id": 42, "correlation_id": 100}]
        eager_blob = json.dumps({"func": "ReduceScatter"})
        _attach_metadata(both, ext_events, {42: eager_blob}, resolver)
        self.assertEqual(json.loads(both[0]["metadata"]), {"func": "ReduceScatter"})

    def test_metadata_store_explicit_id(self):
        # put_external(blob, external_id): an explicit non-zero id targets a specific
        # collective rather than the current pushed one -- the seam for a backend to
        # attach metadata outside the push window. external_id 0 (default) still keys
        # by the current id, and the two paths merge. Pure host-side.
        m = torch._C._profiler._cupti_monitor
        m.drain_decoded()  # clear any residue
        try:
            m.metadata_put_external(json.dumps({"backend": "x"}), 77)  # explicit id
        except TypeError:
            self.skipTest("metadata_put_external(blob, external_id) not built yet")
        # Default (0) path keys by the current pushed id and merges with the above.
        m.note_external_push(77)
        m.metadata_put_external(json.dumps({"extra": 1}))
        m.note_external_pop()
        _, ext_meta = m.drain_decoded()
        self.assertEqual(set(ext_meta), {77})
        self.assertEqual(json.loads(ext_meta[77]), {"backend": "x", "extra": 1})


@unittest.skipIf(not TEST_CUDA, "CUDA required")
class TestCuptiMonitorCUDA(TestCase):
    """Collection through CuptiMonitor directly (not via torch.profiler.profile)."""

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_fence_enables_sync_transiently(self):
        # flush(sync=True) fences at a SYNCHRONIZATION sync point, enabled only for
        # the fence (even when no observer requested it) and disabled again after.
        from torch.profiler._cupti.cupti_python import ActivityKind
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Kernel

        sync = int(ActivityKind.SYNCHRONIZATION)
        monitor = CuptiMonitor()
        obs = monitor.register(
            {ActivityKind.CONCURRENT_KERNEL: {Kernel.END}}, lambda c: None
        )
        self.addCleanup(monitor.unregister, obs)
        self.assertNotIn(sync, monitor._enabled)

        x = torch.randn(64, 64, device="cuda")
        (x @ x).relu().sum().item()
        start = time.time()
        monitor.flush(sync=True)
        self.assertLess(time.time() - start, 2.0)
        self.assertNotIn(sync, monitor._enabled)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_v2_columnar_collection(self):
        # End-to-end columnar collection: the monitor turns on a per-activity field
        # selection, decodes each buffer against CUPTI's captured layout, and hands
        # the observer the columns for its selection.
        from torch.profiler._cupti.cupti_python import ActivityKind, CuptiError
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Kernel

        kind = ActivityKind.CONCURRENT_KERNEL
        want = {kind: {Kernel.START, Kernel.END, Kernel.CORRELATION_ID, Kernel.NAME}}

        lock = threading.Lock()
        columns: list = []
        monitor = CuptiMonitor()

        def on_columns(cols):
            if kind in cols:
                with lock:
                    columns.append(cols[kind])

        try:
            obs = monitor.register(want, on_columns)
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        self.addCleanup(monitor.unregister, obs)
        self.assertIsNotNone(monitor._subscriber)

        x = torch.randn(256, 256, device="cuda")
        for _ in range(4):
            x = torch.relu(x @ x)
        x.sum().item()
        torch.cuda.synchronize()

        monitor.flush(sync=True)
        monitor.unregister(obs)

        total = sum(len(c[int(Kernel.START)]) for c in columns)
        self.assertGreater(total, 0)
        for c in columns:
            for fld in want[kind]:
                self.assertIn(int(fld), c)
            start = c[int(Kernel.START)]
            end = c[int(Kernel.END)]
            name = c[int(Kernel.NAME)]
            self.assertEqual(len(start), len(end))
            self.assertEqual(len(start), len(name))
            self.assertTrue(all(int(e) - int(s) >= 0 for s, e in zip(start, end)))
        self.assertTrue(any(len(n) > 0 for c in columns for n in c[int(Kernel.NAME)]))

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_singleton_flush_accessible(self):
        # A user can reach the process-wide monitor singleton through the public
        # accessors and flush it: instance() constructs/returns it, get_monitor()
        # hands back that same object, and flush(sync=True) on the singleton
        # delivers everything collected up to the call.
        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.cupti_python import ActivityKind, CuptiError
        from torch.profiler._cupti.records import Kernel

        kernel = ActivityKind.CONCURRENT_KERNEL
        lock = threading.Lock()
        columns: list = []

        def on_columns(cols):
            if kernel in cols:
                with lock:
                    columns.append(cols[kernel])

        mon = cupti_monitor.instance()
        self.assertIs(cupti_monitor.get_monitor(), mon)
        # Drop the singleton after the observer is torn down so the next instance()
        # caller gets a fresh monitor (cleanups run LIFO: unregister first).
        self.addCleanup(setattr, cupti_monitor, "_instance", None)
        try:
            obs = mon.register({kernel: {Kernel.START, Kernel.END}}, on_columns)
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        self.addCleanup(mon.unregister, obs)

        x = torch.randn(128, 128, device="cuda")
        for _ in range(3):
            x = torch.relu(x @ x)
        x.sum().item()
        torch.cuda.synchronize()

        # Flush via the singleton fetched from the public accessor, not the local
        # handle, to exercise the user-visible path.
        cupti_monitor.get_monitor().flush(sync=True)

        total = sum(len(c[int(Kernel.START)]) for c in columns)
        self.assertGreater(total, 0)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_multiple_observers(self):
        # The monitor is the multiplexer: it enables the union of fields on its one
        # subscriber, then hands each observer only the columns it selected. Two
        # observers on the same kind with disjoint selections each see only their own
        # slice (plus KIND id 0) and the same set of records.
        from torch.profiler._cupti.cupti_python import ActivityKind, CuptiError
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Kernel

        kernel = ActivityKind.CONCURRENT_KERNEL
        lock = threading.Lock()
        a_slices: list = []
        b_slices: list = []

        def collect(sink):
            def cb(cols):
                kc = cols.get(kernel)
                if kc:
                    with lock:
                        sink.append({fid: len(col) for fid, col in kc.items()})

            return cb

        monitor = CuptiMonitor()
        try:
            obs_a = monitor.register(
                {kernel: {Kernel.START, Kernel.END}}, collect(a_slices)
            )
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        obs_b = monitor.register(
            {kernel: {Kernel.CORRELATION_ID, Kernel.NAME}}, collect(b_slices)
        )
        self.addCleanup(monitor.unregister, obs_b)
        self.addCleanup(monitor.unregister, obs_a)
        self.assertGreaterEqual(
            set(monitor._enabled.get(int(kernel), frozenset())),
            {0, int(Kernel.START), int(Kernel.CORRELATION_ID)},
        )

        x = torch.randn(128, 128, device="cuda")
        for _ in range(3):
            x = torch.relu(x @ x)
        x.sum().item()
        torch.cuda.synchronize()
        monitor.flush(sync=True)

        self.assertTrue(a_slices)
        self.assertTrue(b_slices)
        a_fields = set().union(*(set(s) for s in a_slices))
        b_fields = set().union(*(set(s) for s in b_slices))
        self.assertLessEqual(a_fields, {0, int(Kernel.START), int(Kernel.END)})
        self.assertLessEqual(
            b_fields, {0, int(Kernel.CORRELATION_ID), int(Kernel.NAME)}
        )
        a_count = sum(s[int(Kernel.START)] for s in a_slices)
        b_count = sum(s[int(Kernel.CORRELATION_ID)] for s in b_slices)
        self.assertGreater(a_count, 0)
        self.assertEqual(a_count, b_count)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_node_timer_collects_kernel_spans(self):
        # NodeTimerObserver is the minimal timing consumer: it registers just the
        # kernel timing fields (START/END/GRAPH_NODE_ID) and drain() returns flat
        # (graph_node_id, start, end) numpy columns. Eager kernels share node 0.
        import numpy as np

        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver()
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")
        try:
            x = torch.randn(256, 256, device="cuda")
            for _ in range(4):
                x = torch.relu(x @ x)
            x.sum().item()
            torch.cuda.synchronize()
            # drain()'s own flush is plain/best-effort, so deterministically deliver
            # everything via the monitor's sync flush, then drain.
            obs._monitor.flush(sync=True)
            gnode, start, end = obs.drain()
            # drain() resets; with no new work the next drain is empty.
            _, start2, _ = obs.drain()
        finally:
            obs.close()

        self.assertGreater(len(start), 0)
        self.assertEqual(len(start), len(end))
        self.assertEqual(len(gnode), len(start))
        self.assertEqual(len(start2), 0)
        # durations are non-negative, and the columns have the documented dtypes.
        self.assertTrue(bool((end >= start).all()))
        self.assertEqual(gnode.dtype, np.dtype("<u8"))
        self.assertEqual(start.dtype, np.dtype("<i8"))

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_node_timer_drain_annotated_eager(self):
        # With eager=True, eager kernels bracketed by annotate(name) resolve to that
        # region via the correlation_id -> external_id -> name join, and
        # drain_annotated() returns {name: [(start, end), ...]}.
        from torch.profiler._cupti.observers.base import ObserverAnnotationSettings
        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver(annotations=ObserverAnnotationSettings(eager=True))
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")
        try:
            x = torch.randn(128, 128, device="cuda")
            with obs.annotate("regionA"):
                for _ in range(3):
                    x = torch.relu(x @ x)
                x.sum().item()
                torch.cuda.synchronize()
            # drain_annotated()'s flush is best-effort; deliver deterministically first.
            obs._monitor.flush(sync=True)
            spans = obs.drain_annotated()
        finally:
            obs.close()

        self.assertIn("regionA", spans)
        self.assertGreater(len(spans["regionA"]), 0)
        for start_ns, end_ns in spans["regionA"]:
            self.assertGreaterEqual(end_ns, start_ns)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_node_timer_nested_resolves_to_enclosing_region(self):
        # A kernel launched under an inner (unnamed) external-correlation push -- as a
        # collective would push, leaf and innermost -- nested inside a named region
        # still resolves to that region. The single-kind record carries only the inner
        # id; node_timer recovers the enclosing region from the monitor's active-id
        # chain at dispatch.
        from torch.profiler._cupti.observers.base import ObserverAnnotationSettings
        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver(annotations=ObserverAnnotationSettings(eager=True))
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")
        mon = obs._monitor
        try:
            x = torch.randn(128, 128, device="cuda")
            with obs.annotate("outer"):
                # Inner unnamed push (mimics a nested collective tag): innermost id,
                # not a named region -> the kernels must fall back to "outer".
                mon.push_external_correlation_id()
                try:
                    for _ in range(3):
                        x = torch.relu(x @ x)
                    x.sum().item()
                    torch.cuda.synchronize()
                finally:
                    mon.pop_external_correlation_id()
            mon.flush(sync=True)
            spans = obs.drain_annotated()
        finally:
            obs.close()

        self.assertIn("outer", spans)
        self.assertGreater(len(spans["outer"]), 0)
        for start_ns, end_ns in spans["outer"]:
            self.assertGreaterEqual(end_ns, start_ns)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_node_timer_drain_annotated_unnamed_bucket(self):
        # With no annotations configured, drain_annotated() doesn't drop or raise --
        # every span lands in the "" bucket.
        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver()
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")
        try:
            x = torch.randn(128, 128, device="cuda")
            for _ in range(3):
                x = torch.relu(x @ x)
            x.sum().item()
            torch.cuda.synchronize()
            obs._monitor.flush(sync=True)
            spans = obs.drain_annotated()
        finally:
            obs.close()

        self.assertEqual(set(spans), {""})
        self.assertGreater(len(spans[""]), 0)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_small_buffer_chain_no_deadlock(self):
        # A tiny buffer maximizes buffer-completion churn: the decode worker runs
        # constantly and the foreground + background (20ms) flushes drain, dispatch,
        # and GC the chain back-to-back -- where a flush/decode/lock deadlock would
        # surface. Drive chain push/pop (incl. nested) through it; the test completing
        # is the no-deadlock assertion (a regression would hang here), and we also
        # check the pipeline actually delivered. stop() (final sync flush + decoder
        # teardown) on unregister must not hang either.
        from torch.profiler._cupti.cupti_python import ActivityKind, CuptiError
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Api, ExternalCorrelation, Kernel

        counts: dict[int, int] = {}

        def cb(cols):
            for k, c in cols.items():
                counts[int(k)] = counts.get(int(k), 0) + (
                    len(next(iter(c.values()))) if c else 0
                )

        m = CuptiMonitor(buffer_size=1024, flush_period_s=0.02)
        want = {
            ActivityKind.CONCURRENT_KERNEL: {
                Kernel.START,
                Kernel.END,
                Kernel.CORRELATION_ID,
                Kernel.GRAPH_NODE_ID,
                Kernel.NAME,
            },
            ActivityKind.EXTERNAL_CORRELATION: {
                ExternalCorrelation.EXTERNAL_ID,
                ExternalCorrelation.CORRELATION_ID,
            },
            ActivityKind.RUNTIME: {Api.CORRELATION_ID},
            ActivityKind.DRIVER: {Api.CORRELATION_ID},
        }
        try:
            obs = m.register(want, cb)
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        try:
            x = torch.randn(256, 256, device="cuda")
            for i in range(200):
                m.push_external_correlation_id()
                nested = i % 3 == 0
                if nested:
                    m.push_external_correlation_id()
                x = torch.relu(x @ x)
                if nested:
                    m.pop_external_correlation_id()
                m.pop_external_correlation_id()
                if i % 40 == 0:
                    m.flush()  # foreground drain racing the background flush loop
            x.sum().item()
            torch.cuda.synchronize()
            m.flush(sync=True)
            stats = m.stats()
        finally:
            m.unregister(obs)  # stop(): final sync flush + decoder teardown

        self.assertGreater(counts.get(int(ActivityKind.CONCURRENT_KERNEL), 0), 0)
        self.assertGreater(stats["buffers_completed"], 0)


class TestWindowFinalizer(TestCase):
    """Cover-and-finalize loop of WindowFinalizerMixin -- pure Python, no CUDA/CUPTI.
    A fake user supplies a settable native clock and a synthetic record buffer."""

    class _Fake(WindowFinalizerMixin):
        def __init__(self) -> None:
            self._clock = 0
            self._delivered: list[int] = []  # starts the "monitor" has handed over
            self._buf: list[int] = []  # collected, not-yet-finalized record starts
            # (window_id, boundary, [selected starts]) in finalize order
            self.finalized: list[tuple[int, int, list[int]]] = []
            # Huge interval so the poll thread never fires mid-test; we drive
            # _poll_once() by hand for determinism.
            self._init_windowing(poll_interval_ms=3_600_000)

        def now_native_ns(self) -> int:
            return self._clock

        def deliver(self, *starts: int) -> None:
            self._delivered.extend(starts)

        def _collect_delivered(self, *, sync: bool) -> None:
            self._buf.extend(self._delivered)
            self._delivered.clear()

        def _window_watermark_ns(self) -> int:
            return max(self._buf) if self._buf else -1

        def _finalize_window(self, window_id: int, boundary_ns: int) -> None:
            selected = [s for s in self._buf if s < boundary_ns]
            self._buf = [s for s in self._buf if s >= boundary_ns]
            self.finalized.append((window_id, boundary_ns, selected))

    def test_boundary_defers_until_covered(self):
        w = self._Fake()
        w._clock = 100
        self.assertEqual(w.mark_boundary(), 0)
        w.deliver(10, 20)  # watermark 20 < 100 -> not covered yet
        w._poll_once()
        self.assertEqual(w.finalized, [])
        w.deliver(150)  # 150 >= 100 -> boundary covered
        w._poll_once()
        self.assertEqual(w.finalized, [(0, 100, [10, 20])])
        self.assertEqual(w._buf, [150])  # consumed trimmed, tail kept
        w._stop_windowing()

    def test_boundaries_finalize_in_order_as_covered(self):
        w = self._Fake()
        w._clock = 100
        w.mark_boundary()
        w._clock = 200
        w.mark_boundary()
        w.deliver(50, 120)  # watermark 120 covers b0(100), not b1(200)
        w._poll_once()
        self.assertEqual(w.finalized, [(0, 100, [50])])
        self.assertEqual(w._buf, [120])
        w.deliver(250)
        w._poll_once()
        self.assertEqual(w.finalized[-1], (1, 200, [120]))
        w._stop_windowing()

    def test_drain_all_finalizes_remaining_uncovered(self):
        w = self._Fake()
        w._clock = 100
        w.mark_boundary()
        w._clock = 200
        w.mark_boundary()
        w.deliver(50)  # covers neither by watermark
        w._poll_once()
        self.assertEqual(w.finalized, [])
        w._stop_windowing()  # drain_all -> finalize both regardless of watermark
        self.assertEqual([f[0] for f in w.finalized], [0, 1])

    def test_poll_thread_starts_once_and_stops(self):
        w = self._Fake()
        w._clock = 10
        w.mark_boundary()
        t = w._poll_thread
        self.assertIsNotNone(t)
        w._clock = 20
        w.mark_boundary()
        self.assertIs(w._poll_thread, t)  # not restarted
        w._stop_windowing()
        self.assertIsNone(w._poll_thread)


if __name__ == "__main__":
    run_tests()
