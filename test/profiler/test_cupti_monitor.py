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
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

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

    def test_field_registry_invariants(self):
        # Cross-cutting invariants on the record-field registry (the source of truth for
        # columnar decode) across EVERY record kind -- guards against editing mistakes
        # (duplicate ids, registry/string/correlation drift) that test_field_catalog's
        # Kernel-only check would not catch.
        from torch.profiler._cupti.records import (
            CORRELATION_FIELD,
            FIELD_REGISTRY,
            FIELDS,
            STRING_FIELDS,
        )

        kinds = set(FIELDS)
        # Every derived map covers exactly the same kinds.
        self.assertEqual(set(FIELD_REGISTRY), kinds)
        self.assertEqual(set(STRING_FIELDS), kinds)
        self.assertEqual(set(CORRELATION_FIELD), kinds)
        for kind, fields in FIELDS.items():
            ids = [f.id for f in fields]
            # Field ids are unique within a record class (a collision would silently
            # alias two fields during decode).
            self.assertEqual(
                len(ids), len(set(ids)), f"duplicate field ids: kind {kind}"
            )
            # FIELD_REGISTRY / STRING_FIELDS derive exactly from FIELDS.
            self.assertEqual(FIELD_REGISTRY[kind], frozenset(ids))
            self.assertEqual(
                STRING_FIELDS[kind], frozenset(f.id for f in fields if f.string)
            )
            self.assertLessEqual(STRING_FIELDS[kind], FIELD_REGISTRY[kind])
            # CUPTI requires *_FIELD_KIND (id 0) first at enable, so every kind has it.
            self.assertIn(0, FIELD_REGISTRY[kind])
            # The correlation field id is a real field of this kind.
            self.assertIn(CORRELATION_FIELD[kind], FIELD_REGISTRY[kind])

    def test_cupti_python_bindings(self):
        # The cupti-python binding layer: ABI constants, the overhead-kind table,
        # and the ctypes structs that mirror CUPTI's ABI.
        from torch.profiler._cupti import cupti_python as cp

        self.assertEqual(cp.LIBCUPTI_MIN_VERSION, 130300)
        self.assertEqual(cp._ATTR_USER_DEFINED_RECORDS, 11)
        self.assertEqual(cp._ATTR_ENABLE_KERNEL_LATENCY_TIMESTAMPS, 15)
        # Overhead-kind code -> name (cupti-python does not surface these as an enum).
        self.assertEqual(cp.OVERHEAD_KIND_NAMES[0], "Unknown")
        self.assertEqual(cp.OVERHEAD_KIND_NAMES[7 << 16], "Activity Buffer Request")
        # An unknown module attribute is a clean AttributeError.
        with self.assertRaises(AttributeError):
            cp.NotAReexportedName
        # ctypes structs must match CUPTI's ABI member layout (order matters).
        self.assertEqual(
            [n for n, _ in cp._SubscriberParams._fields_],
            [
                "structSize",
                "subscriberName",
                "oldSubscriberName",
                "oldSubscriberSize",
                "allowMultipleSubscribers",
                "padding",
            ],
        )
        self.assertEqual(
            [n for n, _ in cp._UDFieldSelection._fields_],
            ["structSize", "numFields", "pFieldIds"],
        )
        self.assertEqual(
            [n for n, _ in cp._UDActivityConfig._fields_],
            ["structSize", "fieldSelection"],
        )
        # The field-selection struct is nested by value inside the activity config.
        self.assertIs(
            dict(cp._UDActivityConfig._fields_)["fieldSelection"], cp._UDFieldSelection
        )

    def test_decode_columns(self):
        # records.decode demuxes a whole buffer (possibly interleaving
        # kinds) into {kind: {field_id: column}} against CUPTI's captured layout list
        # [(kind, record_size, [(field_id, offset, size), ...])] -- every field in
        # the layout. Three strategies: single kind (stride), uniform size (stride +
        # dispatch by KIND), variable size (per-record walk); plus a bounds guard
        # dropping a trailing record that overruns valid_size, and const char* fields
        # dereferenced in place. Synthetic layouts + buffers, no CUDA.
        import numpy as np
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

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
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

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
        # The CollTrace-replacement join: a blob keyed by external_id is attached as a
        # "metadata" column onto the GPU-op kinds whose correlation_id maps (via the
        # window's EXTERNAL_CORRELATION columns -- all ours, the monitor is the sole
        # pusher) to that external_id. Pure host-side, columnar.
        import numpy as np

        from torch.profiler._cupti.observers.profiler import _attach_metadata

        columns = {
            "kernel": {
                "correlation_id": np.array([100, 200, 300]),  # 100->ext 42->blob
                "graph_node_id": np.array([0, 0, 0]),
            },
            "external_correlation": {
                "correlation_id": np.array([100, 200]),
                "external_id": np.array([42, 43]),
            },
        }
        blob = json.dumps({"func": "AllReduce", "count": 4096})
        _attach_metadata(columns, {42: blob}, None)
        meta = columns["kernel"]["metadata"]
        self.assertEqual(json.loads(meta[0]), {"func": "AllReduce", "count": 4096})
        self.assertIsNone(meta[1])  # external id has no blob
        self.assertIsNone(meta[2])  # no correlation -> external mapping
        # No metadata + no resolver is a no-op (the non-collective path pays nothing).
        clean = {
            "kernel": {
                "correlation_id": np.array([100]),
                "graph_node_id": np.array([0]),
            },
            "external_correlation": {
                "correlation_id": np.array([100]),
                "external_id": np.array([42]),
            },
        }
        _attach_metadata(clean, {}, None)
        self.assertNotIn("metadata", clean["kernel"])

    def test_attach_metadata_graph_resolver(self):
        # CUDA-graph-captured collectives have no replay-time external correlation;
        # their blob is resolved by graph_node_id via the metadata resolver (the same
        # mechanism as graph annotation names). Pure host-side, columnar.
        import numpy as np

        from torch.profiler._cupti.observers.profiler import _attach_metadata

        blob = json.dumps({"func": "AllReduce"})
        registry = {7: blob}  # stack-managed graph_node_id -> blob
        resolver = registry.get

        # Replay op: stable node 7, fresh correlation id, no EXTERNAL_CORRELATION
        # -> the eager join misses but the resolver hits.
        replay = {
            "kernel": {
                "correlation_id": np.array([999]),
                "graph_node_id": np.array([7]),
            }
        }
        _attach_metadata(replay, {}, resolver)
        self.assertEqual(
            json.loads(replay["kernel"]["metadata"][0]), {"func": "AllReduce"}
        )

        # Unknown node -> no blob (None).
        other = {
            "kernel": {
                "correlation_id": np.array([1000]),
                "graph_node_id": np.array([8]),
            }
        }
        _attach_metadata(other, {}, resolver)
        self.assertIsNone(other["kernel"]["metadata"][0])

        # Eager wins over the resolver when both could resolve (and avoids calling it).
        both = {
            "kernel": {
                "correlation_id": np.array([100]),
                "graph_node_id": np.array([7]),
            },
            "external_correlation": {
                "correlation_id": np.array([100]),
                "external_id": np.array([42]),
            },
        }
        eager_blob = json.dumps({"func": "ReduceScatter"})
        _attach_metadata(both, {42: eager_blob}, resolver)
        self.assertEqual(
            json.loads(both["kernel"]["metadata"][0]), {"func": "ReduceScatter"}
        )

    def test_metadata_blob_rendered_into_trace_args(self):
        # The comms descriptor blob attached as the "metadata" column is spread into the
        # chrome-trace kernel args (so func/count/... show up), the same way a dict
        # annotation is. Drives the columnar merge directly (no CUDA).
        import numpy as np

        from torch.profiler._cupti.monitor_trace import _trace_window_entries

        def col(v):
            return np.array([v], dtype=np.int64)

        blob = json.dumps({"func": "AllReduce", "count": 4096})
        columns = {
            "kernel": {
                "start_ns": col(1000),
                "end_ns": col(2000),
                "device_id": col(0),
                "context_id": col(1),
                "stream_id": col(7),
                "correlation_id": col(5),
                "graph_id": col(0),
                "graph_node_id": col(0),
                "name": np.array(["ncclDevKernel"], dtype=object),
                "annotation": np.array([None], dtype=object),
                "grid_x": col(1),
                "grid_y": col(1),
                "grid_z": col(1),
                "block_x": col(1),
                "block_y": col(1),
                "block_z": col(1),
                "registers_per_thread": col(0),
                "static_shared_memory": col(0),
                "dynamic_shared_memory": col(0),
                "priority": col(0),
                "queued": col(0),
                "channel": col(0),
                "channel_type": col(0),
                "metadata": np.array([blob], dtype=object),
            }
        }
        _, events = _trace_window_entries({"columns": columns}, base_ns=0)
        kernels = [e for e in events if e.get("cat") == "kernel"]
        self.assertEqual(len(kernels), 1)
        args = kernels[0]["args"]
        self.assertEqual(args["func"], "AllReduce")
        self.assertEqual(args["count"], 4096)

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
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

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
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.cupti_python import CuptiError
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
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.cupti_python import CuptiError
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
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.cupti_python import CuptiError
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
    def test_small_buffer_chain_no_deadlock(self):
        # A tiny buffer maximizes buffer-completion churn: the decode worker runs
        # constantly and the foreground + background (20ms) flushes drain, dispatch,
        # and GC the chain back-to-back -- where a flush/decode/lock deadlock would
        # surface. Drive chain push/pop (incl. nested) through it; the test completing
        # is the no-deadlock assertion (a regression would hang here), and we also
        # check the pipeline actually delivered. stop() (final sync flush + decoder
        # teardown) on unregister must not hang either.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.cupti_python import CuptiError
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
