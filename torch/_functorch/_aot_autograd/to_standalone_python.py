"""Lower a GraphModule to a self-contained Python module via AOTAutograd + Inductor.

This is the AOT half of the backend contract behind ``torch.compiler.precompile``:

    python_code, cache = compile_to_python(gm, example_inputs)

``inductor.compile_to_python`` produces only the inner Inductor ``call`` (kernels
for the post-AOTAutograd dense graph). This module wraps that with the prelude /
epilogue (input-mutation reflection, output-alias regen, subclass wrap/unwrap,
...) by COMPOSING AOTAutograd's own codegen'd runtime-wrapper source -- captured
during compile -- rather than reimplementing it. Every wrapper (the orchestration and
any chain wrappers: subclass / dedup / functionalized-RNG) is spliced verbatim as a real
top-level ``def``, with its closed-over globals hoisted to module-scope assignments, so
the module reads as ordinary code. Cross-wrapper references, the inner ``call`` chain,
public helpers, and baked metadata objects (reconstructed as source -- see
``_emit_value``) are wired by name; a guard rejects the rare case where a wrapper def
name or hoisted global would collide with another top-level name (a sibling wrapper or
an inner-module binding) rather than silently rebinding one.
The result is a standalone module exposing ``call(flat_inputs) -> outputs`` that
runs on its own (JIT-compiling kernels); ``cache`` is an opaque acceleration (or
None).

Baked metadata is emitted as plain Python source (no pickle / base64 blobs), so the
generated module is fully auditable and exec'ing it never invokes ``pickle.loads``.
A leaf that cannot be expressed as source raises NotImplementedError rather than
falling back to an opaque embedding.

Contract note: the standalone ``call`` deliberately substitutes ``nullcontext`` / a
no-op for the runtime's first-invocation context and profiler prologue, dropping the
cold-start custom-op aliasing analysis and the profiler prologue -- both diagnostics
with no effect on numerics (see the generated-call emission site). One caveat: that
dropped first-invocation custom-op aliasing analysis can itself RAISE under
``config.error_on_custom_op_aliasing`` (default on in CI), so a graph whose custom op
violates the aliasing contract runs SILENTLY in the standalone artifact where the
eager / compiled path would error -- an intentional trade-off, not a numerics bug.
"""

from __future__ import annotations

import ast
import inspect
import re
import threading
from typing import Any, TYPE_CHECKING

from .subclass_codegen import capture_generated_sources, GeneratedSource


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torch._inductor.standalone_compile import DynamicShapesType
    from torch.fx import GraphModule


# Serializes compile_to_python: the wrapper-source capture is thread-local, but the
# underlying standalone_compile swaps process-global cache state (see the THREADING
# note on compile_to_python), so concurrent compiles must not overlap. An RLock (not a
# plain Lock) because this entry point is re-entrant on a single thread: a custom
# backend or inductor pass invoked while the lock is held may call back into this
# lowering to compile a subgraph on the SAME thread, and a plain Lock would
# self-deadlock on that re-entry. On-thread re-entry is safe here: the capture sink and
# the cache-state swap (CacheArtifactManager.with_fresh_cache) are each self-contained
# per call, so a nested compile neither corrupts the outer capture nor the outer cache
# scope.
_COMPILE_LOCK = threading.RLock()


# Global objects the codegen'd wrappers close over that are reproducible as an
# import in the standalone module (rather than reconstructed field-by-field). Maps
# object id -> (import_statement, expression). Built lazily to avoid import cycles.
def _known_helper_table() -> dict[int, tuple[str, str]]:
    # Generated artifacts import runtime helpers from the single stable surface
    # ``standalone_runtime`` (not scattered AOTAutograd internals).
    import torch

    from . import standalone_runtime as rt

    _RT = "from torch._functorch._aot_autograd.standalone_runtime import"
    table: dict[int, tuple[str, str]] = {
        id(torch): ("import torch", "torch"),
        id(rt.normalize_as_list): (f"{_RT} normalize_as_list", "normalize_as_list"),
        id(rt.mark_dynamo_propagated_dynamic_indices): (
            f"{_RT} mark_dynamo_propagated_dynamic_indices",
            "mark_dynamo_propagated_dynamic_indices",
        ),
        id(torch.autograd.graph.increment_version): (
            "import torch",
            "torch.autograd.graph.increment_version",
        ),
        id(rt.gen_alias_from_base): (
            f"{_RT} gen_alias_from_base",
            "gen_alias_from_base",
        ),
        id(rt._unwrap_tensoralias): (
            f"{_RT} _unwrap_tensoralias",
            "_unwrap_tensoralias",
        ),
        id(rt.CUDARngStateHelper.get_torch_state_as_tuple): (
            f"{_RT} CUDARngStateHelper",
            "CUDARngStateHelper.get_torch_state_as_tuple",
        ),
        id(rt.CUDARngStateHelper.set_new_offset): (
            f"{_RT} CUDARngStateHelper",
            "CUDARngStateHelper.set_new_offset",
        ),
    }
    return table


_MODULE_HEADER = """\
# Generated by torch._functorch.aot_autograd.compile_to_python -- do not edit.
#
# Self-contained, executable module exposing ``call(flat_inputs) -> outputs`` for
# the post-AOTAutograd graph. The Inductor kernels JIT-compile from the inlined
# source on first call (no cache needed). The prelude/epilogue is AOTAutograd's own
# codegen'd runtime wrappers, not reimplemented: each (the orchestration and any chain
# wrappers) is spliced as a real top-level ``def`` with its closed-over globals (inner
# ``call``, sibling wrappers, public helpers, baked metadata reconstructed as source)
# hoisted to module-scope assignments -- so results match eager. The companion opaque
# cache is only an acceleration; this module never reads it.
"""


# The runtime helper for reduce-based metadata reconstruction (see _emit_via_reduce).
# Emitted only when some baked value actually needs it -- many graphs (e.g. a plain
# module with no tensor-subclass metadata) reference none, so it would just be dead
# code. Includes its own trailing blank lines so the splice stays clean.
_REBUILD_HELPER: list[str] = [
    "def _rebuild(obj, state):",
    "    # Apply pickle-style reduce state to a freshly __new__'d object, emitted as",
    "    # readable source instead of an opaque blob (no pickle.loads). Used for the",
    "    # opaque value leaves of baked metadata -- e.g. tensor-subclass placement",
    "    # objects -- mirroring pickle's load_build: prefer __setstate__, else update",
    "    # __dict__ (and slot state for the (dict, slots) 2-tuple form).",
    "    if state is None:",
    "        return obj",
    "    if hasattr(obj, '__setstate__'):",
    "        obj.__setstate__(state)",
    "        return obj",
    "    slotstate = None",
    "    if isinstance(state, tuple) and len(state) == 2:",
    "        state, slotstate = state",
    "    if state:",
    "        obj.__dict__.update(state)",
    "    if slotstate:",
    "        for _k, _v in slotstate.items():",
    "            setattr(obj, _k, _v)",
    "    return obj",
    "",
    "",
]


def _resolve_global(
    obj: Any,
    helper_table: dict[int, tuple[str, str]],
    inner_call_id: int | None,
    fn_id_to_name: dict[int, str],
    imports: set[str],
) -> str:
    """Return a Python expression (valid in the generated module) that reproduces
    ``obj``, recording any needed import. Raises NotImplementedError if ``obj`` is
    neither the inner call, a sibling wrapper, a known helper, nor source-
    reconstructible (see ``_emit_value``)."""
    if inner_call_id is not None and id(obj) == inner_call_id:
        return "_inner_call"
    if id(obj) in fn_id_to_name:
        return fn_id_to_name[id(obj)]
    if id(obj) in helper_table:
        import_stmt, expr = helper_table[id(obj)]
        if import_stmt:
            imports.add(import_stmt)
        return expr
    # Some orchestration globals are thin forwarding shims around a captured
    # wrapper (e.g. ``_replay_aliases_`` / ``_apply_mutations_`` are bound methods
    # that just forward to the codegen'd ``_alias_fn`` / ``_mutations_fn``). Unwrap a
    # shim ONLY when it closes over exactly one cell and that cell is a captured
    # wrapper / the inner call. A shim with any extra meaningful closure state falls
    # through to _emit_value (which raises) rather than silently dropping that state.
    # MAINTENANCE HAZARD: this check inspects closure STRUCTURE (one cell holding a
    # recognized ref), not the shim BODY -- it cannot tell a pure forward from a shim
    # that does extra LOGIC (rather than carrying extra state) around the same single
    # cell. Such a logic-bearing single-cell shim would be silently replaced by the
    # bare inner ref, dropping that logic; this case is not detected, so any new shim
    # added to the orchestration globals must remain a pure pass-through forwarder.
    import types as _types

    func = obj.__func__ if isinstance(obj, _types.MethodType) else obj
    closure = getattr(func, "__closure__", None)
    if closure:
        refs: list[str] = []
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if id(val) in fn_id_to_name:
                refs.append(fn_id_to_name[id(val)])
            elif inner_call_id is not None and id(val) == inner_call_id:
                refs.append("_inner_call")
        # Require the shim to have exactly one closure cell AND that cell to be the one
        # recognized ref; any extra cell (incl. an unreadable/empty one) forces a
        # fall-through to _emit_value rather than silently dropping closed-over state.
        if len(refs) == 1 and len(closure) == 1:
            return refs[0]
    # Not a wired reference (inner call / sibling / helper / shim): emit ``obj`` as
    # plain reconstruction source. Raises if it is not source-expressible.
    return _emit_value(obj, imports)


def _emit_importable(obj: Any, imports: set[str]) -> str:
    """Return an expression referencing ``obj`` via its defining module (recording the
    import). Works for classes, functions, and modules. Raises if ``obj`` is not
    reachable as ``module.qualname`` (e.g. a local or lambda), since then it cannot be
    reproduced in the standalone module without embedding it."""
    import importlib

    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not module or not qualname or "<" in qualname:
        raise NotImplementedError(
            f"compile_to_python cannot reference {obj!r} by import: it has no "
            "module/qualname or is a local definition."
        )
    target: Any = importlib.import_module(module)
    for part in qualname.split("."):
        target = getattr(target, part, None)
    if target is not obj:
        raise NotImplementedError(
            f"compile_to_python cannot reference {qualname} from {module}: it does "
            "not round-trip to the same object."
        )
    imports.add(f"import {module}")
    return f"{module}.{qualname}"


def _emit_value(
    obj: Any, imports: set[str], _in_progress: set[int] | None = None
) -> str:
    """Emit a Python expression (valid in the generated module) that rebuilds ``obj``
    from source -- the pickle-free replacement for embedding a base64 blob.

    Recurses through containers and metadata so the artifact stays auditable and
    exec'ing it never runs ``pickle.loads``. Bottoms out for opaque value objects (e.g.
    tensor-subclass placement objects) at the pickle reduce protocol, but EMITTED AS
    SOURCE via ``_rebuild`` (see ``_emit_via_reduce``). Raises NotImplementedError at
    any leaf that cannot be expressed as source (e.g. a live tensor or a lambda).

    ``_in_progress`` is an identity-keyed set of the objects currently being emitted
    on this recursion path; revisiting one means the metadata is self-referential,
    which cannot be expressed as a source literal, so we raise rather than recurse
    forever (otherwise a cyclic structure would blow the stack with RecursionError)."""
    import dataclasses
    import enum
    import functools
    import math
    import types

    import torch
    from torch._C import _functionalization as _F
    from torch.fx.experimental.symbolic_shapes import SymIntEqByExpr

    from .functional_utils import ViewMetaSequence

    # Live tensors / storages must never be baked: their reduce form is a
    # ``torch.storage._load_from_bytes(b'...')`` pickle blob, which both embeds the
    # raw weight bytes and invokes ``pickle.loads`` at exec time -- violating the
    # module's no-weights-baked / never-pickle.loads / fully-auditable guarantees.
    # Reject them explicitly here, before they can fall through to _emit_via_reduce.
    # A non-Tensor object whose reduce IS ``_load_from_bytes`` (a wrapper that
    # delegates pickling to a storage's bytes) is caught at the reduce callable in
    # _emit_via_reduce, where that callable is actually visible.
    if isinstance(
        obj, (torch.Tensor, torch.storage.TypedStorage, torch.UntypedStorage)
    ):
        raise NotImplementedError(
            f"compile_to_python cannot bake a live {type(obj).__qualname__} into "
            "standalone source: that would embed raw tensor bytes and require "
            "pickle.loads at exec time. The standalone artifact bakes no weights."
        )

    # Cycle / depth guard: thread an identity-keyed in-progress set down the recursion
    # so a self-referential metadata object raises NotImplementedError naming the
    # offending type instead of recursing until RecursionError.
    if _in_progress is None:
        _in_progress = set()
    if id(obj) in _in_progress and not isinstance(
        obj, (bool, int, float, complex, str, bytes, bytearray, type(None))
    ):
        raise NotImplementedError(
            f"compile_to_python cannot bake {type(obj).__qualname__}: it is "
            "self-referential, which is not expressible as source."
        )

    _child = _in_progress | {id(obj)}

    def _emit_value(obj: Any, imports: set[str]) -> str:
        # Shadow the module-level recursion entry point so every recursive call in
        # this function body automatically threads the current in-progress set
        # (with the parent ``obj`` added) without touching each call site.
        return _emit_value_recurse(obj, imports, _child)

    # Non-finite floats: repr() gives bare ``inf`` / ``-inf`` / ``nan``, which are
    # NameErrors in the generated module (it imports no such names). Emit a
    # self-contained constructor instead. Likewise for a complex with a non-finite
    # component (repr would be e.g. ``(inf+0j)``).
    if isinstance(obj, float) and not math.isfinite(obj):
        if math.isnan(obj):
            return "float('nan')"
        return "float('inf')" if obj > 0 else "float('-inf')"
    if isinstance(obj, complex) and not (
        math.isfinite(obj.real) and math.isfinite(obj.imag)
    ):
        real = _emit_value(obj.real, imports)
        imag = _emit_value(obj.imag, imports)
        return f"complex({real}, {imag})"

    # Plain constants reproduce via repr (but not IntEnum/StrEnum members, whose repr
    # is not valid source -- those fall through to the enum handler below). Match EXACT
    # builtin types only: a subclass of a builtin scalar (e.g. a ``str`` subclass, or an
    # ``int`` subclass with a constructor-style ``__repr__``) would mis-bake under repr
    # -- losing its type or emitting a NameError -- so it must fall through to the
    # importable/reduce path, which reconstructs it faithfully or raises.
    if obj is None or (
        not isinstance(obj, enum.Enum)
        and type(obj) in (bool, int, float, complex, str, bytes, bytearray)
    ):
        return repr(obj)

    # torch scalar singletons whose repr round-trips, plus device/Size/SymInt.
    if isinstance(obj, (torch.dtype, torch.layout, torch.memory_format)):
        imports.add("import torch")
        return repr(obj)
    if isinstance(obj, torch.device):
        imports.add("import torch")
        return f"torch.device({str(obj)!r})"
    if isinstance(obj, torch.Size):
        imports.add("import torch")
        return f"torch.Size([{', '.join(_emit_value(x, imports) for x in obj)}])"
    if isinstance(obj, torch.SymInt):
        concrete = obj.node.maybe_as_int()
        if concrete is None:
            raise NotImplementedError(
                "compile_to_python cannot bake a symbolic SymInt; precompile "
                "specializes to static shapes."
            )
        return repr(concrete)

    # Importable definitions: classes, functions, modules.
    if isinstance(obj, type) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return _emit_importable(obj, imports)
    if isinstance(obj, types.ModuleType):
        imports.add(f"import {obj.__name__}")
        return obj.__name__

    # Python enums and pybind11 enums (e.g. InverseReturnMode) both reference a member
    # by name off the importable type. pybind enum members are not singletons (as_tuple
    # hands back fresh copies), so match by value via the type's __members__ map.
    members = getattr(type(obj), "__members__", None)
    name = getattr(obj, "name", None)
    is_enum_like = isinstance(obj, enum.Enum) or (
        members is not None and name in members and members[name] == obj
    )
    if is_enum_like:
        if name is not None and members is not None and name in members:
            return f"{_emit_importable(type(obj), imports)}.{name}"
        # A combined Flag/IntFlag member (A | B) has no single member name, so emit by
        # value -- ``Type(value)`` reproduces the same combined member. A non-Flag enum
        # with no resolvable name is not source-expressible, so reject it rather than
        # emit ``Type.None`` (a SyntaxError that would break the whole module).
        if isinstance(obj, enum.Flag):
            cls = _emit_importable(type(obj), imports)
            return f"{cls}({_emit_value(obj.value, imports)})"
        raise NotImplementedError(
            f"compile_to_python cannot bake enum member {obj!r}: it has no resolvable "
            "member name and is not a Flag."
        )

    if isinstance(obj, functools.partial):
        parts = [_emit_value(obj.func, imports)]
        parts += [_emit_value(a, imports) for a in obj.args]
        parts += [
            f"{k}={_emit_value(v, imports)}" for k, v in (obj.keywords or {}).items()
        ]
        imports.add("import functools")
        return f"functools.partial({', '.join(parts)})"

    # AOT view-replay recipes: ViewMeta C++ objects round-trip through as_tuple();
    # ViewMetaSequence has no public constructor so use its _from_parts factory;
    # SymIntEqByExpr wraps a sympy expr that must be a concrete integer here.
    if isinstance(obj, _F.ViewMeta):
        # as_tuple lives on the ViewMeta subclass bindings (create_binding_with_pickle),
        # not on the base ViewMeta stub, so pyrefly cannot see it on the isinstance type.
        tup = obj.as_tuple()  # pyrefly: ignore[missing-attribute]
        return f"{_emit_importable(type(obj), imports)}({_emit_value(tuple(tup), imports)})"
    if isinstance(obj, ViewMetaSequence):
        cls = _emit_importable(ViewMetaSequence, imports)
        seq = _emit_value(list(obj.sequence), imports)
        return f"{cls}._from_parts({seq}, {_emit_value(obj.metadata, imports)})"
    if isinstance(obj, SymIntEqByExpr):
        if not getattr(obj.val, "is_Integer", False):
            raise NotImplementedError(
                "compile_to_python cannot bake symbolic view metadata; precompile "
                "specializes to static shapes."
            )
        return f"{_emit_importable(SymIntEqByExpr, imports)}({int(obj.val)})"

    # namedtuple before plain tuple (it has a richer, by-name constructor). The plain
    # container branches match EXACT builtin types only (like the scalar branch above): a
    # container SUBCLASS (e.g. a ``list`` subclass carrying extra state) would otherwise be
    # silently downcast to its base type, dropping the subclass and its state. Subclasses
    # fall through to _emit_via_reduce, which reconstructs them faithfully or rejects them
    # (its listitems/dictitems guard fires for list/dict subclasses).
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        cls = _emit_importable(type(obj), imports)
        return f"{cls}({', '.join(_emit_value(x, imports) for x in obj)})"
    if type(obj) is tuple:
        items = [_emit_value(x, imports) for x in obj]
        return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"
    if type(obj) is list:
        return f"[{', '.join(_emit_value(x, imports) for x in obj)}]"
    if type(obj) is set or type(obj) is frozenset:
        ctor = "frozenset" if isinstance(obj, frozenset) else "set"
        # Iteration order of a set is not byte-stable across processes, so emit the
        # elements in a canonical order: sort them when they are mutually orderable
        # (e.g. the int dim-index sets seen here). When they are not, sort by each
        # element's own EMITTED SOURCE EXPRESSION rather than by ``repr``: the emitted
        # expression is the exact text that lands in the artifact, so it is the only
        # key guaranteed to be byte-stable across processes. A ``repr``-keyed sort
        # would silently admit a custom-but-nondeterministic ``__repr__`` (e.g. one
        # embedding ``id(self)``), making the emitted source differ run to run.
        # Building the key via ``_emit_value`` also forces each element through the
        # same source-expressibility gate, so a non-source-expressible element raises
        # here exactly as it would when finally emitted (into a throwaway imports set
        # so the keying pass never leaks an import the final emit would not).
        elems = list(obj)
        try:
            elems = sorted(elems)
        except TypeError:
            elems = sorted(elems, key=lambda x: _emit_value(x, set()))
        return f"{ctor}([{', '.join(_emit_value(x, imports) for x in elems)}])"
    if type(obj) is dict:
        items = [
            f"{_emit_value(k, imports)}: {_emit_value(v, imports)}"
            for k, v in obj.items()
        ]
        return f"{{{', '.join(items)}}}"

    # Dataclasses (e.g. MetadataKey, SubclassViewMetaSequence) reproduce field-by-field
    # through their generated constructor, passing only the init fields. That path is
    # faithful ONLY when the object carries no state outside those init fields: a
    # non-init field (init=False) holding state, or a __post_init__ that derives state
    # the constructor call would not reproduce, would be silently dropped. Guard it the
    # same way the rest of this module guards opaque leaves -- reject rather than emit a
    # subtly-wrong artifact -- by round-tripping through the constructor and requiring
    # the rebuilt instance to compare equal to the original. The in-scope metadata
    # types (MetadataKey, SubclassViewMetaSequence, ...) are plain value dataclasses, so
    # they round-trip and keep working; a type with hidden/derived state does not and
    # raises NotImplementedError naming itself.
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = _emit_importable(type(obj), imports)
        fields = dataclasses.fields(obj)
        init_kwargs = {f.name: getattr(obj, f.name) for f in fields if f.init}
        try:
            # type(obj) is the concrete dataclass at runtime, not the protocol pyrefly
            # infers from the is_dataclass narrowing; the call is correct.
            rebuilt = type(obj)(**init_kwargs)  # pyrefly: ignore[bad-instantiation]
            round_trips = rebuilt == obj
        except Exception:
            round_trips = False
        if not round_trips:
            raise NotImplementedError(
                f"compile_to_python cannot bake dataclass {type(obj).__qualname__}: it "
                "does not round-trip through its constructor from its init fields alone "
                "(it likely has a stateful non-init field or a __post_init__ deriving "
                "state the constructor call would not reproduce), so emitting only the "
                "init fields would silently drop that state."
            )
        kw = [f"{k}={_emit_value(v, imports)}" for k, v in init_kwargs.items()]
        return f"{cls}({', '.join(kw)})"

    return _emit_via_reduce(obj, imports, _child)


# The recursion entry point used by the in-body shadow of ``_emit_value`` and by
# ``_emit_via_reduce`` to thread the identity-keyed in-progress set (finding 4).
_emit_value_recurse = _emit_value


def _emit_via_reduce(
    obj: Any, imports: set[str], _in_progress: set[int] | None = None
) -> str:
    """Last resort for opaque value objects (e.g. DTensor placements, which are C++
    objects with no source-friendly constructor): reconstruct from the pickle reduce
    protocol, but EMITTED AS SOURCE -- ``cls.__new__(cls)`` plus ``_rebuild`` applying
    the reduce state -- so there is no ``pickle.loads`` at exec time and the bytes
    are readable. The reduce state recurses back through ``_emit_value``, so any
    non-source-expressible leaf inside it still raises."""
    import copyreg

    import torch

    def _emit_value(obj: Any, imports: set[str]) -> str:
        # Continue threading the cycle guard through the reduce-state recursion.
        return _emit_value_recurse(obj, imports, _in_progress)

    try:
        reduced = obj.__reduce_ex__(2)
    except Exception as e:
        raise NotImplementedError(
            f"compile_to_python cannot make {type(obj).__module__}."
            f"{type(obj).__qualname__} self-contained: it is not source-expressible "
            f"and has no usable reduce ({type(e).__name__})."
        ) from e
    if not isinstance(reduced, tuple) or len(reduced) < 2:
        raise NotImplementedError(
            f"compile_to_python cannot reconstruct {type(obj).__qualname__}: "
            "unsupported __reduce__ form."
        )
    func, args = reduced[0], reduced[1]
    state = reduced[2] if len(reduced) > 2 else None
    listitems = reduced[3] if len(reduced) > 3 else None
    dictitems = reduced[4] if len(reduced) > 4 else None
    # A reduce whose callable is ``torch.storage._load_from_bytes`` would embed raw
    # bytes and require a pickle.loads-equivalent at exec time. This is the ONLY place
    # that callable is visible (it is the reduce result's func, not the object's
    # __reduce_ex__ method), so reject it here to uphold the no-bytes / fully-auditable
    # guarantee for any non-Tensor wrapper that delegates pickling to storage bytes.
    if func is getattr(torch.storage, "_load_from_bytes", None):
        raise NotImplementedError(
            f"compile_to_python cannot bake {type(obj).__qualname__} into standalone "
            "source: its reduce is torch.storage._load_from_bytes, which would embed "
            "raw bytes and require pickle.loads at exec time."
        )
    if listitems is not None or dictitems is not None:
        raise NotImplementedError(
            f"compile_to_python cannot reconstruct {type(obj).__qualname__}: reduce "
            "produced list/dict items (container subclass)."
        )
    # A non-None 6th element is a protocol-5 ``state_setter``: the object opted out of
    # the default __setstate__/__dict__ install that ``_rebuild`` implements, so applying
    # state via _rebuild would silently use the wrong mechanism. Reject rather than emit
    # a subtly-wrong object.
    state_setter = reduced[5] if len(reduced) > 5 else None
    if state_setter is not None:
        raise NotImplementedError(
            f"compile_to_python cannot reconstruct {type(obj).__qualname__}: reduce "
            "produced a state_setter (protocol-5 form) that _rebuild cannot apply."
        )
    if func is getattr(copyreg, "__newobj__", None):
        cls = _emit_value(args[0], imports)
        extra = ", ".join(_emit_value(a, imports) for a in args[1:])
        base = f"{cls}.__new__({cls}{', ' + extra if extra else ''})"
    elif func is getattr(copyreg, "__newobj_ex__", None):
        cls_obj, new_args, new_kwargs = args
        cls = _emit_value(cls_obj, imports)
        pos = ", ".join(_emit_value(a, imports) for a in new_args)
        kw = ", ".join(f"{k}={_emit_value(v, imports)}" for k, v in new_kwargs.items())
        joined = ", ".join(p for p in (pos, kw) if p)
        base = f"{cls}.__new__({cls}{', ' + joined if joined else ''})"
    else:
        func_expr = _emit_value(func, imports)
        base = f"{func_expr}({', '.join(_emit_value(a, imports) for a in args)})"
    if state is None:
        return base
    return f"_rebuild({base}, {_emit_value(state, imports)})"


def _module_level_names(tree: ast.Module) -> set[str]:
    """Names bound at module scope by a parsed module. Used to seed ``_reserve`` so an
    inlined wrapper's def name or hoisted global (chain wrapper or orchestration) cannot
    silently shadow a top-level name the inner Inductor module already binds."""
    names: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            # Walk each target so tuple/list/starred unpacking (``a, b = ...`` /
            # ``first, *rest = ...``) is covered, not just bare-name targets.
            for t in n.targets:
                names.update(x.id for x in ast.walk(t) if isinstance(x, ast.Name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in n.names)
    return names


def _compose_standalone_module(
    inner_python: str, captured: list[GeneratedSource]
) -> str:
    """Compose the inner Inductor ``call`` with AOTAutograd's captured runtime
    wrappers into one standalone module exposing ``call(flat_inputs) -> outputs``.

    Every wrapper (chain wrappers and the orchestration) is spliced as a real top-level
    ``def`` with its closed-over globals hoisted to module-scope assignments (resolved
    here). They are chained by name: the orchestration is invoked with the InductorWrapper
    chain head as its inner.
    """
    # The capture sink is duration-scoped over the inner inductor compile, with no
    # originating-graph id at install time, so a re-entrant on-thread AOTAutograd
    # lowering inside that window would append ITS wrappers here too. Each captured
    # wrapper is tagged at append time with its TracingContext identity (origin_id);
    # filter to the target graph's origin before composing. The target is the origin of
    # the LAST captured orchestration wrapper: the outer lowering emits its
    # orchestration only after any nested lowering it triggered has fully completed
    # (and appended), so the final orchestration is always the outer/target one.
    orchestrations = [
        g for g in captured if g.artifact_name == "runtime_wrapper_orchestration"
    ]
    if orchestrations:
        # target_origin is None only without an ambient TracingContext, which the real
        # precompile path never hits (capture always runs under one); in that defensive
        # case the filter keeps every None-origin wrapper and the count/wiring guards
        # below remain the backstop.
        target_origin = orchestrations[-1].origin_id
        captured = [g for g in captured if g.origin_id == target_origin]

    # Backward wrappers are out of scope for forward lowering; reject them up front.
    # Every other wrapper that can appear in a composable (cacheable) forward graph is
    # codegen'd as source and captured here. The one non-codegen'd wrapper,
    # FakifiedOutWrapper, only activates under fakify_first_call, which makes the graph
    # non-cacheable -- so such a graph is rejected before it ever reaches composition.
    unsupported = [g.artifact_name for g in captured if "backward" in g.artifact_name]
    if unsupported:
        raise NotImplementedError(
            "aot_autograd.compile_to_python cannot yet compose these runtime "
            f"wrappers into standalone source: {sorted(set(unsupported))}."
        )

    orchestration = [
        g for g in captured if g.artifact_name == "runtime_wrapper_orchestration"
    ]
    if len(orchestration) != 1:
        raise NotImplementedError(
            "aot_autograd.compile_to_python expected exactly one forward "
            f"orchestration wrapper, captured {len(orchestration)}."
        )
    orch = orchestration[0]
    non_orch = [g for g in captured if g is not orch]

    # The generated ``call`` invokes the orchestration POSITIONALLY by its own name (see
    # the bottom of this function): _runtime_wrapper(chain_head, contextlib.nullcontext,
    # lambda: None, flat_inputs). That mapping is hardcoded to the codegen'd signature in
    # runtime_wrappers.py (``def _runtime_wrapper(_compiled_fn_, _first_ctx_,
    # _on_before_call_, args)``). Verify the captured signature still matches so a future
    # rename/reorder fails loudly here instead of silently passing wrong arguments.
    expected_orch_params = ["_compiled_fn_", "_first_ctx_", "_on_before_call_", "args"]
    orch_def = next(
        (
            n
            for n in ast.walk(ast.parse(orch.source))
            if isinstance(n, ast.FunctionDef) and n.name == orch.fn_name
        ),
        None,
    )
    orch_params = [a.arg for a in orch_def.args.args] if orch_def is not None else None
    if orch_params != expected_orch_params:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: the orchestration wrapper signature "
            f"changed (expected {expected_orch_params}, got {orch_params}); the "
            "standalone module invokes it positionally and must be updated to match."
        )

    helper_table = _known_helper_table()
    # Every wrapper is inlined (below) as a real def at module scope under its OWN codegen'd
    # name, so references resolve to that name. Note these names are NOT distinct in general
    # -- the subclass, dedup, and debug-assert chain wrappers all codegen ``inner_fn``; what
    # holds today is that at most one chain wrapper appears per composable forward graph (see
    # the test note on multi-link chains), so the names don't actually clash. ``_reserve``
    # fails loudly if that ever stops holding (the old ``_wrapper_{i}`` scheme could carry
    # multiple same-named wrappers in private exec namespaces; inlining deliberately cannot).
    fn_id_to_name = {id(g.fn): g.fn_name for g in non_orch}

    # A chain wrapper references the inner it wraps via one of these globals
    # (subclass/dedup use ``compiled_fn``; the functionalized-RNG wrapper uses
    # ``_compiled_fn_``). The orchestration takes its inner as a call-time arg, not a
    # global, so it is never a chain wrapper. MAINTAINERS: if AOTAutograd adds a
    # forward chain wrapper that names its inner via a new global, add that name here,
    # otherwise inner-call/chain-head detection silently bypasses it.
    _INNER_NAMES = ("compiled_fn", "_compiled_fn_")

    def _inner_ref(g: GeneratedSource) -> Any:
        for nm in _INNER_NAMES:
            if nm in g.globals_dict:
                return g.globals_dict[nm]
        return None

    # The inner Inductor call is the inner-reference that is not itself a captured
    # wrapper (the innermost link of the chain). It may be absent (dense / alias
    # graphs have no chain wrapper); then the orchestration is invoked with
    # ``_inner_call`` directly.
    inner_call_id: int | None = None
    for g in captured:
        cf = _inner_ref(g)
        if cf is not None and id(cf) not in fn_id_to_name:
            inner_call_id = id(cf)
            break

    # Chain head passed to the orchestration: the outermost InductorWrapper (last non-orch
    # wrapper that wraps via an inner reference), else the inner call. Computed up front (a
    # pure capture-order check) so the order-inversion guard below fires before the later
    # name-uniqueness guard -- a mis-ordered chain is the more specific diagnosis.
    chain_head = "_inner_call"
    chain_head_g: GeneratedSource | None = None
    for g in non_orch:
        if _inner_ref(g) is not None:
            chain_head = fn_id_to_name[id(g.fn)]
            chain_head_g = g

    # "Last with an inner-ref == outermost" holds only when capture order is
    # innermost-to-outermost (it is today: subclass before functionalized-RNG). Back that
    # assumption with a guard: the true outermost wrapper is the one NO other wrapper
    # wraps, i.e. whose fn is not referenced as another wrapper's inner. If the chosen
    # head is itself wrapped, capture order inverted and the chain would be mis-ordered --
    # reject rather than silently emit a wrong chain (the wiring guard below would not
    # catch this, since every wrapper is still referenced somewhere).
    referenced_inner_ids = {
        id(_inner_ref(g)) for g in non_orch if _inner_ref(g) is not None
    }
    if chain_head_g is not None and id(chain_head_g.fn) in referenced_inner_ids:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: the selected chain head is itself wrapped "
            "by another captured wrapper, so capture order is not innermost-to-outermost "
            "as assumed; refusing to emit a mis-ordered runtime-wrapper chain."
        )

    imports: set[str] = set()

    # Parse the inner module once: to verify it binds a module-level ``call`` (the
    # inner-call contract, checked below) and to collect its top-level names so no inlined
    # wrapper's def name or hoisted global can silently shadow one.
    inner_tree = ast.parse(inner_python)
    inner_module_names = _module_level_names(inner_tree)

    # Every runtime wrapper is inlined: its def is spliced at module scope and its
    # closed-over globals hoisted to top-level assignments (no exec / private namespace).
    # So every emitted top-level name -- each wrapper's def name and each hoisted global --
    # must be unique and must not shadow a name the inner Inductor module binds. This holds
    # in practice because at most one chain wrapper appears per composable forward graph, so
    # its def name (``inner_fn`` is shared across subclass/dedup/debug-assert wrappers) and
    # the inner-ref global ``compiled_fn`` each occur once, and metadata globals are
    # per-wrapper suffixed. ``_reserve`` guards it: a collision fails loudly (rename/namespace
    # needed) rather than silently rebinding a name.
    emitted_names = set(inner_module_names) | {
        "call",
        "_inner_call",
        "_rebuild",
        "contextlib",
    }

    def _reserve(name: str) -> None:
        if name in emitted_names:
            raise NotImplementedError(
                "aot_autograd.compile_to_python: generated top-level name "
                f"{name!r} collides with another top-level name in the composed module; "
                "inlining the runtime wrappers would shadow a binding."
            )
        emitted_names.add(name)

    # Reserve every wrapper def name up front (before hoists) so a hoisted global cannot
    # shadow a def and two wrappers cannot share a name.
    for _g in (*non_orch, orch):
        _reserve(_g.fn_name)

    def _resolve_globals(globals_dict: dict[str, object]) -> list[tuple[str, str]]:
        # Resolve each global a wrapper closes over to a standalone source expression.
        # ``globals_dict`` is the pre-exec snapshot from subclass_codegen.py, so the
        # interpreter ``__builtins__`` is absent; the skip is kept defensively in case a
        # future caller hands us a post-exec live dict.
        out: list[tuple[str, str]] = []
        for gname, gobj in globals_dict.items():
            if gname == "__builtins__":
                continue
            expr = _resolve_global(
                gobj, helper_table, inner_call_id, fn_id_to_name, imports
            )
            out.append((gname, expr))
        return out

    def _emit_inline(source: str, globals_dict: dict[str, object]) -> str:
        # Splice the wrapper's def verbatim at module scope, hoisting each closed-over global
        # to a top-level assignment (skipping a name already module-available -- an imported
        # helper or ``torch`` -- detected as gname == its resolved expr). No exec / private
        # namespace: the def reads as ordinary code and is referenced by its own name. Each
        # hoisted name is ``_reserve``'d so a collision fails loudly rather than rebinding.
        hoists: list[str] = []
        for gname, expr in _resolve_globals(globals_dict):
            if gname == expr:
                continue  # already at module scope (an import / ``torch``)
            _reserve(gname)
            hoists.append(f"{gname} = {expr}")
        return "\n".join(hoists + [source, ""])

    # Chain wrappers first (innermost-to-outermost capture order), then the orchestration --
    # all spliced as real defs; a chain wrapper's hoisted inner-ref (``compiled_fn``)
    # references ``_inner_call`` / a sibling, both emitted earlier, so order is satisfied.
    wrapper_blocks = [_emit_inline(g.source, g.globals_dict) for g in non_orch]
    orch_block = _emit_inline(orch.source, orch.globals_dict)

    # _INNER_NAMES detection is a hardcoded allowlist (see above). If AOTAutograd adds
    # a forward wrapper that names its inner via an unrecognized global, that wrapper
    # is captured but may never be wired into the module -- silently composing a
    # structurally-wrong result. Enforce that every captured non-orch wrapper is
    # actually referenced somewhere: as the chain head, or by name in another block
    # (another wrapper's globals, or the orchestration's -- e.g. the ``_alias_fn`` /
    # ``_mutations_fn`` shims wired into the orchestration). A wrapper whose name
    # appears in no other block went unwired, so reject rather than emit a wrong module.
    other_text = "\n".join(wrapper_blocks + [orch_block])
    for i, g in enumerate(non_orch):
        name = fn_id_to_name[id(g.fn)]
        own = wrapper_blocks[i]
        elsewhere = other_text.replace(own, "", 1)
        # Whole-token match: ``name`` is a wrapper def name (e.g. ``inner_fn``); a raw
        # substring test would treat ``inner_fn`` as wired whenever a longer token like
        # ``inner_fn2`` is referenced, silently defeating this guard.
        wired = re.search(r"\b" + re.escape(name) + r"\b", elsewhere) is not None
        if name != chain_head and not wired:
            raise NotImplementedError(
                "aot_autograd.compile_to_python could not wire captured runtime "
                f"wrapper {g.fn_name!r} into the module (an inner-call global may be "
                "unrecognized; see _INNER_NAMES)."
            )

    # The module splices ``_inner_call = call`` below, relying on inner_python binding a
    # module-level ``call`` entry point. Inductor emits this in one of two forms: the
    # flat path defines ``def call(args):`` (FunctionDef) while the graph_partition Runner
    # path binds ``call = runner.call`` (Assign with a Name target). Verify one is present
    # so a future inductor codegen drift fails loudly here -- like the orchestration /
    # chain / wiring guards above -- instead of surfacing as a bare NameError at exec of
    # the generated module. (``inner_tree`` was parsed once up front.)
    binds_call = any(
        (isinstance(n, ast.FunctionDef) and n.name == "call")
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "call" for t in n.targets)
        )
        for n in inner_tree.body
    )
    if not binds_call:
        raise NotImplementedError(
            "compile_to_python: inner Inductor module does not bind a module-level "
            "'call' entry point (the inner-call contract); the standalone module "
            "splices ``_inner_call = call`` and must be updated to match."
        )

    # Only emit the _rebuild helper if a baked value actually reconstructs through it.
    needs_rebuild = (
        any("_rebuild(" in b for b in wrapper_blocks) or "_rebuild(" in orch_block
    )

    parts = [
        _MODULE_HEADER,
        "import contextlib",
        *sorted(imports),
        "",
        "",
        *(_REBUILD_HELPER if needs_rebuild else []),
        "# " + "=" * 70,
        "# Inner Inductor output code (kernels + ``call``)",
        "# " + "=" * 70,
        inner_python,
        "_inner_call = call",
        "",
        "# " + "=" * 70,
        "# AOTAutograd runtime wrappers (codegen'd): each inlined as a real def with its",
        "# closed-over globals hoisted to module scope -- chain wrappers first, then the",
        "# orchestration that the outer call invokes",
        "# " + "=" * 70,
        *wrapper_blocks,
        orch_block,
        "",
        "def call(flat_inputs):  # noqa: F811",
        "    # AOTAutograd orchestration: disables grad, invokes the inner chain,",
        "    # bumps mutated-input versions, applies the output epilogue.",
        "    #",
        "    # The 2nd/3rd positional args INTENTIONALLY substitute contextlib.nullcontext",
        "    # for the runtime's first-invocation context (_FirstInvocationContext) and a",
        "    # no-op for the profiler-prologue exit. This drops two cold-start diagnostics:",
        "    # the first-call custom-op aliasing analysis (_AnalyzeCustomOpInputOutputMode,",
        "    # active when check_custom_op_aliasing is set, which can even RAISE under",
        "    # error_on_custom_op_aliasing) and the profiler prologue. Neither affects",
        "    # numerics, so this is not a bug -- the standalone artifact deliberately omits",
        "    # them. (See the positional-mapping note in _compose_standalone_module.)",
        f"    return {orch.fn_name}(",
        f"        {chain_head}, contextlib.nullcontext, lambda: None, list(flat_inputs)",
        "    )",
        "",
    ]
    return "\n".join(parts)


def _find_effectful_op(gm: GraphModule, get_effect: Any) -> Any:
    """Return the first effectful OpOverload target reachable from ``gm``, or None.

    Walks the graph and descends into any child GraphModule a node references -- a
    HOP (cond/while_loop/scan) holds its body as a get_attr'd submodule or passes it
    directly as a node arg -- so an effect nested inside a HOP subgraph is caught, not
    just effects at the top level."""
    import torch
    from torch.fx import GraphModule as _GraphModule

    seen: set[int] = set()

    def _scan(g: _GraphModule) -> Any:
        if id(g) in seen:
            return None
        seen.add(id(g))
        for node in g.graph.nodes:
            if (
                node.op == "call_function"
                and isinstance(node.target, torch._ops.OpOverload)
                and get_effect(node.target) is not None
            ):
                return node.target
            for sub in _iter_subgraphs(g, node):
                found = _scan(sub)
                if found is not None:
                    return found
        return None

    def _walk_values(value: Any) -> Iterator[_GraphModule]:
        # A GraphModule can appear as a direct node arg OR nested one level (or more)
        # inside a list/tuple/dict arg -- some HOPs pass their branch/body callables
        # inside a container -- so descend into containers before the isinstance check,
        # otherwise the recursive effect scan would never enter that nested subgraph.
        if isinstance(value, _GraphModule):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from _walk_values(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from _walk_values(item)

    def _iter_subgraphs(g: _GraphModule, node: Any) -> Iterator[_GraphModule]:
        # A child graph reaches a node either as an attribute fetched by get_attr or
        # as a (possibly container-nested) argument (the form HOPs use for their
        # branch/body callables). make_fx emits FLAT get_attr targets (e.g.
        # true_graph_0) since _scan recurses per-GraphModule, so a plain getattr
        # suffices (no dotted walk needed here).
        if node.op == "get_attr":
            attr = getattr(g, node.target, None)
            if isinstance(attr, _GraphModule):
                yield attr
        for arg in (*node.args, *node.kwargs.values()):
            yield from _walk_values(arg)

    return _scan(gm)


def compile_to_python(
    gm: GraphModule,
    example_inputs: Sequence[Any],
    *,
    dynamic_shapes: DynamicShapesType = "from_example_inputs",
    options: dict[str, Any] | None = None,
) -> tuple[str, bytes | None]:
    """Compile ``gm`` to ``(python_code, cache)``; see the module docstring.

    THREADING: serialized by a process-global lock (``_COMPILE_LOCK``). The wrapper-
    source capture is thread-local, but the underlying ``standalone_compile`` enters
    ``CacheArtifactManager.with_fresh_cache()``, which swaps process-global class
    state; a concurrent compile on another thread would corrupt the captured cache
    artifacts, so concurrent calls (including via ``torch.compiler.precompile``) are serialized
    rather than run in parallel.
    """
    import torch
    from torch._higher_order_ops.effects import _get_effect
    from torch._inductor import compile_to_python as _inductor_compile_to_python

    # Validate up front: this layer dereferences ``gm.graph`` (the effectful-op scan
    # below) before reaching inductor's own type-check, so a non-GraphModule would
    # otherwise surface as an opaque AttributeError instead of this clear contract error.
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(
            "aot_autograd.compile_to_python expects a post-AOTAutograd "
            f"torch.fx.GraphModule, got {type(gm)}."
        )

    # Effectful ops thread effect tokens through a calling convention the standalone
    # composition does not reproduce (and their with_effects HOP is non-cacheable);
    # reject them up front with a concrete reason. Not supported yet. (Detected here
    # too, not only in torch.compiler.precompile's capture-time guard, so direct callers of
    # this lowering get the same clear failure rather than a silently-wrong artifact.)
    # Scan recursively: a HOP (cond/while_loop/scan) carries its body as a child
    # GraphModule referenced by a get_attr node (or passed directly as a node arg), so
    # effects nested in such a subgraph would be missed by a top-level-only scan.
    effectful = _find_effectful_op(gm, _get_effect)
    if effectful is not None:
        raise NotImplementedError(
            "aot_autograd.compile_to_python cannot lower this graph to standalone "
            f"source: it contains an effectful op ({effectful}), which is not "
            "supported yet."
        )

    with _COMPILE_LOCK:
        captured: list[GeneratedSource] = []
        with capture_generated_sources(captured):
            inner_python, cache = _inductor_compile_to_python(
                gm,
                example_inputs,
                dynamic_shapes=dynamic_shapes,
                options=options,
            )
        source = _compose_standalone_module(inner_python, captured)
    return source, cache
