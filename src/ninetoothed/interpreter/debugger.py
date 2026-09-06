"""CPU differential checks and portable, non-executable SSA replay bundles."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ninetoothed.ir import (
    AccessMap,
    IndexExpr,
    LayoutLevel,
    TensorLayout,
    TensorSpec,
    ir_to_dict,
    ssa,
)
from ninetoothed.ir.provenance import record_pass, seed_origins, source_candidates

from .runtime import InterpretationError, _adapt_inputs, interpret_program


class DebuggerQuit(InterpretationError):
    """The user stopped CPU execution at a completed operation boundary."""


class StepDebugger:
    """A synchronous operation debugger, usable interactively or from a script.

    Pass this object as ``interpret_program(..., callback=debugger)``. Commands
    are read from an optional iterable, otherwise from ``input()``. Exhausting
    a scripted command stream continues execution. Pauses happen after an
    operation has completed; its inputs were captured before execution.
    """

    def __init__(
        self,
        *,
        commands=None,
        output=print,
        breakpoints=(),
        watch=(),
        stop_on_entry=True,
    ):
        self._commands = None if commands is None else iter(commands)
        self.output = output
        self.breakpoints = set(breakpoints)
        self._watch = list(dict.fromkeys(watch))
        self._stepping = bool(stop_on_entry)
        self.values = {}
        self.pauses = []
        self.events_seen = 0
        self.last_event = None

    @property
    def watch_symbols(self):
        """Return names the execution engine should snapshot at each event."""
        return tuple(self._watch)

    def add_breakpoint(self, location):
        """Break at an exact location, location prefix, or opcode name."""
        self.breakpoints.add(str(location))

    def remove_breakpoint(self, location):
        """Remove a breakpoint if it exists."""
        self.breakpoints.discard(str(location))

    def add_watch(self, name):
        """Track an SSA name in subsequent execution events."""
        if name not in self._watch:
            self._watch.append(str(name))

    def remove_watch(self, name):
        """Stop requesting an SSA name in subsequent events."""
        if name in self._watch:
            self._watch.remove(name)

    def inspect(self, name):
        """Return the latest observed snapshot of a value in this program."""
        return self.values[name]

    def step(self):
        """Pause at the next completed operation."""
        self._stepping = True

    def continue_(self):
        """Continue until a breakpoint, without disabling breakpoints."""
        self._stepping = False

    def __call__(self, event):
        if self.last_event is not None and (
            self.last_event.program_id,
            self.last_event.lane,
        ) != (event.program_id, event.lane):
            self.values.clear()

        self.last_event = event
        self.events_seen += 1
        self.values.update(event.inputs or {})
        self.values.update(event.results or {})
        self.values.update(event.watched or {})
        hit = any(
            event.opcode == breakpoint
            or event.location == breakpoint
            or event.location.startswith(f"{breakpoint}:")
            for breakpoint in self.breakpoints
        )

        if not self._stepping and not hit:
            return

        self.pauses.append(event)
        self.output(
            f"paused {event.program_id} {event.location} lane={event.lane} iteration={event.iteration}"
        )

        for name in self._watch:
            if name in self.values:
                self.output(
                    f"watch {name} = {json.dumps(self.values[name], sort_keys=True)}"
                )

        while True:
            if self._commands is None:
                command = input("cpu-debug> ")
            else:
                command = next(self._commands, "continue")

            parts = command.strip().split(maxsplit=1)
            action = parts[0] if parts else "step"
            argument = parts[1] if len(parts) > 1 else None

            if action in {"s", "step"}:
                self.step()

                return

            if action in {"c", "continue"}:
                self.continue_()

                return

            if action in {"q", "quit"}:
                raise DebuggerQuit(
                    f"Debugger stopped at {event.location}, program {event.program_id}."
                )

            if action in {"b", "break"} and argument:
                self.add_breakpoint(argument)
                self.output(f"breakpoint added: {argument}")
            elif action in {"d", "delete"} and argument:
                self.remove_breakpoint(argument)
                self.output(f"breakpoint removed: {argument}")
            elif action in {"w", "watch"} and argument:
                self.add_watch(argument)
                self.output(f"watch added: {argument}")
            elif action in {"p", "print"} and argument:
                self.output(
                    f"{argument} = {json.dumps(self.values.get(argument, '<not observed>'), sort_keys=True)}"
                )
            else:
                self.output(
                    "commands: step, continue, break LOCATION|OPCODE, delete LOCATION|OPCODE, watch NAME, print NAME, quit"
                )


@dataclass(frozen=True)
class OperationDifference:
    """First mismatching corresponding operation in two compatible traces."""

    program_id: tuple
    location: str
    opcode: str
    result_name: str
    iteration: tuple
    lane: tuple | None = None


@dataclass(frozen=True)
class ProgramComparison:
    """Output differences, safe trace locations and declared source candidates.

    Source candidates describe the scope of a recorded transformation. They are
    not evidence of value correspondence or a unique cause of the output error.
    """

    equal: bool
    output_differences: tuple
    first_operation: OperationDifference | None
    traces_aligned: bool
    source_candidates: tuple = ()


@dataclass(frozen=True)
class PassCheck:
    """The first observed semantic failure in a sequence of SSA passes."""

    passed: bool
    checked_passes: tuple
    first_bad_pass: str | None
    difference: ProgramComparison | None = None
    error: str | None = None
    source_candidates: tuple = ()


def _copy_array_layout(value, *, strides=None, writeable=None):
    """Copy numeric storage while retaining signed strides and write permission."""
    if value.dtype.kind not in "biufc":
        raise TypeError("Differential replay requires numeric arrays.")

    strides = value.strides if strides is None else tuple(strides)

    if len(strides) != value.ndim or any(
        not isinstance(stride, int) for stride in strides
    ):
        raise ValueError("Array strides must contain one integer per dimension.")

    if value.size:
        extents = tuple(
            (size - 1) * stride for size, stride in zip(value.shape, strides)
        )
        lower = sum(min(0, extent) for extent in extents)
        upper = sum(max(0, extent) for extent in extents)
    else:
        lower = upper = 0

    storage = np.empty(upper - lower + value.itemsize, dtype=np.uint8)
    result = np.ndarray(
        value.shape, dtype=value.dtype, buffer=storage, offset=-lower, strides=strides
    )
    result[...] = value
    result.flags.writeable = value.flags.writeable if writeable is None else writeable

    return result


def _copy_inputs(inputs):
    inputs, _originals = _adapt_inputs(inputs)
    arrays = [
        (name, value) for name, value in inputs.items() if isinstance(value, np.ndarray)
    ]

    for index, (name, first) in enumerate(arrays):
        for other_name, second in arrays[index + 1 :]:
            if first is not second and np.shares_memory(first, second):
                raise ValueError(
                    f"Differential replay does not support overlapping views `{name}` and `{other_name}`."
                )

    memo = {}
    result = {}

    for name, value in inputs.items():
        if isinstance(value, np.ndarray):
            if id(value) not in memo:
                memo[id(value)] = _copy_array_layout(value)

            value = memo[id(value)]

        result[name] = value
    return result


def _same(first, second, rtol, atol):
    first, second = np.asarray(first), np.asarray(second)

    if first.shape != second.shape or first.dtype != second.dtype:
        return False

    if first.dtype.kind in "fc":
        return bool(np.allclose(first, second, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(first, second))


def _same_snapshot(first, second, rtol, atol):
    if "value" not in first or "value" not in second:
        return first == second

    if first["shape"] != second["shape"] or first["dtype"] != second["dtype"]:
        return False
    return _same(
        np.asarray(first["value"], dtype=first["dtype"]),
        np.asarray(second["value"], dtype=second["dtype"]),
        rtol,
        atol,
    )


def _structure(program):
    def block(value):
        return tuple(
            (
                op.opcode,
                op.operands,
                tuple(result.name for result in op.results),
                tuple(block(region) for region in op.regions),
            )
            for op in value.operations
        )

    return tuple(block(value) for value in program.blocks)


def compare_programs(
    reference,
    candidate,
    inputs,
    *,
    tensors=(),
    grid=None,
    symbols=None,
    rtol=1e-3,
    atol=1e-3,
):
    """Compare independent executions without changing the caller's buffers.

    An operation is attributed only when SSA structures and full event ordering
    align. Restructuring passes can be checked by outputs and report declared
    source candidates, but provenance alone never establishes value equivalence.
    """
    options = dict(tensors=tensors, grid=grid, symbols=symbols, trace=True)
    first = interpret_program(reference, _copy_inputs(inputs), **options)
    second = interpret_program(candidate, _copy_inputs(inputs), **options)
    differing = tuple(
        name
        for name in sorted(set(first.outputs) | set(second.outputs))
        if name not in first.outputs
        or name not in second.outputs
        or not _same(first.outputs[name], second.outputs[name], rtol, atol)
    )

    def key(event):
        return (
            event.program_id,
            event.location,
            event.opcode,
            event.iteration,
            event.lane,
        )

    aligned = _structure(reference) == _structure(candidate) and tuple(
        map(key, first.trace)
    ) == tuple(map(key, second.trace))
    first_operation = None

    if aligned:
        for left, right in zip(first.trace, second.trace):
            for name in sorted(set(left.results) | set(right.results)):
                if (
                    name not in left.results
                    or name not in right.results
                    or not _same_snapshot(
                        left.results[name], right.results[name], rtol, atol
                    )
                ):
                    first_operation = OperationDifference(
                        left.program_id,
                        left.location,
                        left.opcode,
                        name,
                        left.iteration,
                        left.lane,
                    )
                    break

            if first_operation is not None:
                break

    candidates = (
        source_candidates(
            candidate,
            (first_operation.location,) if first_operation is not None else None,
        )
        if differing
        else ()
    )
    return ProgramComparison(
        not differing, differing, first_operation, aligned, candidates
    )


def check_passes(
    program,
    passes,
    inputs,
    *,
    tensors=(),
    grid=None,
    symbols=None,
    rtol=1e-3,
    atol=1e-3,
):
    """Stop at the first bad named ``(name, Program -> Program)`` pass.

    Adapters can call existing pass objects with their genuine pass Context.
    Each transformed program is compared to the original semantic reference,
    preventing accumulated small differences from escaping detection.
    """
    program = seed_origins(program)
    current = program
    checked = []

    for name, transform in passes:
        checked.append(str(name))
        transformed = None

        try:
            previous = current
            transformed = record_pass(previous, transform(previous), str(name))
            current = transformed
            difference = compare_programs(
                program,
                current,
                inputs,
                tensors=tensors,
                grid=grid,
                symbols=symbols,
                rtol=rtol,
                atol=atol,
            )
        except (InterpretationError, ValueError, TypeError) as exc:
            return PassCheck(
                False,
                tuple(checked),
                str(name),
                error=str(exc),
                source_candidates=(
                    source_candidates(transformed) if transformed is not None else ()
                ),
            )

        if not difference.equal:
            return PassCheck(
                False,
                tuple(checked),
                str(name),
                difference,
                source_candidates=difference.source_candidates,
            )
    return PassCheck(True, tuple(checked), None)


def export_reproducer(
    directory, program, inputs, *, tensors=(), grid=None, symbols=None, seed=None
):
    """Save the supplied case as JSON SSA, numeric NPZ inputs and a replay script.

    This exports exactly the provided case; it does not claim to minimize its
    shapes or operations. Object arrays, pickles and executable SSA are excluded.
    Existing bundle files are never overwritten.
    """
    inputs, _originals = _adapt_inputs(inputs)
    directory = Path(directory)
    files = ("program.json", "program.ssa", "inputs.npz", "manifest.json", "replay.py")

    if any((directory / filename).exists() for filename in files):
        raise FileExistsError("Refusing to overwrite an existing replay bundle.")

    ssa.verify_program(program)
    _copy_inputs(inputs)  # Validate the aliasing restriction before writing files.
    arrays, bindings, aliases, names_by_id = {}, {}, {}, {}

    for index, (name, value) in enumerate(inputs.items()):
        if isinstance(value, np.ndarray) and id(value) in names_by_id:
            aliases[name] = names_by_id[id(value)]
            continue

        array = np.asarray(value)

        if array.dtype.kind not in "biufc":
            raise TypeError(f"Reproducer input `{name}` is not a numeric array/scalar.")

        key = f"input_{index}"
        arrays[key] = array
        bindings[name] = {
            "key": key,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "strides": list(array.strides),
            "writeable": bool(array.flags.writeable),
            "scalar": not isinstance(value, np.ndarray),
        }

        if isinstance(value, np.ndarray):
            names_by_id[id(value)] = name

    metadata = {
        "schema": 1,
        "seed": seed,
        "inputs": bindings,
        "aliases": aliases,
        "grid": grid,
        "symbols": dict(symbols or {}),
        "tensors": [ir_to_dict(spec) for spec in tensors],
    }
    program_text = json.dumps(ir_to_dict(program), indent=2)
    manifest_text = json.dumps(metadata, indent=2)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "program.json").write_text(program_text, encoding="utf-8")
    (directory / "program.ssa").write_text(ssa.render(program), encoding="utf-8")
    np.savez_compressed(directory / "inputs.npz", **arrays)
    (directory / "manifest.json").write_text(manifest_text, encoding="utf-8")
    (directory / "replay.py").write_text(
        '"""Replay this exact CPU case using an installed NineToothed checkout."""\n'
        "from pathlib import Path\n"
        "from ninetoothed.interpreter import interpret_program\n"
        "from ninetoothed.interpreter.debugger import load_reproducer\n"
        "program, inputs, options = load_reproducer(Path(__file__).parent)\n"
        "result = interpret_program(program, inputs, **options)\n"
        "print({name: (value.shape, str(value.dtype)) for name, value in result.outputs.items()})\n",
        encoding="utf-8",
    )

    return directory


def _program(data):
    def value(item):
        return ssa.Value(name=item["name"], type=ssa.Type(**item["type"]))

    def block(item):
        return ssa.Block(
            name=item["name"],
            args=tuple(value(arg) for arg in item["args"]),
            operations=tuple(
                ssa.Operation(
                    opcode=op["opcode"],
                    operands=tuple(op["operands"]),
                    results=tuple(value(result) for result in op["results"]),
                    attrs=op["attrs"],
                    regions=tuple(block(region) for region in op["regions"]),
                    origins=tuple(op.get("origins", ())),
                )
                for op in item["operations"]
            ),
        )

    return ssa.verify_program(
        ssa.Program(
            kind=data["kind"],
            inputs=tuple(value(item) for item in data["inputs"]),
            outputs=tuple(value(item) for item in data["outputs"]),
            blocks=tuple(block(item) for item in data["blocks"]),
            metadata=data["metadata"],
        )
    )


def _tensor(data):
    def expr(item):
        return IndexExpr(
            op=item["op"],
            value=item["value"],
            operands=tuple(expr(child) for child in item["operands"]),
        )

    def access(item):
        return AccessMap(
            source_indices=tuple(expr(value) for value in item["source_indices"]),
            linear_index=expr(item["linear_index"]),
            predicate=expr(item["predicate"]),
        )

    data = dict(data)
    layout = data["layout"]

    if layout is not None:
        data["layout"] = TensorLayout(
            **{
                name: tuple(expr(value) for value in layout[name])
                for name in (
                    "source_shape",
                    "source_strides",
                    "view_shape",
                    "application_shape",
                )
            },
            levels=tuple(
                LayoutLevel(
                    shape=tuple(expr(value) for value in level["shape"]),
                    target_dims=tuple(
                        None if value is None else expr(value)
                        for value in level["target_dims"]
                    ),
                )
                for level in layout["levels"]
            ),
            view_access=None
            if layout["view_access"] is None
            else access(layout["view_access"]),
            value_accesses=tuple(access(value) for value in layout["value_accesses"]),
        )
    return TensorSpec(**data)


def load_reproducer(directory):
    """Load only structured JSON and non-pickled numeric arrays from a bundle."""
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    if metadata.get("schema") != 1:
        raise ValueError("Unsupported replay bundle schema.")

    program = _program(
        json.loads((directory / "program.json").read_text(encoding="utf-8"))
    )
    inputs = {}

    with np.load(directory / "inputs.npz", allow_pickle=False) as arrays:
        for name, binding in metadata["inputs"].items():
            array = arrays[binding["key"]].copy()

            if (
                list(array.shape) != binding["shape"]
                or str(array.dtype) != binding["dtype"]
            ):
                raise ValueError(f"Input `{name}` disagrees with the bundle manifest.")

            inputs[name] = (
                array[()]
                if binding["scalar"]
                else _copy_array_layout(
                    array,
                    strides=binding.get("strides"),
                    writeable=binding.get("writeable", True),
                )
            )

    for name, source in metadata.get("aliases", {}).items():
        inputs[name] = inputs[source]
    return (
        program,
        inputs,
        {
            "tensors": tuple(_tensor(item) for item in metadata["tensors"]),
            "grid": metadata["grid"],
            "symbols": metadata["symbols"],
        },
    )


__all__ = [
    "StepDebugger",
    "DebuggerQuit",
    "compare_programs",
    "check_passes",
    "export_reproducer",
    "load_reproducer",
    "ProgramComparison",
    "PassCheck",
    "OperationDifference",
]
