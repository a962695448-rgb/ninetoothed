"""Check declared numeric results and explain their observed value dependencies."""

import hashlib
from dataclasses import dataclass

import numpy as np

from ninetoothed.ir import ssa
from ninetoothed.ir.provenance import operation_locations


@dataclass(frozen=True)
class ResultMismatch:
    """A failed declared equality, not a proof of unique causal blame."""

    reference_index: int
    candidate_index: int
    reference_result: str
    candidate_result: str


@dataclass(frozen=True)
class DependencyEvent:
    """One executed producer on a backward value slice."""

    trace_index: int
    location: str
    opcode: str
    program_id: tuple
    iteration: tuple
    lane: tuple | None


@dataclass(frozen=True)
class DependencySlice:
    """Observed SSA value edges with explicit memory/control-flow boundaries."""

    events: tuple[DependencyEvent, ...]
    boundaries: tuple[str, ...]
    scope: str = "Observed value dependencies; not a minimal slice or unique cause."


def _contexts(program):
    contexts = {}

    def visit(block, path, parents):
        for index, operation in enumerate(block.operations):
            location = f"{path}:{index}:{operation.opcode}"
            contexts[location] = parents

            for region_index, region in enumerate(operation.regions):
                header = (
                    operation.opcode,
                    operation.operands,
                    operation.attrs,
                    region_index,
                    region.args,
                )
                visit(region, f"{location}/region{region_index}", (*parents, header))

    visit(program.blocks[0], "entry", ())

    return contexts


def compare_mapped_results(
    reference, candidate, first, second, same_snapshot, rtol, atol
):
    """Compare explicitly declared results only at matching execution contexts.

    All occurrences of a mapping must align before any of its mismatches is
    reported. A lane projection checks executed scalar lanes against a numeric
    reference tile; it does not claim coverage of inactive padded lanes.
    """
    history = candidate.metadata.get("provenance", {}).get("passes", ())

    if not history or not history[-1].get("value_mappings"):
        return None, ()

    latest = history[-1]

    for key, program in (
        ("input_fingerprint", reference),
        ("output_fingerprint", candidate),
    ):
        if latest.get(key) != hashlib.sha256(ssa.render(program).encode()).hexdigest():
            return None, (
                "Result mapping does not match the compared SSA fingerprints.",
            )

    before, after = (
        dict(operation_locations(reference)),
        dict(operation_locations(candidate)),
    )
    left_contexts, right_contexts = _contexts(reference), _contexts(candidate)
    mismatches, issues, used_targets = [], [], set()

    for mapping in latest["value_mappings"]:
        source, target = mapping["source_location"], mapping["target_location"]
        left_name, right_name = mapping["source_result"], mapping["target_result"]
        right_key = (target, right_name)

        if right_key in used_targets:
            return None, ("Duplicate result mapping is ambiguous.",)

        used_targets.add(right_key)

        if source not in before or target not in after:
            issues.append(f"Missing result producer: {source} -> {target}.")
            continue

        if left_contexts[source] != right_contexts[target]:
            issues.append(f"Changed enclosing control flow: {source} -> {target}.")
            continue

        if not any(
            source in relation["inputs"]
            and target in relation["outputs"]
            and relation["relation"] in {"preserve", "replace", "split", "merge"}
            for relation in latest["relations"]
        ):
            issues.append(f"Missing operation relation: {source} -> {target}.")
            continue

        left = [
            (index, event)
            for index, event in enumerate(first.trace)
            if event.location == source
        ]
        right = [
            (index, event)
            for index, event in enumerate(second.trace)
            if event.location == target
        ]
        projection = mapping["projection"]
        pairs = []

        if projection == "identity":

            def key(event):
                return event.program_id, event.iteration, event.lane

            left_keys, right_keys = (
                [key(event) for _, event in left],
                [key(event) for _, event in right],
            )

            if left_keys == right_keys and len(set(left_keys)) == len(left_keys):
                pairs = list(zip(left, right))
        elif projection == "lane" and all(event.lane is None for _, event in left):
            catalog = {
                (event.program_id, event.iteration): (index, event)
                for index, event in left
            }
            right_keys = [
                (event.program_id, event.iteration, event.lane) for _, event in right
            ]

            if (
                len(catalog) == len(left)
                and len(set(right_keys)) == len(right_keys)
                and set(catalog)
                == {(event.program_id, event.iteration) for _, event in right}
                and all(event.lane is not None for _, event in right)
            ):
                pairs = [
                    (catalog[event.program_id, event.iteration], (index, event))
                    for index, event in right
                ]

        if not pairs:
            issues.append(
                f"Result events unavailable or unaligned: {source} -> {target}."
            )
            continue

        local_mismatches = []

        for (left_index, left_event), (right_index, right_event) in pairs:
            lhs, rhs = (
                left_event.results.get(left_name),
                right_event.results.get(right_name),
            )

            if (
                not isinstance(lhs, dict)
                or not isinstance(rhs, dict)
                or "value" not in lhs
                or "value" not in rhs
            ):
                issues.append(
                    f"Result is not a numeric snapshot: {source} -> {target}."
                )
                break

            if projection == "lane":
                array = np.asarray(lhs["value"], dtype=lhs["dtype"])
                lane = right_event.lane

                if (
                    rhs["shape"] != []
                    or len(lane) != array.ndim
                    or any(i < 0 or i >= size for i, size in zip(lane, array.shape))
                ):
                    issues.append(
                        f"Invalid scalar lane projection: {source} -> {target}."
                    )
                    break

                lhs = {"dtype": lhs["dtype"], "shape": [], "value": array[lane].item()}

            if not same_snapshot(lhs, rhs, rtol, atol):
                local_mismatches.append(
                    ResultMismatch(left_index, right_index, left_name, right_name)
                )
        else:
            mismatches.extend(local_mismatches)
    return min(mismatches, key=lambda item: item.candidate_index, default=None), tuple(
        issues
    )


def backward_slice(program, trace, observation):
    """Follow value, selected-yield and loop-carried execution dependencies.

    Program ID, scalar lane and iteration scope keep separate executions apart.
    Mutable memory stops traversal explicitly: no guessed last-writer/alias
    relation is inserted. Slice membership is not independently blamed guilt.
    """
    operations = dict(operation_locations(program))
    arguments, parents = {}, {}

    def describe(block, path, enclosing=(), loop_depth=0):
        for offset, op in enumerate(block.operations):
            location = f"{path}:{offset}:{op.opcode}"
            parents[location] = enclosing

            for region_index, region in enumerate(op.regions):
                depth = loop_depth + (op.opcode == "scf.for")

                for index, value in enumerate(region.args):
                    arguments[value.name] = (location, op, index, depth)

                describe(
                    region,
                    f"{location}/region{region_index}",
                    (*enclosing, (location, op, depth)),
                    depth,
                )

    describe(program.blocks[0], "entry")
    definitions, yields, edges, boundaries = {}, {}, [], []
    roots = {value.name for value in program.inputs} | set(
        program.metadata.get("symbols", ())
    )

    for index, event in enumerate(trace):
        operation = operations[event.location]
        links, stops = set(), set()

        def resolve(name, seen=()):
            if name in seen:
                stops.add(f"{event.location}: cyclic value dependency {name}")

                return

            producer = definitions.get((event.program_id, event.lane, name))

            if producer is not None:
                origin = trace[producer]

                if event.iteration[: len(origin.iteration)] == origin.iteration:
                    links.add(producer)

                    return

            if name in arguments:
                location, parent, position, depth = arguments[name]

                if parent.opcode == "scf.for" and len(event.iteration) >= depth:
                    if position == 0:
                        # The induction value is determined by the loop schedule.
                        for operand in parent.operands[:3]:
                            resolve(operand, (*seen, name))
                    else:
                        prior = yields.get(
                            (
                                event.program_id,
                                event.lane,
                                location,
                                event.iteration[: depth - 1],
                            )
                        )

                        if prior is not None:
                            links.add(prior)
                        else:
                            resolve(parent.operands[position + 2], (*seen, name))
                    return

            if name not in roots:
                stops.add(
                    f"{event.location}: unresolved region argument or value {name}"
                )

        names = set(operation.operands) | set(operation.attrs.get("indices", ()))

        for name in names:
            resolve(name)

        for _location, parent, _depth in parents[event.location]:
            for name in parent.operands[: 3 if parent.opcode == "scf.for" else 1]:
                resolve(name)

        if operation.regions:
            yielded = yields.get(
                (event.program_id, event.lane, event.location, event.iteration)
            )

            if yielded is not None:
                links.add(yielded)
            elif operation.opcode not in {"scf.for", "scf.if"}:
                stops.add(f"{event.location}: unsupported region-result dependency")

        if operation.opcode in {"mem.load", "mem.store", "tensor.extract"}:
            stops.add(f"{event.location}: memory/alias history not reconstructed")

        edges.append(links)
        boundaries.append(stops)

        for result in operation.results:
            definitions[event.program_id, event.lane, result.name] = index

        if operation.opcode == "scf.yield" and parents[event.location]:
            location, parent, depth = parents[event.location][-1]
            outer = (
                event.iteration[: depth - 1]
                if parent.opcode == "scf.for"
                else event.iteration
            )
            yields[event.program_id, event.lane, location, outer] = index

    indices = [
        index
        for index, event in enumerate(trace)
        if (
            event.location == observation.location
            and event.program_id == observation.program_id
            and event.iteration == observation.iteration
            and event.lane == observation.lane
        )
    ]

    if len(indices) != 1:
        return DependencySlice((), ("Observation has no unique execution event.",))

    visited, pending = set(), list(indices)

    while pending:
        index = pending.pop()

        if index not in visited:
            visited.add(index)
            pending.extend(edges[index])

    events = tuple(
        DependencyEvent(
            index,
            trace[index].location,
            trace[index].opcode,
            trace[index].program_id,
            trace[index].iteration,
            trace[index].lane,
        )
        for index in sorted(visited)
    )

    return DependencySlice(
        events,
        tuple(sorted({message for index in visited for message in boundaries[index]})),
    )
