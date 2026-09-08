"""Capture and replay differential failures without serializing Python passes."""

import json
import platform
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from ninetoothed.ir import ir_to_dict, ssa

from .debugger import compare_programs, export_reproducer, load_reproducer
from .runtime import InterpretationError


def export_failure(
    directory,
    reference,
    candidate,
    inputs,
    *,
    tensors=(),
    grid=None,
    symbols=None,
    rtol=1e-3,
    atol=1e-3,
    seed=None,
    previous=None,
    report=None,
    error=None,
    phase="execution",
):
    """Write a new failure directory, retaining an incomplete marker on I/O error.

    Execution mismatches/errors are replayable from saved SSA alone. A Python
    transform which raises or returns invalid SSA is saved as diagnostic data,
    explicitly without claiming to reproduce the Python transform itself.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    incomplete = directory / "INCOMPLETE"
    incomplete.write_text("Failure export has not completed.\n", encoding="utf-8")
    options = dict(tensors=tensors, grid=grid, symbols=symbols, seed=seed)
    export_reproducer(directory / "reference", reference, inputs, **options)

    if previous is not None:
        export_reproducer(directory / "previous", previous, inputs, **options)

    replayable = phase == "execution"

    if replayable:
        export_reproducer(directory / "candidate", candidate, inputs, **options)
    elif isinstance(candidate, ssa.Program):
        (directory / "candidate.json").write_text(
            json.dumps(ir_to_dict(candidate), indent=2), encoding="utf-8"
        )

    try:
        package_version = version("ninetoothed")
    except PackageNotFoundError:
        package_version = None

    metadata = {
        "schema": 1,
        "kind": "passes" if previous is not None else "comparison",
        "phase": phase,
        "replayable": replayable,
        "seed": seed,
        "rtol": rtol,
        "atol": atol,
        "error": None
        if error is None
        else {
            "type": type(error).__name__,
            "message": str(error),
        },
        "report": None if report is None else asdict(report),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ninetoothed": package_version,
        },
        "scope": "Exact supplied SSA and inputs; no automatic program or shape minimization.",
    }
    (directory / "replay.py").write_text(
        '"""Verify the recorded differential failure using installed NineToothed."""\n'
        "from pathlib import Path\n"
        "from ninetoothed.interpreter.failure import replay_failure\n"
        "replay_failure(Path(__file__).parent)\n",
        encoding="utf-8",
    )
    (directory / "failure.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    incomplete.unlink()

    return directory


def _verify_comparison(actual, expected):
    if (
        actual.equal != expected["equal"]
        or list(actual.output_differences) != expected["output_differences"]
    ):
        raise RuntimeError(
            "The saved candidate no longer reproduces the recorded output difference."
        )

    for name in ("first_operation", "aligned_prefix_operation", "retained_operation"):
        observed = getattr(actual, name)
        observed = (
            None if observed is None else json.loads(json.dumps(asdict(observed)))
        )

        if observed != expected[name]:
            raise RuntimeError(
                "The saved candidate no longer reproduces the recorded operation difference."
            )

    if actual.traces_aligned != expected["traces_aligned"]:
        raise RuntimeError(
            "The saved candidate no longer reproduces the recorded trace alignment."
        )


def replay_failure(directory):
    """Re-execute saved SSA and require the recorded failure, even under python -O.

    The saved Python pass is never imported or executed. A pass report is
    reproduced by comparing its original, previous and candidate SSA snapshots.
    Mismatch in failure kind, outputs, operation or exception raises an error.
    """
    directory = Path(directory)

    if (directory / "INCOMPLETE").exists():
        raise ValueError("The failure bundle is incomplete.")

    metadata = json.loads((directory / "failure.json").read_text(encoding="utf-8"))

    if metadata.get("schema") != 1:
        raise ValueError("Unsupported differential failure schema.")

    if not metadata["replayable"]:
        raise ValueError(
            "This is a diagnostic snapshot of a Python pass failure; executable pass code was not exported."
        )

    reference, inputs, options = load_reproducer(directory / "reference")
    candidate, _, _ = load_reproducer(directory / "candidate")
    options.update(rtol=metadata["rtol"], atol=metadata["atol"])
    expected_error = metadata["error"]

    try:
        comparison = compare_programs(reference, candidate, inputs, **options)
    except (InterpretationError, ValueError, TypeError) as exc:
        actual_error = {"type": type(exc).__name__, "message": str(exc)}

        if actual_error != expected_error:
            raise RuntimeError(
                "The saved candidate produced a different execution failure."
            ) from exc

        print(f"Reproduced execution failure: {exc}")

        return None

    if expected_error is not None:
        raise RuntimeError(
            "The saved candidate no longer reproduces the recorded execution failure."
        )

    report = metadata["report"]

    if metadata["kind"] == "passes":
        previous, _, _ = load_reproducer(directory / "previous")
        adjacent = compare_programs(previous, candidate, inputs, **options)
        _verify_comparison(adjacent, report["adjacent_difference"])
        _verify_comparison(comparison, report["difference"])
        print(f"First bad pass (saved boundary): {report['first_bad_pass']}")
        print(f"Operation localization: {report['localization']}")
        displayed = comparison if not comparison.equal else adjacent
    else:
        _verify_comparison(comparison, report)
        displayed = comparison

    if displayed.equal:
        raise RuntimeError(
            "The saved candidate no longer reproduces a differential failure."
        )

    print(f"Different outputs: {displayed.output_differences}")
    print(f"First different operation: {displayed.first_operation}")

    return comparison


__all__ = ["replay_failure"]
