"""Run with PYTHONPATH=src python docs/cpu_interpreter_demo.py [--export PATH]."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ninetoothed import Tensor, interpret
from ninetoothed.interpreter.debugger import (
    StepDebugger,
    check_passes,
    export_reproducer,
)

_DIFFERENTIAL_REPLAY = '''"""Replay the deliberately injected fault from saved SSA and inputs."""

from pathlib import Path

import numpy as np

from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.debugger import compare_programs, load_reproducer

directory = Path(__file__).parent
reference, inputs, options = load_reproducer(directory / "reference")
candidate, candidate_inputs, candidate_options = load_reproducer(directory / "candidate")
comparison = compare_programs(reference, candidate, inputs, **options)
assert not comparison.equal, "The saved candidate no longer reproduces the injected fault."
assert comparison.output_differences == ("out",)
assert comparison.first_operation is not None
assert comparison.first_operation.opcode == "arith.constant"
expected = inputs["x"] * 2 + 1
result = interpret_program(reference, inputs, **options)
np.testing.assert_allclose(result.outputs["out"], expected, rtol=1e-3, atol=1e-3)
candidate_result = interpret_program(candidate, candidate_inputs, **candidate_options)
np.testing.assert_allclose(
    candidate_result.outputs["out"],
    candidate_inputs["x"] * 3 + 1,
    rtol=1e-3,
    atol=1e-3,
)
print("Fault type: deliberately injected constant 2 -> 3; saved SSA comparison")
print("Correct reference output verified against NumPy")
print(f"Different outputs: {comparison.output_differences}")
print(f"First different operation: {comparison.first_operation}")
'''


def arrangement(x, out):
    return x.tile((4,)), out.tile((4,))


def application(x, out):
    out = x * 2 + 1  # noqa: F841


def deliberately_bad_pass(program):
    """Inject one visible error to demonstrate automatic localization."""
    block = program.blocks[0]
    operations = list(block.operations)

    for index, operation in enumerate(operations):
        if operation.opcode == "arith.constant" and operation.attrs.get("value") == 2:
            operations[index] = replace(operation, attrs={"value": 3})
            break
    return replace(program, blocks=(replace(block, operations=tuple(operations)),))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        type=Path,
        help="Save reference/candidate SSA and a differential replay in a new directory.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Demonstrate scripted stepping and breakpoints.",
    )
    parser.add_argument(
        "--interactive-debug",
        action="store_true",
        help="Read debugger commands from the terminal.",
    )
    arguments = parser.parse_args()
    seed = 2026
    x = np.random.default_rng(seed).normal(size=11).astype(np.float32)
    out = np.zeros_like(x)
    debugger = None

    if arguments.debug or arguments.interactive_debug:
        commands = (
            None
            if arguments.interactive_debug
            else ("watch %0", "print %0", "step", "break mem.store", "continue")
        )
        debugger = StepDebugger(commands=commands)

    kernel = interpret(
        arrangement,
        application,
        (Tensor(1, name="x", dtype="float32"), Tensor(1, name="out", dtype="float32")),
        trace=True,
        backend="triton",
        callback=debugger,
    )
    result = kernel(x, out)
    np.testing.assert_allclose(out, x * 2 + 1, rtol=1e-3, atol=1e-3)
    print(
        f"Correct CPU output: {out.shape}, {out.dtype}; {len(result.trace)} trace events"
    )
    inputs = {"x": x, "out": np.zeros_like(out)}
    report = check_passes(
        kernel.frontend_program,
        (
            ("unchanged", lambda program: program),
            ("injected_bad_constant", deliberately_bad_pass),
        ),
        inputs,
        tensors=kernel.tensors,
        symbols=kernel.meta,
    )
    assert report.first_bad_pass == "injected_bad_constant"
    print(f"First bad pass: {report.first_bad_pass}")
    print(f"First different operation: {report.difference.first_operation}")
    print("Fault type: deliberately injected constant 2 -> 3; not a historical bug")

    if arguments.export:
        directory = arguments.export
        directory.mkdir(parents=True)
        candidate = deliberately_bad_pass(kernel.frontend_program)

        for name, program in (
            ("reference", kernel.frontend_program),
            ("candidate", candidate),
        ):
            export_reproducer(
                directory / name,
                program,
                inputs,
                tensors=kernel.tensors,
                symbols=kernel.meta,
                seed=seed,
            )

        (directory / "replay.py").write_text(_DIFFERENTIAL_REPLAY, encoding="utf-8")
        print(f"Differential replay: {directory / 'replay.py'}")


if __name__ == "__main__":
    main()
