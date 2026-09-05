"""Independent differential and on-disk replay checks using real frontend SSA."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from ninetoothed import Tensor, interpret
from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.debugger import (
    check_passes,
    compare_programs,
    export_reproducer,
    load_reproducer,
)
from ninetoothed.ir import ssa


def _arrangement(x, out):
    return x.tile((4,)), out.tile((4,))


def _affine(x, out):
    out = x * 2 + 1  # noqa: F841


def _affine_restructured(x, out):
    out = x + x + 1  # noqa: F841


def _affine_restructured_wrong(x, out):
    out = x + x + 2  # noqa: F841


def _kernel(application=_affine):
    return interpret(
        _arrangement,
        application,
        (Tensor(1, name="x", dtype="float32"), Tensor(1, name="out", dtype="float32")),
    )


def _inputs():
    return {
        "x": np.random.default_rng(2026).normal(size=7).astype(np.float32),
        "out": np.full(7, -731, dtype=np.float32),
    }


def _change_scale(program):
    operations = tuple(
        replace(operation, attrs=dict(operation.attrs, value=3))
        if operation.opcode == "arith.constant" and operation.attrs["value"] == 2
        else operation
        for operation in program.blocks[0].operations
    )
    return replace(program, blocks=(replace(program.blocks[0], operations=operations),))


def test_difference_identifies_first_aligned_operation_without_mutating_inputs():
    kernel = _kernel()
    inputs = _inputs()
    unchanged = {name: value.copy() for name, value in inputs.items()}
    comparison = compare_programs(
        kernel.program, _change_scale(kernel.program), inputs, tensors=kernel.tensors
    )
    assert not comparison.equal
    assert comparison.output_differences == ("out",)
    assert comparison.traces_aligned
    difference = comparison.first_operation
    assert difference is not None
    assert difference.opcode == "arith.constant"
    assert difference.location.startswith("entry:")
    assert difference.program_id == (0, 0, 0)
    operation = next(
        op
        for op in kernel.program.blocks[0].operations
        if op.opcode == "arith.constant" and op.attrs["value"] == 2
    )
    assert difference.result_name == operation.results[0].name
    for name in inputs:
        np.testing.assert_array_equal(inputs[name], unchanged[name])


@pytest.mark.parametrize(
    "application,expected_equal",
    ((_affine_restructured, True), (_affine_restructured_wrong, False)),
)
def test_restructured_programs_do_not_get_a_guessed_operation_location(
    application, expected_equal
):
    reference = _kernel()
    candidate = _kernel(application)
    comparison = compare_programs(
        reference.program, candidate.program, _inputs(), tensors=reference.tensors
    )
    assert comparison.equal is expected_equal
    assert not comparison.traces_aligned
    assert comparison.first_operation is None


def test_pass_checker_stops_at_the_first_semantic_failure():
    kernel = _kernel()
    visited = []

    def after_failure(program):
        visited.append("should-not-run")
        return program

    result = check_passes(
        kernel.program,
        (
            ("identity", lambda program: program),
            (
                "metadata_only",
                lambda program: replace(program, metadata={"note": "test"}),
            ),
            ("broken_scale", _change_scale),
            ("unreachable", after_failure),
        ),
        _inputs(),
        tensors=kernel.tensors,
    )
    assert not result.passed
    assert result.checked_passes == ("identity", "metadata_only", "broken_scale")
    assert result.first_bad_pass == "broken_scale"
    assert result.difference.first_operation.opcode == "arith.constant"
    assert visited == []


def test_pass_checker_records_an_unsupported_operation_as_failure():
    kernel = _kernel()

    def unsupported(program):
        block = program.blocks[0]
        return replace(
            program,
            blocks=(
                replace(
                    block,
                    operations=(ssa.Operation(opcode="broken.op"), *block.operations),
                ),
            ),
        )

    result = check_passes(
        kernel.program,
        (("introduce_unsupported", unsupported),),
        _inputs(),
        tensors=kernel.tensors,
    )
    assert not result.passed
    assert result.first_bad_pass == "introduce_unsupported"
    assert "broken.op" in result.error
    assert "entry:0" in result.error


def test_reproducer_round_trips_arrangement_inputs_and_runs_its_replay_script(tmp_path):
    kernel = _kernel()
    inputs = _inputs()
    expected = inputs["x"] * 2 + 1
    original_x = inputs["x"].copy()
    directory = export_reproducer(
        tmp_path / "case",
        kernel.program,
        inputs,
        tensors=kernel.tensors,
        grid=(2,),
        symbols={"test_symbol": 17},
        seed=2026,
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 2026
    assert {path.name for path in directory.iterdir()} == {
        "program.json",
        "program.ssa",
        "inputs.npz",
        "manifest.json",
        "replay.py",
    }
    inputs["x"][:] = 999
    program, restored, options = load_reproducer(directory)
    assert ssa.render(program) == ssa.render(kernel.program)
    np.testing.assert_array_equal(restored["x"], original_x)
    assert options["grid"] == [2]
    assert options["symbols"] == {"test_symbol": 17}
    result = interpret_program(program, restored, **options)
    np.testing.assert_allclose(result.outputs["out"], expected, rtol=1e-3, atol=1e-3)
    completed = subprocess.run(
        [sys.executable, str(directory / "replay.py")],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "out" in completed.stdout


def test_reproducer_preserves_exact_alias_bindings(tmp_path):
    value_type = ssa.Type(kind="tensor", shape=("3",), dtype="int32")
    x = ssa.Value(name="x", type=value_type)
    y = ssa.Value(name="y", type=value_type)
    program = ssa.Program(
        kind="aliases", inputs=(x, y), outputs=(x, y), blocks=(ssa.Block(),)
    )
    shared = np.array([2, 3, 5], dtype=np.int32)
    directory = export_reproducer(
        tmp_path / "alias", program, {"x": shared, "y": shared}
    )
    restored_program, restored, options = load_reproducer(directory)
    assert restored["x"] is restored["y"]
    result = interpret_program(restored_program, restored, **options)
    np.testing.assert_array_equal(result.outputs["x"], shared)


def test_reproducer_rejects_overwrite_and_manifest_input_mismatch(tmp_path):
    kernel = _kernel()
    inputs = _inputs()
    directory = export_reproducer(
        tmp_path / "case", kernel.program, inputs, tensors=kernel.tensors
    )
    original = (directory / "program.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_reproducer(directory, kernel.program, inputs, tensors=kernel.tensors)
    assert (directory / "program.json").read_bytes() == original
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["x"]["shape"] = [99]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        load_reproducer(directory)


def test_reproducer_rejects_object_arrays_without_pickle(tmp_path):
    kernel = _kernel()
    inputs = _inputs()
    inputs["x"] = inputs["x"].astype(object)
    with pytest.raises(TypeError, match="numeric"):
        export_reproducer(
            tmp_path / "object", kernel.program, inputs, tensors=kernel.tensors
        )
    assert not (tmp_path / "object" / "inputs.npz").exists()
