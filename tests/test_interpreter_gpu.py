"""Required CPU/real-Triton differential checks; missing GPU is not a skip."""

import hashlib
from dataclasses import dataclass, replace

import numpy as np
import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor, interpret
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.interpreter import interpret_program
from ninetoothed.ir import ssa

SEED = 2026
RTOL = ATOL = 1e-3


def vector_arrangement(x, y, out):
    return tuple(tensor.tile((256,)) for tensor in (x, y, out))


def vector_add(x, y, out):
    out = x + y  # noqa: F841


def broadcast_arrangement(x, bias, out):
    return x.tile((1, 256)), bias.tile((256,)), out.tile((1, 256))


def broadcast_add(x, bias, out):
    out = x + bias  # noqa: F841


def reduction_arrangement(x, out):
    return x.tile((1, 512)), out.tile((1,))


def row_sum(x, out):
    out = ntl.sum(x, axis=1)  # noqa: F841


def comparison_arrangement(x, out):
    return x.tile((256,)), out.tile((256,))


def comparison(x, out):
    out = (x >= -2) & (x < 5)  # noqa: F841


def control_arrangement(x, positive, out):
    return x.tile((256,)), positive, out.tile((256,))


def control_flow(x, positive, out):
    accumulator = x
    for i in range(3):
        if positive:
            accumulator = accumulator + i
        else:
            accumulator = accumulator - i
    out = accumulator  # noqa: F841


def softmax_arrangement(x, out):
    return x.tile((1, 512)), out.tile((1, 512))


def row_softmax(x, out):
    shifted = x - ntl.max(x, axis=1)[:, None]
    numerator = ntl.exp(shifted)
    out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841


@dataclass(frozen=True)
class GPUCase:
    name: str
    category: str
    dtype: str = "float32"
    size: int = 1031
    positive: bool = True


GPU_CASES = (
    GPUCase("elementwise_float32_aligned", "elementwise", size=1024),
    GPUCase("masked_float32_tail", "masked_tail"),
    GPUCase("masked_int32_tail", "masked_tail", dtype="int32"),
    GPUCase("broadcast_float32_tail", "broadcast"),
    GPUCase("row_reduction_float32", "row_reduction"),
    GPUCase("row_reduction_int32", "row_reduction", dtype="int32"),
    GPUCase("comparison_bool_exact", "comparison", dtype="bool"),
    GPUCase("if_for_true", "if_for", positive=True),
    GPUCase("if_for_false", "if_for", positive=False),
    GPUCase("softmax_float32", "softmax"),
)


def require_gpu(device_index=0):
    """Load GPU packages only when GPU validation is actually requested."""
    try:
        import torch
        import triton
    except ImportError as error:
        raise RuntimeError(
            "GPU validation UNVERIFIED: PyTorch and Triton must be installed."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("GPU validation UNVERIFIED: no usable CUDA GPU.")
    if not 0 <= device_index < torch.cuda.device_count():
        raise RuntimeError(f"GPU validation UNVERIFIED: invalid device {device_index}.")
    torch.cuda.set_device(device_index)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return torch, triton


def _descriptor(name, ndim, dtype="float32", **kwargs):
    return Tensor(ndim, name=name, dtype=dtype, **kwargs)


def case_inputs(case):
    """Construct fixed-seed inputs and an independent NumPy oracle."""
    rng = np.random.default_rng(SEED)
    dtype = np.dtype(case.dtype)
    if case.category in {"elementwise", "masked_tail"}:
        if dtype.kind == "i":
            x = rng.integers(-100, 100, size=case.size, dtype=dtype)
            y = rng.integers(-100, 100, size=case.size, dtype=dtype)
        else:
            x = rng.normal(size=case.size).astype(dtype)
            y = rng.normal(size=case.size).astype(dtype)
        return (
            vector_arrangement,
            vector_add,
            tuple(_descriptor(name, 1, case.dtype) for name in ("x", "y", "out")),
            {"x": x, "y": y},
            x + y,
        )
    if case.category == "broadcast":
        x = rng.normal(size=(7, 257)).astype(np.float32)
        bias = rng.normal(size=257).astype(np.float32)
        return (
            broadcast_arrangement,
            broadcast_add,
            (_descriptor("x", 2), _descriptor("bias", 1), _descriptor("out", 2)),
            {"x": x, "bias": bias},
            x + bias,
        )
    if case.category == "row_reduction":
        x = (
            rng.integers(-10, 10, size=(7, 257), dtype=dtype)
            if dtype.kind == "i"
            else rng.normal(size=(7, 257)).astype(dtype)
        )
        return (
            reduction_arrangement,
            row_sum,
            (
                _descriptor("x", 2, case.dtype, other=0),
                _descriptor("out", 1, case.dtype),
            ),
            {"x": x},
            x.sum(axis=1, dtype=dtype),
        )
    if case.category == "comparison":
        x = rng.integers(-10, 10, size=case.size, dtype=np.int32)
        return (
            comparison_arrangement,
            comparison,
            (_descriptor("x", 1, "int32"), _descriptor("out", 1, "bool")),
            {"x": x},
            (x >= -2) & (x < 5),
        )
    if case.category == "if_for":
        x = rng.normal(size=case.size).astype(np.float32)
        return (
            control_arrangement,
            control_flow,
            (
                _descriptor("x", 1),
                _descriptor("positive", 0, "bool", constexpr=True),
                _descriptor("out", 1),
            ),
            {"x": x, "positive": case.positive},
            x + (3 if case.positive else -3),
        )
    if case.category == "softmax":
        x = rng.normal(size=(7, 257)).astype(np.float32) * 3
        x[0] += 1000  # Stable subtraction must avoid exponential overflow.
        numerator = np.exp(x - np.max(x, axis=1, keepdims=True))
        expected = numerator / np.sum(numerator, axis=1, keepdims=True)
        return (
            softmax_arrangement,
            row_softmax,
            (_descriptor("x", 2, other=float("-inf")), _descriptor("out", 2)),
            {"x": x},
            expected,
        )
    raise ValueError(f"Unknown validation category: {case.category}")


def _program_from_metadata(data):
    """Reconstruct the exact emitted SSA, after genuine compiler lowering."""

    def value(item):
        return ssa.Value(name=item["name"], type=ssa.Type(**item["type"]))

    def block(item):
        return ssa.Block(
            name=item["name"],
            args=tuple(value(arg) for arg in item["args"]),
            operations=tuple(
                ssa.Operation(
                    opcode=op["opcode"],
                    operands=op["operands"],
                    results=tuple(value(result) for result in op["results"]),
                    attrs=op["attrs"],
                    regions=tuple(block(region) for region in op["regions"]),
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


def _assert_equal(actual, expected, label):
    assert actual.shape == expected.shape, label
    assert actual.dtype == expected.dtype, label
    if expected.dtype.kind == "f":
        np.testing.assert_allclose(
            actual, expected, rtol=RTOL, atol=ATOL, err_msg=label
        )
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=label)


def run_gpu_case(case, torch, device_index=0):
    """Compare raw SSA, emitted target SSA, actual GPU output, and NumPy."""
    arrangement, application, tensors, source_inputs, expected = case_inputs(case)
    cpu = interpret(arrangement, application, tensors)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            backend="triton",
            kernel_name=f"gpu_validation_{case.name}",
            num_warps=4,
            max_num_configs=1,
        )
    )
    assert compilation.artifact.metadata["lowering_ir"] == "ssa.Program"
    assert compilation.artifact.metadata["generation_py_fallback"] is False
    assert ssa.render(
        replace(cpu.program, kind=compilation.kernel.ssa.kind)
    ) == ssa.render(compilation.kernel.ssa)
    lowered = _program_from_metadata(compilation.artifact.metadata["ssa"])
    assert "ssa.triton.optimize_schedule" in lowered.metadata["pass_trace"]

    def cpu_inputs():
        return {
            **{
                name: value.copy() if isinstance(value, np.ndarray) else value
                for name, value in source_inputs.items()
            },
            "out": np.full_like(expected, -123 if expected.dtype.kind != "b" else True),
        }

    raw_inputs = cpu_inputs()
    cpu(**raw_inputs)
    _assert_equal(raw_inputs["out"], expected, "frontend SSA versus NumPy")
    lowered_inputs = cpu_inputs()
    interpreted = interpret_program(
        lowered,
        lowered_inputs,
        tensors=compilation.kernel.tensors,
        symbols=compilation.kernel.metadata.get("meta_defaults", {}),
    )
    _assert_equal(
        interpreted.outputs["out"], expected, "emitted target SSA versus NumPy"
    )

    device = torch.device("cuda", device_index)
    gpu_inputs = {
        name: torch.from_numpy(value.copy()).to(device)
        if isinstance(value, np.ndarray)
        else value
        for name, value in source_inputs.items()
    }
    guard = True if expected.dtype.kind == "b" else -13579
    backing = torch.from_numpy(
        np.full(expected.size + 8, guard, dtype=expected.dtype)
    ).to(device)
    gpu_inputs["out"] = backing[4:-4].reshape(expected.shape)
    launch = DEFAULT_COMPILER.materialize(compilation)
    launch(**gpu_inputs)
    torch.cuda.synchronize(device)
    actual = gpu_inputs["out"].cpu().numpy()
    _assert_equal(actual, raw_inputs["out"], "actual Triton GPU versus frontend SSA")
    _assert_equal(
        actual, interpreted.outputs["out"], "actual Triton GPU versus emitted SSA"
    )
    _assert_equal(actual, expected, "actual Triton GPU versus independent NumPy")
    guards = backing.cpu().numpy()[np.r_[0:4, expected.size + 4 : expected.size + 8]]
    np.testing.assert_array_equal(
        guards, np.full(8, guard, dtype=expected.dtype), err_msg="GPU output overrun"
    )
    for name, expected_input in source_inputs.items():
        if isinstance(expected_input, np.ndarray):
            np.testing.assert_array_equal(
                gpu_inputs[name].cpu().numpy(),
                expected_input,
                err_msg=f"Input mutated: {name}",
            )
    return {
        "name": case.name,
        "category": case.category,
        "program": application.__name__,
        "dtype": str(expected.dtype),
        "output_shape": list(expected.shape),
        "seed": SEED,
        "status": "PASS",
        "guard_lanes_unchanged": True,
        "pass_trace": list(compilation.pass_trace),
        "emitted_ssa_sha256": hashlib.sha256(ssa.render(lowered).encode()).hexdigest(),
        "max_abs_error": float(
            np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
        ),
    }


@pytest.fixture(scope="module")
def gpu_runtime():
    try:
        return require_gpu()
    except RuntimeError as error:
        pytest.fail(str(error), pytrace=False)


@pytest.mark.parametrize("case", GPU_CASES, ids=lambda case: case.name)
def test_cpu_interpreter_matches_actual_triton_gpu(case, gpu_runtime):
    torch, _triton = gpu_runtime
    run_gpu_case(case, torch)
