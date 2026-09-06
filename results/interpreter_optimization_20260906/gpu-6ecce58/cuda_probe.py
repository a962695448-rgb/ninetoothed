#!/usr/bin/env python3
"""Run one frozen NineToothed CUDA-backend dot validation in an approved GPU window.

Uses the repository's dot_float32_multi_program_tail fixture and compiler;
there is no hand-written GPU kernel. --help performs no GPU initialization.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_HEAD = "6ecce58da28bb9709aa35fc6c25c1f361aff736f"
SOURCE_SCOPE = (
    "src",
    "tests",
    "scripts",
    "pyproject.toml",
    "requirements.txt",
    ".github",
)
CASE_NAME = "dot_float32_multi_program_tail"
GUARD = -13579


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def source_state(repo):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=repo).decode().strip()

    head = git("rev-parse", "HEAD")
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"Expected frozen source {EXPECTED_HEAD}; got {head}.")
    if git("ls-files", "--others", "--exclude-standard", "--", *SOURCE_SCOPE):
        raise RuntimeError(
            "Untracked files exist in the frozen source/test/config scope."
        )
    names = git("ls-files", "--", *SOURCE_SCOPE).splitlines()
    requests = "".join(f"{head}:{name}\n" for name in names).encode()
    blobs = io.BytesIO(
        subprocess.check_output(
            ["git", "cat-file", "--batch"], cwd=repo, input=requests
        )
    )
    files = {}
    for name in names:
        object_id, kind, size = blobs.readline().decode().split()
        if kind != "blob":
            raise RuntimeError(f"Expected a Git blob: {name}.")
        reference = blobs.read(int(size))
        if blobs.read(1) != b"\n":
            raise RuntimeError("Malformed git cat-file response.")
        actual = (repo / name).read_bytes()
        normalized = actual
        if actual != reference:
            actual.decode("utf-8")
            reference.decode("utf-8")
            if b"\0" in actual:
                raise RuntimeError(f"Modified binary source: {name}.")
            normalized = actual.replace(b"\r\n", b"\n")
        if normalized != reference:
            raise RuntimeError(f"Source differs from frozen Git blob: {name}.")
        files[name] = {
            "git_blob": object_id,
            "raw_sha256": sha256(actual),
            "normalized_sha256": sha256(normalized),
        }
    return {"head": head, "normalization": "UTF-8 CRLF to LF only", "files": files}


def run_probe(repo, out, device_index, report):
    # Cache isolation must precede ninetoothed.compiler.cache import. The fresh
    # output directory ensures this run cannot reuse an earlier CUDA binary.
    os.environ["NINETOOTHED_CACHE_DIR"] = str(out / "build-cache")
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(repo / "src"), str(repo)]

    import numpy as np
    import torch
    from ninetoothed.backends.core import Target
    from ninetoothed.backends.toolchain import find_nvcc
    from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
    from ninetoothed.interpreter import interpret_program
    from ninetoothed.ir import ir_to_dict, ssa
    from tests.test_interpreter_gpu import (
        ATOL,
        GPU_CASES,
        RTOL,
        SEED,
        _assert_equal,
        _program_from_metadata,
        _scalar_dot_offsets,
        case_inputs,
    )

    import ninetoothed
    from ninetoothed import interpret

    if not Path(ninetoothed.__file__).resolve().is_relative_to(repo / "src"):
        raise RuntimeError(
            "NineToothed was imported from outside the frozen repository."
        )
    if (
        not torch.cuda.is_available()
        or not 0 <= device_index < torch.cuda.device_count()
    ):
        raise RuntimeError("No usable CUDA device at the requested index.")
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    capability = torch.cuda.get_device_capability(device_index)
    architecture = f"sm_{capability[0]}{capability[1]}"
    report.update(
        numpy_version=np.__version__,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(device_index),
        compute_capability=list(capability),
        device_index=device_index,
        architecture=architecture,
        seed=SEED,
        rtol=RTOL,
        atol=ATOL,
    )
    (out / "nvcc-version.txt").write_bytes(
        subprocess.check_output([find_nvcc(), "--version"], stderr=subprocess.STDOUT)
    )
    case = next(case for case in GPU_CASES if case.name == CASE_NAME)
    arrangement, application, tensors, source_inputs, expected = case_inputs(case)
    report["input_shapes"] = {
        name: list(value.shape) for name, value in source_inputs.items()
    }
    report["output_shape"] = list(expected.shape)
    report["program"] = application.__name__
    report["dtype"] = str(expected.dtype)
    np.savez(out / "oracle.npz", **source_inputs, expected=expected)
    frontend = interpret(arrangement, application, tensors)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            backend="cuda",
            caller="torch",
            kernel_name="nine_opt_cuda_dot_probe",
            num_warps=4,
            max_num_configs=1,
            backend_options={"arch": architecture},
        )
    )
    if compilation.artifact.backend != Target.CUDA:
        raise RuntimeError("The requested CUDA backend was not selected.")
    metadata = compilation.artifact.metadata
    assert metadata["lowering_ir"] == "ssa.Program"
    assert metadata["generation_py_fallback"] is False
    lowered = _program_from_metadata(metadata["ssa"])
    assert ir_to_dict(lowered) == metadata["ssa"]
    _scalar_dot_offsets(lowered)
    assert "ssa.cuda.optimize_schedule" in compilation.pass_trace
    report.update(
        emitted_backend=compilation.artifact.backend.value,
        complete_ir_roundtrip=True,
        pass_trace=list(compilation.pass_trace),
        scalar_dot_decomposition_verified=True,
    )
    (out / "kernel.cu").write_text(
        compilation.artifact.primary_source, encoding="utf-8", newline="\n"
    )
    (out / "frontend.ssa.txt").write_text(
        ssa.render(frontend.program), encoding="utf-8"
    )
    (out / "emitted.ssa.txt").write_text(ssa.render(lowered), encoding="utf-8")
    write_json(out / "emitted-ir.json", ir_to_dict(lowered))
    write_json(out / "artifact-ir.json", metadata["ssa"])
    report["emitted_source_sha256"] = sha256(
        compilation.artifact.primary_source.encode()
    )

    def guarded(value):
        backing = np.full(value.size + 8, GUARD, dtype=value.dtype)
        view = backing[4:-4].reshape(value.shape)
        view[...] = value
        return backing, view

    def check_guards(backings, originals, actual_inputs, label):
        for name in source_inputs:
            np.testing.assert_array_equal(
                backings[name],
                originals[name],
                err_msg=f"{label}: {name} or input guard mutated",
            )
            _assert_equal(
                actual_inputs[name], source_inputs[name], f"{label}: input {name}"
            )
        np.testing.assert_array_equal(
            backings["out"][:4],
            originals["out"][:4],
            err_msg=f"{label}: leading output guard",
        )
        np.testing.assert_array_equal(
            backings["out"][-4:],
            originals["out"][-4:],
            err_msg=f"{label}: trailing output guard",
        )

    initial = dict(source_inputs, out=np.full_like(expected, -123))
    outputs = {}
    for label, program in (
        ("frontend_cpu", frontend.program),
        ("cuda_ssa_cpu", lowered),
    ):
        allocations = {name: guarded(value) for name, value in initial.items()}
        backings = {name: pair[0] for name, pair in allocations.items()}
        values = {name: pair[1] for name, pair in allocations.items()}
        originals = {name: value.copy() for name, value in backings.items()}
        interpret_program(
            program,
            values,
            tensors=compilation.kernel.tensors,
            symbols=compilation.kernel.metadata.get("meta_defaults", {}),
        )
        np.savez(out / f"{label}.npz", **backings)
        check_guards(backings, originals, values, label)
        _assert_equal(values["out"], expected, f"{label} versus NumPy")
        outputs[label] = values["out"].copy()

    host_allocations = {name: guarded(value) for name, value in initial.items()}
    originals = {name: pair[0].copy() for name, pair in host_allocations.items()}
    gpu_backings = {
        name: torch.from_numpy(pair[0]).to(device)
        for name, pair in host_allocations.items()
    }
    gpu_inputs = {
        name: value[4:-4].reshape(initial[name].shape)
        for name, value in gpu_backings.items()
    }
    handle = DEFAULT_COMPILER.materialize(
        compilation, output_dir=out / "build", mode="jit"
    )
    # These Handle fields are defined in compiler/runtime.py and preserve the
    # exact artifact selected by the actual materializer, including its binary.
    assert handle._backend == "cuda"
    assert ir_to_dict(handle._artifact.metadata["ssa"]) == metadata["ssa"]
    report["runtime_backend"] = handle._backend
    report["gpu_launch_attempted"] = True
    handle(**gpu_inputs)
    torch.cuda.synchronize(device)
    actual_backings = {
        name: value.cpu().numpy() for name, value in gpu_backings.items()
    }
    actual_inputs = {
        name: value[4:-4].reshape(initial[name].shape)
        for name, value in actual_backings.items()
    }
    np.savez(out / "cuda_gpu.npz", **actual_backings)
    outputs["cuda_gpu"] = actual_inputs["out"].copy()
    check_guards(actual_backings, originals, actual_inputs, "cuda_gpu")
    for label, reference in (("NumPy", expected), *outputs.items()):
        if label != "cuda_gpu":
            _assert_equal(outputs["cuda_gpu"], reference, f"CUDA GPU versus {label}")
    report["max_abs_error_vs_numpy"] = {
        label: float(
            np.max(np.abs(value.astype(np.float64) - expected.astype(np.float64)))
        )
        for label, value in outputs.items()
    }
    materialized_source = Path(handle._source).read_bytes()
    assert materialized_source == compilation.artifact.primary_source.encode()
    report["materialized_source_sha256"] = sha256(materialized_source)
    report["cuda_binary_sha256"] = sha256(Path(handle._library).read_bytes())
    report["all_input_and_output_guards_unchanged"] = True
    report["inputs_unchanged"] = True
    report["four_way_comparison_passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="New directory outside the repository"
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    if out.is_relative_to(repo):
        parser.error("--out must be outside the frozen repository")
    out.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    report = {
        "status": "RUNNING",
        "case": CASE_NAME,
        "backend_requested": "cuda",
        "expected_head": EXPECTED_HEAD,
        "script_sha256": sha256(script.read_bytes()),
        "python": platform.python_version(),
        "gpu_launch_attempted": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "One scalar float32 CUDA-backend correctness case; not a full suite or performance benchmark.",
            "Four 4x4 M/N output tiles with complete K, including M/N/K tails; no split-K or Tensor Core claim.",
            "Results apply only to the GPU identified by this run; the script alone is not GPU evidence.",
        ],
    }
    shutil.copyfile(script, out / "probe-script.py")
    write_json(out / "report.json", report)
    started = time.monotonic()
    exit_code = 1
    before = None
    try:
        before = source_state(repo)
        write_json(out / "source-before.json", before)
        run_probe(repo, out, args.device, report)
        after = source_state(repo)
        write_json(out / "source-after.json", after)
        assert before == after
        report["source_commit"] = before["head"]
        report["source_unchanged"] = True
        report["status"] = "PASS"
        exit_code = 0
    except Exception as error:  # noqa: BLE001 -- Persist evidence and return nonzero.
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if before is not None:
            try:
                after = source_state(repo)
                write_json(out / "source-after.json", after)
                report["source_unchanged"] = before == after
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.SubprocessError,
            ) as source_error:
                report["source_unchanged"] = False
                report["source_check_error"] = str(source_error)
    finally:
        report.update(
            exit_code=exit_code,
            elapsed_seconds=round(time.monotonic() - started, 3),
            ended_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        report["artifacts"] = {
            str(path.relative_to(out)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()),
            }
            for path in sorted(out.rglob("*"))
            if path.is_file() and path != out / "report.json"
        }
        write_json(out / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "exit_code": exit_code,
                "report": str(out / "report.json"),
            }
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
