#!/usr/bin/env python3
"""Run one frozen-source A100 validation mode and retain its complete evidence."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_SHA = "b5a3206f8351e5a138d16ee13f6d6ef9c620044b"
PROBE = r'''
import importlib.metadata
import json
import platform
import sys
import torch

packages = {}
for name in (
    "torch", "triton", "numpy", "sympy", "pytest", "pytest-cov", "coverage",
    "tilelang", "matplotlib", "pandas", "jupyter", "jupytext", "nbconvert",
    "ipykernel", "ninetoothed",
):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
devices = []
available = torch.cuda.is_available()
if available:
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": prop.name,
            "compute_capability": [prop.major, prop.minor],
            "total_memory_bytes": prop.total_memory,
            "multiprocessor_count": prop.multi_processor_count,
        })
print(json.dumps({
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": {"system": platform.system(), "machine": platform.machine()},
    "packages": packages,
    "torch_import_version": torch.__version__,
    "torch_cuda_build_version": torch.version.cuda,
    "torch_cuda_available": available,
    "torch_float8_e5m2_attribute": hasattr(torch, "float8_e5m2"),
    "cuda_devices": devices,
}, indent=2))
'''


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_command(mode, repo, out):
    common = [sys.executable, "-m", "pytest", "--color=no", "--tb=short", "-ra"]
    if mode == "gpu-report":
        return [
            sys.executable, "scripts/verify_interpreter_gpu.py", "--device", "0",
            "--report", str(out / "interpreter_gpu_validation.json"),
        ], ["interpreter_gpu_validation.json"]
    if mode == "smoke":
        common += [
            "-q", "tests/test_interpreter_gpu.py::"
            "test_cpu_interpreter_matches_actual_triton_gpu[elementwise_float32_aligned]",
        ]
    elif mode == "generation":
        common += ["-q", "tests/test_generation.py"]
    elif mode == "jagged":
        common += ["-q", "tests/test_jagged.py"]
    elif mode == "specialist":
        tests = sorted(repo.glob("tests/test_interpreter*.py"))
        if not tests:
            raise RuntimeError("No interpreter specialist test files found.")
        common += ["-q", *(str(path.relative_to(repo)) for path in tests)]
    else:
        common += [
            "-p", "pytest_cov", "-v", "--doctest-modules", "--cov=ninetoothed",
            f"--cov-report=xml:{out / 'coverage.xml'}",
            f"--cov-report=html:{out / 'coverage-html'}",
        ]
    common.append(f"--junitxml={out / 'junit.xml'}")
    artifacts = ["junit.xml"]
    if mode == "full":
        artifacts += ["coverage.xml", "coverage-html/index.html"]
    return common, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "gpu-report", "specialist", "jagged", "generation", "full"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New evidence directory; existing paths are refused.")
    parser.add_argument("--expected-sha", default=EXPECTED_SHA)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_sha):
        parser.error("--expected-sha must be a full 40-character lowercase Git SHA.")
    repo = args.repo.resolve()
    out = args.out.resolve()
    if not (repo / "src/ninetoothed").is_dir():
        parser.error("--repo must point to the ninetoothed repository root.")
    if out.exists():
        parser.error("--out already exists; choose a new directory.")
    out.mkdir(parents=True, exist_ok=False)

    overrides = {
        "PYTHONPATH": os.pathsep.join((str(repo / "src"), str(repo))),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONUNBUFFERED": "1",
        "TRITON_INTERPRET": "0",
    }
    removed = (
        "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_XDIST_WORKER", "NINETOOTHED_BACKEND",
    )
    env = os.environ.copy()
    env.update(overrides)
    for name in removed:
        env.pop(name, None)
    observed = (
        "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH", "OMP_NUM_THREADS",
        "MKL_NUM_THREADS", "TRITON_CACHE_DIR", "NINETOOTHED_CACHE_DIR", "MPLBACKEND",
        "TORCH_CUDA_ARCH_LIST", "CC", "CXX",
        "TRITON_LIBCUDA_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "PATH",
    )
    manifest = {
        "status": "PREFLIGHT", "mode": args.mode, "started_at_utc": utc_now(),
        "repo": str(repo), "output_directory": str(out),
        "expected_sha": args.expected_sha, "python": sys.version,
        "python_executable": sys.executable, "os": platform.system(),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "environment_overrides": overrides, "environment_unset": list(removed),
        "observed_environment": {name: env.get(name) for name in observed},
        "commands": [], "validation_exit_code": None,
        "notes": [
            "One mode only; no dependency installation, test changes, or automatic retries.",
            "GPU inventory initializes CUDA but does not launch validation kernels.",
            "Existing matmul/addmm FP8 tests have no compute-capability skip.",
            "A specialist/full run overlaps the smoke and GPU-report case coverage.",
            "Artifact hashes exclude this manifest itself; hash it externally when archiving.",
        ],
    }

    def save():
        temporary = out / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(out / "manifest.json")

    def execute(label, argv):
        stdout = out / f"{label}.stdout.log"
        stderr = out / f"{label}.stderr.log"
        entry = {
            "name": label, "argv": argv, "cwd": str(repo),
            "stdout": stdout.name, "stderr": stderr.name,
            "started_at_utc": utc_now(), "returncode": None,
        }
        manifest["commands"].append(entry)
        started = time.monotonic()
        process = None
        save()
        try:
            with stdout.open("xb") as output, stderr.open("xb") as errors:
                process = subprocess.Popen(argv, cwd=repo, env=env, stdout=output, stderr=errors)
                entry["pid"] = process.pid
                save()
                entry["returncode"] = process.wait()
        finally:
            if process is not None:
                entry["returncode"] = process.poll()
            entry["ended_at_utc"] = utc_now()
            entry["elapsed_seconds"] = round(time.monotonic() - started, 3)
            save()
        return entry

    def clean_source(label):
        head = execute(f"{label}-head", ["git", "rev-parse", "HEAD"])
        status = execute(f"{label}-status", ["git", "status", "--porcelain", "--untracked-files=no"])
        actual = (out / head["stdout"]).read_text().strip()
        tracked = (out / status["stdout"]).read_text().strip()
        manifest[f"{label}_source"] = {"head": actual, "tracked_status": tracked}
        if head["returncode"] != 0 or status["returncode"] != 0:
            raise RuntimeError(f"{label}: Git inspection failed.")
        if actual != args.expected_sha or tracked:
            raise RuntimeError(f"{label}: Expected SHA and tracked-clean source check failed.")

    wrapper_exit = 2
    try:
        save()
        clean_source("before")
        probe = execute("environment", [sys.executable, "-c", PROBE])
        if probe["returncode"] != 0:
            raise RuntimeError("Torch environment/device probe failed; inspect environment logs.")
        runtime = json.loads((out / probe["stdout"]).read_text())
        manifest["runtime"] = runtime
        devices = runtime["cuda_devices"]
        if not devices or "A100" not in devices[0]["name"] or devices[0]["compute_capability"] != [8, 0]:
            raise RuntimeError("Visible CUDA device 0 must be an A100 with compute capability 8.0.")
        execute("nvidia-smi", [
            "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,mig.mode.current",
            "--format=csv,noheader,nounits",
        ])
        argv, expected_artifacts = validation_command(args.mode, repo, out)
        manifest["expected_artifacts"] = expected_artifacts
        manifest["status"] = "RUNNING"
        print(f"Starting {args.mode}; evidence: {out}", flush=True)
        run = execute("validation", argv)
        manifest["validation_exit_code"] = run["returncode"]
        clean_source("after")
        missing = [name for name in expected_artifacts if not (out / name).is_file()]
        manifest["missing_artifacts"] = missing
        child_exit = run["returncode"]
        wrapper_exit = child_exit if child_exit >= 0 else 128 - child_exit
        if wrapper_exit == 0 and missing:
            wrapper_exit = 1
        manifest["status"] = "PASS" if wrapper_exit == 0 else "FAIL"
    except KeyboardInterrupt:
        manifest["status"] = "INTERRUPTED"
        manifest["error"] = "Runner interrupted; command records retain child PID and observed exit state."
        wrapper_exit = 130
    except Exception as error:
        manifest["status"] = "ERROR" if manifest["status"] == "RUNNING" else "PREFLIGHT_FAILED"
        manifest["error"] = f"{type(error).__name__}: {error}"
    finally:
        manifest["ended_at_utc"] = utc_now()
        manifest["wrapper_exit_code"] = wrapper_exit
        manifest["artifacts"] = [
            {"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(out.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json", "manifest.json.tmp"}
        ]
        save()
    print(f"{manifest['status']}: {out / 'manifest.json'}", flush=True)
    return wrapper_exit


if __name__ == "__main__":
    raise SystemExit(main())
