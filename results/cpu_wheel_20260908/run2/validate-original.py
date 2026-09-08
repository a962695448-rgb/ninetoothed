"""Reproduce the CPU wheel check on Linux; all outputs stay in a new directory.

Usage: python3 validate.py /absolute/repository /absolute/new-output [uv-executable]
Requires Python with venv/pip, Git, and package-index access or cached wheels.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

repo, output = (Path(value).resolve() for value in sys.argv[1:3])
uv = sys.argv[3] if len(sys.argv) == 4 else None
output.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, output / "validate-original.py")
source = output / "source"
work = output / "work"
source.mkdir()
work.mkdir()
environment = dict(os.environ)
for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "NINETOOTHED_BACKEND"):
    environment.pop(key, None)
environment.update(CUDA_VISIBLE_DEVICES="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
                   PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_DEFAULT_TIMEOUT="30")
manifest = {"status": "RUNNING", "commands": [], "environment_overrides": {
    name: environment[name] for name in ("CUDA_VISIBLE_DEVICES", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                                        "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS")}}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save():
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run(name, argv, cwd=work, expected=0):
    started = time.monotonic()
    with (output / f"{name}.stdout.log").open("wb") as stdout, (output / f"{name}.stderr.log").open("wb") as stderr:
        result = subprocess.run([str(arg) for arg in argv], cwd=cwd, env=environment,
                                stdout=stdout, stderr=stderr, timeout=300)
    manifest["commands"].append({"name": name, "argv": [str(arg) for arg in argv],
        "cwd": str(cwd), "returncode": result.returncode, "expected_returncode": expected,
        "elapsed_seconds": round(time.monotonic() - started, 3)})
    save()
    print(f"{name}: exit {result.returncode} (expected {expected})", flush=True)
    assert result.returncode == expected, f"Inspect {output / (name + '.stderr.log')}"


GUARD = '''import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import sysconfig

import ninetoothed

site = Path(sysconfig.get_path("purelib")).resolve()
def prove(stage):
    files = {name: str(Path(module.__file__).resolve())
             for name, module in sys.modules.items()
             if (name == "ninetoothed" or name.startswith("ninetoothed."))
             and getattr(module, "__file__", None)}
    assert files and all(Path(path).is_relative_to(site) for path in files.values())
    assert not any(name.split(".")[0] in ("torch", "triton") for name in sys.modules)
    assert all(importlib.util.find_spec(name) is None for name in ("torch", "triton"))
    installed = {dist.metadata["Name"]: dist.version for dist in metadata.distributions()}
    assert not any(name.lower() in ("torch", "triton") for name in installed)
    print("WHEEL_PROVENANCE " + json.dumps({"stage": stage, "python": sys.version,
        "prefix": sys.prefix, "package": ninetoothed.__file__,
        "distribution_root": str(metadata.distribution("ninetoothed").locate_file("")),
        "installed": installed, "sys_path": sys.path, "loaded_package_files": files,
        "gpu_packages_installed": False, "gpu_modules_loaded": False}), flush=True)

prove("before")
mode, *args = sys.argv[1:]
code = 0
if mode == "pytest":
    import pytest
    code = pytest.main(args)
else:
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
prove("after")
raise SystemExit(code)
'''


try:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest["source_commit"] = head
    archive = subprocess.check_output(["git", "archive", head, "src", "tests", "pyproject.toml",
                                       "README.md", "LICENSE", "docs/cpu_interpreter_demo.py"], cwd=repo)
    manifest["git_archive_sha256"] = digest(archive)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    manifest["source_files"] = {p.relative_to(source).as_posix(): digest(p.read_bytes())
                                for p in source.rglob("*") if p.is_file()}
    save()
    python = work / "venv/bin/python"
    if uv:
        run("00-uv-version", [uv, "--version"])
        run("01-venv", [uv, "venv", "--seed", "--python", sys.executable, work / "venv"])
    else:
        run("01-venv", [sys.executable, "-m", "venv", work / "venv"])
    run("02-build-wheel", [python, "-m", "pip", "wheel", "--verbose", "--no-deps", "--wheel-dir", work / "wheels", "."], cwd=source)
    run("03-cpu-dependencies", [python, "-m", "pip", "install", "numpy>=1.26.4", "sympy>=1.13.0"])
    wheels = list((work / "wheels").glob("ninetoothed-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    manifest["wheel"] = {"path": str(wheel), "sha256": digest(wheel.read_bytes()), "bytes": wheel.stat().st_size}
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        package_files = [name for name in names if name.startswith("ninetoothed/") and not name.endswith("/")]
        assert package_files
        for name in package_files:
            assert z.read(name) == (source / "src" / name).read_bytes(), name
        assert sorted(package_files) == sorted(p.relative_to(source / "src").as_posix()
            for p in (source / "src/ninetoothed").rglob("*") if p.is_file())
        wheel_metadata = z.read(next(name for name in names if name.endswith(".dist-info/METADATA")))
        (output / "wheel-METADATA.txt").write_bytes(wheel_metadata)
        assert b"Requires-Dist: triton>=3.0.0" in wheel_metadata
        manifest["wheel"]["source_package_files_verified"] = len(package_files)
    run("04-install-wheel", [python, "-m", "pip", "install", "--no-deps", wheel])
    shutil.copy2(source / "docs/cpu_interpreter_demo.py", work / "demo.py")
    (work / "guard.py").write_text(GUARD)
    run("05-demo-debug-export", [python, "-I", work / "guard.py", "script", work / "demo.py", "--debug", "--export", work / "replay"])
    run("06-replay", [python, "-I", work / "guard.py", "script", work / "replay/replay.py"])
    run("07-pip-check", [python, "-m", "pip", "check"], expected=1)
    check = (output / "07-pip-check.stdout.log").read_text().strip()
    assert check == "ninetoothed 0.26.0 requires triton, which is not installed.", check
    run("08-test-dependency", [python, "-m", "pip", "install", "pytest"])
    shutil.copytree(source / "tests", work / "tests")
    (work / "docs").mkdir()
    shutil.copy2(source / "docs/cpu_interpreter_demo.py", work / "docs/cpu_interpreter_demo.py")
    assert not (work / "src").exists()
    test_files = ["interpreter_applications", "interpreter_debugger", "interpreter_gpu",
        "interpreter_ssa", "interpreter_step_debugger", "interpreter_default_pipeline",
        "interpreter_demo", "interpreter_matmul", "interpreter_provenance",
        "ssa_application_lowering", "ssa_first_backend_lowering", "ssa_pass_pipeline",
        "ssa_program_domain_regressions", "ssa_validation", "ir_immutability", "kernel_ir"]
    run("09-cpu-regression", [python, work / "guard.py", "pytest", "--import-mode=importlib",
        "-q", "-ra", "--tb=short", "--color=no", *[f"tests/test_{name}.py" for name in test_files],
        "-k", "not test_cpu_interpreter_matches_actual_triton_gpu", f"--junitxml={output / 'junit.xml'}"])
    run("10-installed-packages", [python, "-m", "pip", "list", "--format=json"])
    site = next((work / "venv/lib").glob("python*/site-packages"))
    with zipfile.ZipFile(wheel) as z:
        for name in package_files:
            assert (site / name).read_bytes() == z.read(name), name
    manifest["installed_package_files_verified_after_execution"] = len(package_files)
    manifest["status"] = "PASS"
except BaseException as exc:
    manifest["status"] = "FAIL"
    manifest["error"] = repr(exc)
    raise
finally:
    manifest["artifacts"] = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p.read_bytes())}
        for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
    save()
