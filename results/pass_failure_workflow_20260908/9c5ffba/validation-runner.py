"""Record frozen CPU regression, installed replay and isolated Sphinx validation."""

import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone

repo = Path(sys.argv[1]).resolve()
expected = sys.argv[2]
out = Path(sys.argv[3]).resolve()
out.mkdir(parents=True, exist_ok=False)
(out / "validation-runner.py").write_bytes(Path(__file__).read_bytes())
scope = ["src", "tests", "scripts", "docs", "pyproject.toml", "requirements.txt", "conftest.py", "README.md", "LICENSE", ".github"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=repo)


def state():
    assert git("rev-parse", "HEAD").decode().strip() == expected
    assert not git("ls-files", "--others", "--exclude-standard", "--", *scope)
    paths = git("ls-files", "--", *scope).decode().splitlines()
    request = "".join(f"{expected}:{name}\n" for name in paths).encode()
    stream = io.BytesIO(subprocess.check_output(["git", "cat-file", "--batch"], input=request, cwd=repo))
    hashes = {}
    for name in paths:
        oid, kind, size = stream.readline().decode().split()
        assert kind == "blob"
        reference = stream.read(int(size))
        assert stream.read(1) == b"\n"
        actual = (repo / name).read_bytes()
        normalized = actual
        lfs = reference.startswith(b"version https://git-lfs.github.com/spec/v1\n")
        if lfs and actual != reference:
            lines = dict(line.split(" ", 1) for line in reference.decode().splitlines())
            assert hashlib.sha256(actual).hexdigest() == lines["oid"].split(":", 1)[1], name
            assert len(actual) == int(lines["size"]), name
        elif actual != reference:
            actual.decode("utf-8")
            reference.decode("utf-8")
            assert b"\0" not in actual
            normalized = actual.replace(b"\r\n", b"\n")
        assert lfs or reference == normalized, name
        hashes[name] = {"blob": oid, "sha256": hashlib.sha256(normalized).hexdigest(), "lfs_resolved": lfs and actual != reference}
    return hashes


before = state()
assert all(importlib.util.find_spec(name) is None for name in ("torch", "triton"))
env = dict(os.environ, PYTHONPATH=str(repo / "src"), CUDA_VISIBLE_DEVICES="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MPLBACKEND="Agg")
for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "NINETOOTHED_BACKEND"):
    env.pop(name, None)
manifest = {
    "status": "RUNNING", "source_commit": expected,
    "scope": "CPU regression and installed-wheel replay only; GPU tests explicitly deselected. No Torch or Triton installed.",
    "python": sys.version, "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "packages": {name: importlib.metadata.version(name) for name in ("numpy", "sympy", "pytest", "ruff", "sphinx")},
    "torch_available": False, "triton_available": False,
    "before_source": before, "commands": [],
}


def save():
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run(name, argv, cwd=repo, environ=env):
    start = time.monotonic()
    with (out / (name + ".stdout.log")).open("wb") as stdout, (out / (name + ".stderr.log")).open("wb") as stderr:
        result = subprocess.run([str(arg) for arg in argv], cwd=cwd, env=environ, stdout=stdout, stderr=stderr)
    manifest["commands"].append({"name": name, "argv": [str(arg) for arg in argv], "cwd": str(cwd), "exit_code": result.returncode, "seconds": round(time.monotonic()-start, 3)})
    save()
    print(json.dumps(manifest["commands"][-1]), flush=True)
    return result.returncode


files = [
    "test_interpreter_applications", "test_interpreter_debugger", "test_interpreter_failure_workflow",
    "test_interpreter_gpu", "test_interpreter_ssa", "test_interpreter_step_debugger",
    "test_interpreter_default_pipeline", "test_interpreter_demo", "test_interpreter_matmul",
    "test_interpreter_provenance", "test_ssa_application_lowering", "test_ssa_first_backend_lowering",
    "test_ssa_pass_pipeline", "test_ssa_program_domain_regressions", "test_ssa_validation",
    "test_ir_immutability", "test_kernel_ir",
]
run("ruff-check", [sys.executable, "-m", "ruff", "check"])
run("ruff-format", [sys.executable, "-m", "ruff", "format", "--check"])
run("style", [sys.executable, "scripts/check_contributing_style.py"])
run("demo-style", [sys.executable, "scripts/check_contributing_style.py", "docs/cpu_interpreter_demo.py"])
run("cpu", [sys.executable, "-m", "pytest", "-q", "--color=no", "-ra", "--tb=short", *[f"tests/{name}.py" for name in files], "-k", "not test_cpu_interpreter_matches_actual_triton_gpu", f"--junitxml={out / 'cpu.junit.xml'}"])
run("demo", [sys.executable, "docs/cpu_interpreter_demo.py", "--debug", "--export", out / "demo-case"])
run("replay", [sys.executable, out / "demo-case/replay.py"])

# An isolated archive lets autosummary generate pages without changing the checkout.
snapshot = out / "source"
snapshot.mkdir()
archive = git("archive", expected, *[p for p in scope if (repo / p).exists()])
with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
    bundle.extractall(snapshot, filter="data")
for name, item in before.items():
    if item["lfs_resolved"]:
        (snapshot / name).write_bytes((repo / name).read_bytes())
doc_env = dict(env, PYTHONPATH=str(snapshot / "src"))
if len(sys.argv) > 4:
    run("sphinx", [sys.argv[4], "-m", "sphinx", "-W", "--keep-going", "-b", "html", snapshot / "docs/source", out / "html"], cwd=snapshot, environ=doc_env)
else:
    manifest["documentation"] = "Deferred to a separately recorded documentation environment; the earlier no-Torch Sphinx failure is retained."

uv = Path("/home/xxl/.local/bin/uv")
assert run("wheel-build", [uv, "build", "--python", sys.executable, "--wheel", "--out-dir", out / "wheels", snapshot], cwd=out) == 0
wheel, = (out / "wheels").glob("*.whl")
manifest["wheel"] = {"filename": wheel.name, "bytes": wheel.stat().st_size, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
assert run("wheel-venv", [uv, "venv", "--python", sys.executable, out / "venv"], cwd=out) == 0
installed_python = out / "venv/bin/python"
assert run("wheel-dependencies", [uv, "pip", "install", "--python", installed_python, "numpy=="+manifest["packages"]["numpy"], "sympy=="+manifest["packages"]["sympy"]], cwd=out) == 0
assert run("wheel-install", [uv, "pip", "install", "--python", installed_python, "--no-deps", wheel], cwd=out) == 0
probe = out / "installed_probe.py"
probe.write_text('''import hashlib, importlib.util, json
from pathlib import Path
import ninetoothed
from ninetoothed.interpreter import failure
base = Path(ninetoothed.__file__).parent
assert "site-packages" in str(base)
assert all(importlib.util.find_spec(name) is None for name in ("torch", "triton"))
source = Path(__file__).parent / "source/src/ninetoothed"
count = 0
for path in source.rglob("*.py"):
    target = base / path.relative_to(source)
    assert target.read_bytes() == path.read_bytes(), str(path)
    count += 1
print(json.dumps({"installed_path": str(base), "files_verified": count, "torch": False, "triton": False}))
''')
run("installed-probe", [installed_python, "-I", probe], cwd=out)
run("installed-demo", [installed_python, "-I", snapshot / "docs/cpu_interpreter_demo.py", "--debug", "--export", out / "installed-case"], cwd=out)
run("installed-replay", [installed_python, "-I", out / "installed-case/replay.py"], cwd=out)
manifest["after_source"] = state()
assert manifest["after_source"] == before
manifest["status"] = "PASS" if all(item["exit_code"] == 0 for item in manifest["commands"]) else "FAIL"
manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest["artifacts"] = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.is_file() and p.name != "manifest.json"}
save()
print(json.dumps({"status": manifest["status"], "path": str(out), "cpu": (out / "cpu.stdout.log").read_text()[-1500:]}), flush=True)
raise SystemExit(0 if manifest["status"] == "PASS" else 1)
