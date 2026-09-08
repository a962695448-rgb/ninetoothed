"""Build frozen docs with real CPU Torch and verify its failure-capture adapter."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

base = Path(sys.argv[1]).resolve()
source = base / "source"
out = base / "documentation"
out.mkdir(exist_ok=False)
(out / "runner.py").write_bytes(Path(__file__).read_bytes())
sys.path.insert(0, str(source / "src"))
expected = json.loads((base / "manifest.json").read_text())["source_commit"]


def hashes():
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source / "src").rglob("*.py")}


before = hashes()
environment = dict(os.environ, PYTHONPATH=str(source / "src"), CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
argv = [sys.executable, "-m", "sphinx", "-W", "--keep-going", "-b", "html", str(source / "docs/source"), str(out / "html")]
started = time.monotonic()
with (out / "sphinx.stdout.log").open("wb") as stdout, (out / "sphinx.stderr.log").open("wb") as stderr:
    result = subprocess.run(argv, cwd=source, env=environment, stdout=stdout, stderr=stderr)
manifest = {"source_commit": expected, "argv": argv, "cwd": str(source), "exit_code": result.returncode,
            "seconds": round(time.monotonic()-started, 3), "packages": {name: importlib.metadata.version(name) for name in ("sphinx", "pydata-sphinx-theme", "torch", "numpy", "sympy")},
            "torch_cuda_version": torch.version.cuda, "triton_available": importlib.util.find_spec("triton") is not None,
            "mocked_imports": False, "before_source": before}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({key: value for key, value in manifest.items() if key != "before_source"}), flush=True)

from ninetoothed import Tensor, interpret
from ninetoothed.interpreter.debugger import compare_programs, load_reproducer
spec = importlib.util.spec_from_file_location("verified_demo", source / "docs/cpu_interpreter_demo.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
kernel = interpret(module.arrangement, module.application,
                   (Tensor(1, name="x", dtype="float32"), Tensor(1, name="out", dtype="float32")))
x = torch.arange(21, dtype=torch.float32)[::3]
output = torch.full_like(x, -731)
original_x, original_output = x.clone(), output.clone()
report = compare_programs(kernel.program, module.deliberately_bad_pass(kernel.program), {"x": x, "out": output},
                          tensors=kernel.tensors, failure_dir=out / "torch-cpu-case", seed=None)
assert not report.equal and report.export_error is None
torch.testing.assert_close(x, original_x)
torch.testing.assert_close(output, original_output)
_, inputs, _ = load_reproducer(report.reproducer / "reference")
assert inputs["x"].strides == (x.stride(0) * x.element_size(),)
replay_argv = [str(base / "venv/bin/python"), "-I", str(report.reproducer / "replay.py")]
replayed = subprocess.run(replay_argv, cwd=out, env=environment, text=True, capture_output=True, timeout=60)
(out / "torch-replay.stdout.log").write_text(replayed.stdout)
(out / "torch-replay.stderr.log").write_text(replayed.stderr)
assert replayed.returncode == 0, replayed.stdout + replayed.stderr
manifest["torch_cpu_capture"] = {"status": "PASS", "replay_argv": replay_argv, "replay_exit_code": replayed.returncode, "strided_inputs_preserved": True, "input_output_unchanged": True}
manifest["after_source"] = hashes()
assert manifest["after_source"] == before
manifest["html_pages"] = len(list((out / "html").rglob("*.html")))
manifest["status"] = "PASS" if result.returncode == 0 else "FAIL"
manifest["artifacts"] = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"status": manifest["status"], "html_pages": manifest["html_pages"], "torch_cpu_capture": manifest["torch_cpu_capture"]}), flush=True)
raise SystemExit(result.returncode)
