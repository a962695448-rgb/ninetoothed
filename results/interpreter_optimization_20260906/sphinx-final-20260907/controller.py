import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
repo, out = a.repo.resolve(), a.output.resolve()
out.mkdir(parents=True, exist_ok=False)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot():
    files = sorted((repo / "docs/source").rglob("*.rst")) + sorted((repo / "docs/source").rglob("*.py"))
    files += [repo / "src/ninetoothed/visualization.py"]
    return {x.relative_to(repo).as_posix(): digest(x) for x in files}

before = snapshot()
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
argv = [sys.executable, "-m", "sphinx", "-W", "--keep-going", "-b", "html", str(repo / "docs/source"), str(out / "html")]
env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(repo / "src") + os.pathsep + str(repo), PYTHONDONTWRITEBYTECODE="1")
started = time.time()
with (out / "build.stdout.log").open("wb") as stdout, (out / "build.stderr.log").open("wb") as stderr:
    result = subprocess.run(argv, cwd=repo, env=env, stdout=stdout, stderr=stderr)
after = snapshot()
packages = {}
for name in ("sphinx", "pydata-sphinx-theme", "matplotlib", "numpy", "sympy"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
artifacts = {x.name: {"bytes": x.stat().st_size, "sha256": digest(x)} for x in (out / "build.stdout.log", out / "build.stderr.log")}
html = {x.relative_to(out).as_posix(): {"bytes": x.stat().st_size, "sha256": digest(x)} for x in sorted((out / "html").rglob("*.html"))}
valid = result.returncode == 0 and before == after and "html/cpu_interpreter.html" in html
report = {"status": "PASS" if valid else "FAIL", "exit_code": result.returncode, "source_commit": head,
          "scope": "CPU-only Sphinx HTML build of recorded working docs and lazy GUI imports; no GPU/test-matrix execution",
          "argv": argv, "MPLBACKEND": "Agg", "python": sys.version, "packages": packages,
          "source_sha256_before": before, "source_sha256_after": after, "sources_unchanged_during_build": before == after,
          "elapsed_seconds": round(time.time() - started, 3), "artifacts": artifacts, "html": html,
          "controller_sha256": digest(Path(__file__))}
(out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": report["status"], "exit_code": result.returncode, "html_pages": len(html), "seconds": report["elapsed_seconds"]}))
raise SystemExit(0 if valid else 1)
