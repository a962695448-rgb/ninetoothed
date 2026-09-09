"""Check a slow case with retained caches, then run the unchanged A100 acceptance."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import traceback

root = Path.home() / "infinitensor-2026"
repo = root / "ninetoothed-a100-20260909"
run = root / "runs/nine-a100-20260909-warm"
tools = root / "tools/nine-a100-20260909"
expected = "3fc27e0399d82b02be5407b9542c67763c4631a7"
run.mkdir(parents=True, exist_ok=False)
state_path = run / "sequence.json"
state = {"status": "STARTING", "source_commit": expected, "pid": os.getpid(), "steps": [],
         "started_at_utc": datetime.now(timezone.utc).isoformat(), "cache_policy": "Retained default ~/.ninetoothed; no new empty cache for full regression.",
         "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(run / "sequence.py").write_bytes(Path(__file__).read_bytes())


def save():
    pending = state_path.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2) + "\n")
    pending.replace(state_path)


env = dict(os.environ, PATH=str(root / ".venv/bin") + ":/usr/local/cuda/bin:/home/vipuser/miniconda3/bin:/usr/bin:/bin",
           CUDA_HOME="/usr/local/cuda", CUDA_VISIBLE_DEVICES="0", MPLBACKEND="Agg", MAX_JOBS="1", NVCC_THREADS="1", OMP_NUM_THREADS="2",
           TRITON_LIBCUDA_PATH=str(root / ".driver-libs"), LIBRARY_PATH=str(root / ".driver-libs"),
           PYTHONPATH=str(repo / "src") + ":" + str(repo), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
for name in ("NINETOOTHED_CACHE_DIR", "TRITON_CACHE_DIR", "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_XDIST_WORKER"):
    env.pop(name, None)
python = root / ".venv/bin/python"


def execute(name, argv, timeout=None):
    record = {"name": name, "argv": list(map(str, argv)), "started_at_utc": datetime.now(timezone.utc).isoformat()}
    state.update(status="RUNNING", active_stage=name)
    state["steps"].append(record)
    save()
    with (run / (name + ".controller.log")).open("xb") as log:
        process = subprocess.Popen(record["argv"], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        record["pid"] = process.pid
        save()
        try:
            record["exit_code"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            os.killpg(process.pid, signal.SIGINT)
            try:
                record["exit_code"] = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                record["exit_code"] = process.wait(timeout=10)
    record["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    if record["exit_code"] != 0:
        raise RuntimeError(f"Stage {name} did not pass; inspect its preserved log before continuing.")


try:
    save()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == expected
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip()
    assert (Path.home() / ".ninetoothed").is_dir()
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    (run / "canary").mkdir()
    execute("canary", [python, "-m", "pytest", "-q", "--color=no", "--tb=short", "-ra",
                       "tests/test_conv2d.py::test[4-64-16-16-512-3-3-padding3-dtype0-0.001-0.001-cuda]",
                       f"--junitxml={run / 'canary/junit.xml'}"], timeout=90)
    for mode, name in (("full", "full"), ("gpu-report", "gpu-differential")):
        execute(name, [python, tools / "a100_final_runner.py", "--mode", mode, "--repo", repo,
                       "--out", run / name, "--expected-sha", expected])
    execute("cuda-dot", [python, tools / "cuda_dot_probe.py", "--repo", repo, "--out", run / "cuda-dot"])
    state["status"] = "PASS"
except Exception as error:
    state.update(status="FAIL", error=str(error), traceback=traceback.format_exc())
state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
save()
print(json.dumps(state), flush=True)
raise SystemExit(0 if state["status"] == "PASS" else 1)
