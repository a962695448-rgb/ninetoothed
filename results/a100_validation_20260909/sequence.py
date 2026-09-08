"""Run GPU differential, independent CUDA dot and complete regression sequentially."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import traceback

root = Path.home() / "infinitensor-2026"
repo = root / "ninetoothed-a100-20260909"
run = root / "runs/nine-a100-20260909"
tools = root / "tools/nine-a100-20260909"
expected = "3fc27e0399d82b02be5407b9542c67763c4631a7"
run.mkdir(parents=True, exist_ok=False)
state_path = run / "sequence.json"
state = {"status": "STARTING", "source_commit": expected, "pid": os.getpid(), "steps": [], "started_at_utc": datetime.now(timezone.utc).isoformat()}


def save():
    pending = state_path.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2) + "\n")
    pending.replace(state_path)


env = dict(os.environ, PATH=str(root / ".venv/bin") + ":/usr/local/cuda/bin:/home/vipuser/miniconda3/bin:/usr/bin:/bin",
           CUDA_HOME="/usr/local/cuda", CUDA_VISIBLE_DEVICES="0", MPLBACKEND="Agg", MAX_JOBS="1", NVCC_THREADS="1", OMP_NUM_THREADS="2",
           TRITON_LIBCUDA_PATH=str(root / ".driver-libs"), LIBRARY_PATH=str(root / ".driver-libs"),
           NINETOOTHED_CACHE_DIR=str(run / "compile-cache"), TRITON_CACHE_DIR=str(run / "triton-cache"),
           PYTHONPATH=str(repo / "src") + ":" + str(repo), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
python = root / ".venv/bin/python"
try:
    save()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == expected
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip()
    stages = [
        ("gpu-differential", [python, tools / "a100_final_runner.py", "--mode", "gpu-report", "--repo", repo, "--out", run / "gpu-differential", "--expected-sha", expected]),
        ("cuda-dot", [python, tools / "cuda_dot_probe.py", "--repo", repo, "--out", run / "cuda-dot"]),
        ("full", [python, tools / "a100_final_runner.py", "--mode", "full", "--repo", repo, "--out", run / "full", "--expected-sha", expected]),
    ]
    for name, argv in stages:
        record = {"name": name, "argv": list(map(str, argv)), "started_at_utc": datetime.now(timezone.utc).isoformat()}
        state.update(status="RUNNING", active_stage=name)
        state["steps"].append(record)
        save()
        with (run / (name + ".controller.log")).open("xb") as log:
            process = subprocess.Popen(record["argv"], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            record["pid"] = process.pid
            save()
            record["exit_code"] = process.wait()
        record["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        if record["exit_code"] != 0:
            raise RuntimeError(f"Stage {name} failed; retain its evidence before any retry.")
    state["status"] = "PASS"
except Exception as error:
    state.update(status="FAIL", error=str(error), traceback=traceback.format_exc())
state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
save()
print(json.dumps(state), flush=True)
raise SystemExit(0 if state["status"] == "PASS" else 1)
