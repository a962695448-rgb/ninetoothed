"""Verify automatic failure capture on real dot/transpose default pass pipelines."""

import hashlib
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

base = Path(sys.argv[1]).resolve()
source = base / "source"
sys.path[:0] = [str(source / "src"), str(source)]
from ninetoothed.interpreter.debugger import check_passes
from tests.test_interpreter_provenance import (
    real_linalg_case, _real_pipeline_case, _inject_decomposed_fault,
)

out = base / "pipeline-replays"
out.mkdir(exist_ok=False)
(out / "runner.py").write_bytes(Path(__file__).read_bytes())
installed_python = base / "venv/bin/python"
source_commit = json.loads((base / "manifest.json").read_text())["source_commit"]
records = []
for backend in ("triton", "cuda"):
    for kind in ("dot", "transpose"):
        case = real_linalg_case.__wrapped__(SimpleNamespace(param=kind))
        kind, kernel, pipeline, context, inputs, expected, options = _real_pipeline_case(backend, case)
        before = {name: value.copy() for name, value in inputs.items()}
        name = f"diagnostic.inject_{kind}_fault"
        def checks(wrong):
            result = []
            for pass_ in pipeline.passes:
                result.append((pass_.name, partial(pass_.run, context=context)))
                if pass_.name == "ssa.decompose_linalg":
                    result.append((name, partial(_inject_decomposed_fault, kind=kind, name=name, wrong=wrong)))
            return result
        directory = out / f"{backend}-{kind}"
        report = check_passes(kernel.frontend_program, checks(True), inputs, **options,
                              failure_dir=directory, seed=923 if kind == "dot" else None)
        assert not report.passed and report.first_bad_pass == name
        assert report.export_error is None and report.reproducer == directory
        assert report.localization is not None
        assert report.localization.operation.lane is not None
        for key in inputs:
            np.testing.assert_array_equal(inputs[key], before[key])
        replay = subprocess.run([str(installed_python), "-I", str(directory / "replay.py")],
                                cwd=out, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                                text=True, capture_output=True, timeout=60)
        (out / f"{backend}-{kind}.stdout.log").write_text(replay.stdout)
        (out / f"{backend}-{kind}.stderr.log").write_text(replay.stderr)
        assert replay.returncode == 0, replay.stdout + replay.stderr
        healthy = out / f"{backend}-{kind}-healthy"
        control = check_passes(kernel.frontend_program, checks(False), inputs, **options,
                               failure_dir=healthy)
        assert control.passed and not healthy.exists()
        records.append({"backend_pipeline": backend, "application": kind,
                        "source_commit": source_commit, "replay_exit_code": replay.returncode,
                        "negative_control": "PASS", "inputs_unchanged": True,
                        "localization": asdict(report.localization), "first_bad_pass": report.first_bad_pass,
                        "fault_type": "Deliberately injected into real default-pass SSA; not a discovered historical bug."})
        print(json.dumps(records[-1]), flush=True)
manifest = {"status": "PASS", "source_commit": source_commit,
            "scope": "Four automatic exports replayed using the isolated installed wheel, plus four repaired negative controls. CPU execution only, no GPU imports or launches.",
            "cases": records,
            "artifacts": {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.rglob("*") if p.is_file()}}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
