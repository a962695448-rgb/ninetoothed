"""Independently verify all downloaded bytes and the successful full A100 result."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile

root = Path(__file__).resolve().parents[1] / "experiments/a100-nine-20260909-success"
archive = root / "results.zip"
expected_hash = "d2b2a9fde1b9324647e7c0a77b1ec1ecf848708fe5ae90dc81e4d6c89b99bbfb"
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected_hash
out = root / "raw"
out.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as bundle:
    record = json.loads(bundle.read("download-manifest.json"))
    assert set(bundle.namelist()) == set(record["files"]) | {"download-manifest.json"}
    assert len(bundle.namelist()) == len(set(bundle.namelist()))
    for name in bundle.namelist():
        target = (out / name).resolve()
        assert target.is_relative_to(out.resolve())
        data = bundle.read(name)
        if name in record["files"]:
            expected = record["files"][name]
            assert len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
full = out / "default-full/full"
manifest = json.loads((full / "manifest.json").read_text())
assert manifest["status"] == "PASS" and manifest["validation_exit_code"] == manifest["wrapper_exit_code"] == 0
assert manifest["expected_sha"] == "3fc27e0399d82b02be5407b9542c67763c4631a7"
for side in ("before_source", "after_source"):
    assert manifest[side]["head"] == manifest["expected_sha"] and manifest[side]["tracked_status"] == ""
tree = ET.parse(full / "junit.xml").getroot()
cases = list(tree.iter("testcase"))
suites = [suite.attrib for suite in tree.iter("testsuite")]
assert len(cases) == sum(int(suite["tests"]) for suite in suites) == 705
assert not any(case.find("failure") is not None or case.find("error") is not None for case in cases)
skipped = [case for case in cases if case.find("skipped") is not None]
assert len(skipped) == 2
gpu_cases = [case.get("name") for case in cases if case.get("classname") == "tests.test_interpreter_gpu" and case.get("name", "").startswith("test_cpu_interpreter_matches_actual_triton_gpu[")]
assert len(gpu_cases) == 15
coverage = ET.parse(full / "coverage.xml").getroot().attrib
summary = {"status": "VERIFIED_PASS", "source_commit": manifest["expected_sha"], "files_verified": len(record["files"]),
           "archive_sha256": expected_hash, "tests": 705, "passed": 703, "skipped": [{"name": case.get("name"), "reason": case.find("skipped").get("message")} for case in skipped],
           "gpu_test_cases_from_junit": gpu_cases, "suites": suites, "coverage": coverage,
           "runtime": manifest["runtime"], "console_summary": (full / "validation.stdout.log").read_text().strip().splitlines()[-1],
           "canary": json.loads((out / "bounded-canary/sequence.json").read_text())}
(root / "verified_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key not in ("runtime", "canary", "gpu_test_cases_from_junit")}, indent=2))
