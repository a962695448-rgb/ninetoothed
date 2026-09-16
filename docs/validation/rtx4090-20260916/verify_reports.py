"""Verify exact, non-overlapping coverage of the collected test IDs across two runs."""

import argparse
import collections
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ALLOWED_SKIP_REASONS = (
    "multi-device testing requires at least 2 devices",
    "Triton multi-context testing requires at least 2 CUDA devices",
)


def results_from_junit(path, index):
    results, incomplete = {}, []
    for case in ET.parse(path).getroot().iter("testcase"):
        key = (case.get("classname"), case.get("name"))
        if None in key:
            incomplete.append(dict(case.attrib))
            continue
        if key not in index or index[key] in results:
            raise ValueError("Unexpected or duplicate JUnit test: " + str(key))
        if case.find("failure") is not None or case.find("error") is not None:
            raise ValueError("Cannot combine a failing test: " + str(key))
        skipped = case.find("skipped")
        status = "passed"
        if skipped is not None:
            reason = skipped.get("message", "") + " " + (skipped.text or "")
            if not any(allowed in reason for allowed in ALLOWED_SKIP_REASONS):
                raise ValueError("Unexpected skip reason: " + reason)
            status = "skipped"
        results[index[key]] = {"status": status, "report": str(path)}
    return results, incomplete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.results.resolve()
    selection = json.loads((root / "nine-completion/remaining.json").read_text())
    first = json.loads((root / "nine-parallel/run-summary.json").read_text())
    last = json.loads((root / "nine-remainder-serial/run-summary.json").read_text())
    if last["status"] != "PASS_REMAINDER":
        raise ValueError("The remaining tests have not passed.")
    for key in ("source_commit", "source_sha256", "hardware", "versions"):
        if first[key] != last[key]:
            raise ValueError("Run identity differs: " + key)
    index = {
        (item["classname"], item["name"]): item["nodeid"]
        for item in selection["collection"]
    }
    if (
        len(index) != len(selection["collection"])
        or len(index) != selection["collected"]
    ):
        raise ValueError("The collection is not unique or complete.")
    initial, incomplete = results_from_junit(
        root / "nine-parallel/full-pytest.xml", index
    )
    final, final_incomplete = results_from_junit(
        root / "nine-remainder-serial/remaining-pytest.xml", index
    )
    if final_incomplete or initial.keys() & final.keys():
        raise ValueError("The completion contains partial or duplicate results.")
    combined = initial | final
    if set(combined) != set(index.values()) or set(final) != set(
        selection["remaining"]
    ):
        raise ValueError(
            "The combined report does not exactly cover the collected IDs."
        )
    result = {
        "status": "PASS_COMPLETE_COLLECTION",
        "source_commit": first["source_commit"],
        "counts": dict(
            collections.Counter(item["status"] for item in combined.values())
        ),
        "collected": len(index),
        "hardware": first["hardware"],
        "versions": first["versions"],
        "excluded_incomplete_entries": incomplete,
        "scope": "Complete coverage by exact test ID across two runs; not an uninterrupted full-suite run.",
        "tests": combined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({k: result[k] for k in ("status", "collected", "counts")}, indent=2)
    )


if __name__ == "__main__":
    main()
