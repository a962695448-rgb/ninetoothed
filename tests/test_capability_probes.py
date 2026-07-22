import dataclasses
import json

import pytest

import tests.capabilities as capabilities
from tests.capabilities.model import CapabilityResult
from tests.capabilities.runner import resolve_probe, run_python_probe


def resolved_probe():
    return CapabilityResult(supported=True)


def _protocol_source(payload):
    return f"import json; print(json.dumps({payload!r}))"


def test_capabilities_package_exports_public_protocol():
    assert capabilities.__all__ == (
        "CapabilityResult",
        "resolve_probe",
        "run_python_probe",
    )
    assert capabilities.CapabilityResult is CapabilityResult
    assert capabilities.resolve_probe is resolve_probe
    assert capabilities.run_python_probe is run_python_probe


def test_capability_result_is_frozen():
    result = CapabilityResult(supported=True, reason="")

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.supported = False


def test_capability_result_reason_defaults_to_empty():
    assert CapabilityResult(supported=True) == CapabilityResult(
        supported=True,
        reason="",
    )


def test_supported_probe_uses_final_nonempty_json_line():
    source = "\n".join(
        (
            "import json",
            'print("compiler diagnostic")',
            'print("")',
            'print(json.dumps({"status": "supported"}))',
        )
    )

    result = run_python_probe("sample", source)

    assert result == CapabilityResult(supported=True, reason="")


def test_unavailable_probe_preserves_reason():
    result = run_python_probe(
        "sample",
        _protocol_source(
            {"status": "unavailable", "reason": "The compiler is unavailable."}
        ),
    )

    assert result == CapabilityResult(
        supported=False,
        reason="sample: The compiler is unavailable.",
    )


def test_error_probe_raises_runtime_error():
    source = _protocol_source(
        {"status": "error", "reason": "The probe implementation failed."}
    )

    with pytest.raises(RuntimeError, match="probe implementation failed"):
        run_python_probe("sample", source)


def test_probe_timeout_raises_runtime_error():
    with pytest.raises(RuntimeError, match="timed out"):
        run_python_probe("sample", "import time; time.sleep(10)", timeout=0.01)


@pytest.mark.parametrize(
    "source",
    (
        'print("not json")',
        f"print({json.dumps(json.dumps(['supported']))!r})",
        _protocol_source({"status": "unknown"}),
        _protocol_source({"status": "unavailable"}),
    ),
)
def test_invalid_probe_output_raises_runtime_error(source):
    with pytest.raises(RuntimeError, match="invalid"):
        run_python_probe("sample", source)


def test_abnormal_probe_exit_is_unavailable():
    source = "\n".join(
        (
            "import sys",
            'sys.stderr.write("native compiler aborted\\n")',
            "raise SystemExit(17)",
        )
    )

    result = run_python_probe("sample", source)

    assert not result.supported
    assert "code 17" in result.reason
    assert "native compiler aborted" in result.reason


def test_resolve_probe_executes_fully_qualified_reference():
    result = resolve_probe("tests.test_capability_probes:resolved_probe")

    assert result == CapabilityResult(supported=True)
