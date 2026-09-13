"""Measure diagnostic analysis against the frozen preceding implementation."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time

repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
expected = sys.argv[3]
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == expected
out.mkdir(parents=True, exist_ok=False)
baseline_commit = '3757a1d83df78693a679020e2610d504f83ef43d'
raw = subprocess.check_output(['git', 'show', baseline_commit+':src/ninetoothed/interpreter/localization.py'], cwd=repo)
baseline_file = out/'baseline_localization.py'
baseline_file.write_bytes(raw)
spec = importlib.util.spec_from_file_location('nine_baseline_localization', baseline_file)
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
sys.path[:0] = [str(repo/'src'), str(repo)]
from ninetoothed.interpreter import localization
from ninetoothed.interpreter.debugger import OperationDifference, _same_snapshot
from tests.test_interpreter_trace_index import mapped_constants

def core_mismatch(result):
    mismatch, issues = result
    return (None if mismatch is None else (mismatch.reference_index, mismatch.candidate_index,
             mismatch.reference_result, mismatch.candidate_result)), issues

def core_slice(result):
    return [asdict(event) for event in result.events], result.boundaries

def paired(name, old, new, normalize):
    expected_result = normalize(old())
    assert expected_result == normalize(new())
    times = {'before': [], 'after': []}
    for repeat in range(7):
        order = (('before', old), ('after', new)) if repeat % 2 == 0 else (('after', new), ('before', old))
        for label, function in order:
            started = time.perf_counter_ns()
            value = function()
            times[label].append((time.perf_counter_ns()-started)/1e6)
            assert normalize(value) == expected_result
    medians = {key: statistics.median(values) for key, values in times.items()}
    return {'name': name, 'milliseconds': times, 'median_ms': medians,
            'before_over_after': medians['before']/medians['after'], 'diagnostics_equal': True}

report = {'source_commit': expected, 'baseline_commit': baseline_commit,
    'baseline_sha256': hashlib.sha256(raw).hexdigest(),
    'candidate_sha256': hashlib.sha256((repo/'src/ninetoothed/interpreter/localization.py').read_bytes()).hexdigest(),
    'python': sys.version, 'platform': platform.system()+' '+platform.machine(),
    'started_at_utc': datetime.now(timezone.utc).isoformat(), 'records': [],
    'scope': 'CPU diagnostic analysis microbenchmarks only; trace construction, code generation and GPU kernels are excluded. Seven alternating pairs after one warm-up per implementation. No universal speedup claim.'}
(out/'benchmark-runner.py').write_bytes(Path(__file__).read_bytes())
for count in (256, 1024, 4096):
    source, candidate, first, second = mapped_constants(count)
    args = source, candidate, first, second, _same_snapshot, 0, 0
    record = paired('result-mapping', lambda: baseline.compare_mapped_results(*args), lambda: localization.compare_mapped_results(*args), core_mismatch)
    record['operations_and_mappings'] = count
    report['records'].append(record)
    for index, label in ((8, 'early-observation'), (count-1, 'last-observation')):
        event = second.trace[index]
        observation = OperationDifference(event.program_id, event.location, event.opcode, next(iter(event.results)), event.iteration, event.lane)
        args2 = candidate, second.trace, observation
        record = paired(label, lambda: baseline.backward_slice(*args2), lambda: localization.backward_slice(*args2), core_slice)
        record.update(operations=count, observation_index=index)
        report['records'].append(record)
    (out/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'completed_size': count, 'medians': [record['median_ms'] for record in report['records'][-3:]]}), flush=True)
report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
(out/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
