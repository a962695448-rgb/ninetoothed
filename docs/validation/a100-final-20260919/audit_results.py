"""Check complete A100 outcomes against collected IDs and JUnit counts."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.results
summary = json.loads((root / 'SUMMARY.json').read_text())
assert summary['status'] == 'PASS', summary
assert summary['source_commit'] == 'a86bea9d32a43e122322acd52c0168b19682ffa2'
assert summary['frozen_files_verified_before_and_after'] == 209
assert {job['name'] for job in summary['jobs']} == {
    'gpu-differential', 'traced-targets', 'full-suite'}
assert all(job['returncode'] == 0 and not job['timed_out'] for job in summary['jobs'])
env = json.loads((root / 'ENVIRONMENT.json').read_text())
assert 'A100' in env['devices'][0]['name']
gpu = json.loads((root / 'gpu-differential.json').read_text())
assert gpu['status'] == 'PASS' and gpu['passed_cases'] == gpu['total_cases'] == 15
assert gpu['rtol'] == gpu['atol'] == 1e-3
assert gpu['integer_and_bool_comparison'] == 'exact'
assert len(gpu['passed_programs']) >= 3
assert all(case['status'] == 'PASS' for case in gpu['cases'])
trace = json.loads((root / 'traced-targets.json').read_text())
assert trace['status'] == 'PASS' and len(trace['cases']) == 3
assert all(case['status'] == 'PASS' for case in trace['cases'])

nodeids = json.loads((root / 'full-suite-nodeids.json').read_text())
assert len(nodeids) == len(set(nodeids))
reports = defaultdict(list)
for line in (root / 'full-suite-outcomes.jsonl').read_text().splitlines():
    report = json.loads(line)
    reports[report['nodeid']].append(report)
assert set(reports) == set(nodeids), 'Missing or extra test outcomes'
counts = Counter()
skips = []
for nodeid in nodeids:
    phases = reports[nodeid]
    assert any(item['when'] == 'teardown' for item in phases), nodeid
    assert not any(item['outcome'] == 'failed' for item in phases), nodeid
    skipped = [item for item in phases if item['outcome'] == 'skipped']
    if skipped:
        counts['skipped'] += 1
        skips.append({'nodeid': nodeid, 'reason': skipped[0]['reason']})
    else:
        assert sum(item['when'] == 'call' and item['outcome'] == 'passed'
                   for item in phases) == 1, nodeid
        counts['passed'] += 1
xml = ET.parse(root / 'full-suite.xml').getroot()
cases = list(xml.iter('testcase'))
assert len(cases) == len(nodeids)
assert not list(xml.iter('failure')) and not list(xml.iter('error'))
assert sum(case.find('skipped') is not None for case in cases) == counts['skipped']
result = {'status': 'AUDIT_PASS', 'source_commit': summary['source_commit'],
          'device': env['devices'][0]['name'], 'collected': len(nodeids),
          'counts': dict(counts), 'skips': skips, 'formal_gpu_cases': 15,
          'traced_targets': 3, 'frozen_files': 209}
with args.output.open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result))
