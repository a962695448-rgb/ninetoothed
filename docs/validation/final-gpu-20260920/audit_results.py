"""Audit final targeted GPU coverage without changing acceptance results."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

r = Path(sys.argv[1])
s = json.loads((r / 'SUMMARY.json').read_text())
assert s['status'] == 'PASS' and s['ecc_after'] == '0'
assert s['source_commit'] == '150f5268977174164db383a75254f541aad14ffa'
assert s['frozen_files_verified_before_and_after'] == 215
assert len(s['jobs']) == 4 and all(x['returncode'] == 0 and not x['timed_out'] for x in s['jobs'])
g = json.loads((r / 'gpu-differential.json').read_text())
assert g['status'] == 'PASS' and g['passed_cases'] == g['total_cases'] == 15
assert g['rtol'] == g['atol'] == 1e-3 and g['integer_and_bool_comparison'] == 'exact'
assert 'A100' in g['gpu_name']
t = json.loads((r / 'traced-targets.json').read_text())
assert t['status'] == 'PASS' and len(t['cases']) == 3
a = json.loads((r / 'integer-alias-gpu.json').read_text())
expected = {(alias, kind, backend, mode) for alias in ['i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64']
            for kind in ['scalar', 'cast'] for backend in ['triton', 'cuda'] for mode in ['jit', 'aot']}
assert a['status'] == 'PASS' and not a['cpu_only']
assert len(a['records']) == 64
assert {(x['alias'], x['kind'], x['backend'], x['mode']) for x in a['records']} == expected
for record in a['records']:
    assert record['status'] == 'PASS'
    assert len(record['values']) == (2 if record['kind'] == 'scalar' else 1)
    assert all(x['status'] == 'PASS' for x in record['values'])
    if record['mode'] == 'aot':
        assert record['aot_reload_passed']
ids = json.loads((r / 'targeted-nodeids.json').read_text())
rows = [json.loads(x) for x in (r / 'targeted-outcomes.jsonl').read_text().splitlines()]
assert all(x['outcome'] == 'passed' for x in rows)
passed = [x['nodeid'] for x in rows if x['when'] == 'call']
assert len(ids) == len(set(ids)) == len(passed) == 127 and set(ids) == set(passed)
assert all(any(x['nodeid'] == node and x['when'] == 'teardown' for x in rows) for node in ids)
xml = ET.parse(r / 'targeted.xml').getroot()
assert len(list(xml.iter('testcase'))) == 127
assert not list(xml.iter('failure')) and not list(xml.iter('error')) and not list(xml.iter('skipped'))
print('AUDIT_PASS: 15 formal GPU, 3 traced targets, 64 alias configurations/96 values, 32 AOT reload configurations, 127 regressions')
