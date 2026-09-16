"""Alternate isolated old/new processes using the frozen workload and gate."""
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'results/nine-confirmation';out.mkdir(exist_ok=False)
protocol=json.loads((root/'PROTOCOL.json').read_text())
records=[]
for round_id in range(3):
    order=('control','candidate') if round_id%2==0 else ('candidate','control')
    for version in order:
        target=out/f'{round_id}-{version}.json'
        command=[sys.executable,str(root/'tools/profile_interpreter.py'),'--source',str(root/version/'nine'),
                 '--output',str(target),'--repeats','5','--measure-only']
        with (out/f'{round_id}-{version}.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        assert result.returncode==0,(round_id,version,result.returncode)
        print(round_id,version,'complete',flush=True)
    old=json.loads((out/f'{round_id}-control.json').read_text())
    new=json.loads((out/f'{round_id}-candidate.json').read_text())
    assert old['status']==new['status']=='PASS'
    for a,b in zip(old['cases'],new['cases']):
        assert (a['name'],a['trace'])==(b['name'],b['trace'])
        assert a['trace_digest']==b['trace_digest']
        ratio=statistics.median(a['runtime_seconds'])/statistics.median(b['runtime_seconds'])
        memory_ratio=b['tracemalloc_peak_bytes']/a['tracemalloc_peak_bytes']
        target=a['name'] in protocol['nine_gate']['target_cases']
        passed=ratio>=(1.10 if target else 1/1.05) and (not a['trace'] or memory_ratio<=1.05)
        records.append({'round':round_id,'name':a['name'],'trace':a['trace'],'speedup':ratio,
                        'peak_memory_ratio':memory_ratio,'trace_equal':True,'gate_passed':passed})
summary={'status':'ACCEPT' if all(r['gate_passed'] for r in records) else 'REJECT',
         'protocol_sha256':hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest(),
         'profiler_sha256':hashlib.sha256((root/'tools/profile_interpreter.py').read_bytes()).hexdigest(),
         'records':records}
(out/'decision.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary['status'],flush=True)
