"""Keep all three fresh-process pairs and enforce the frozen timing gates."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess
import sys

root=Path(__file__).resolve().parents[1];out=root/'results/nine-confirmation';out.mkdir()
protocol=json.loads((root/'PROTOCOL.json').read_text())['nine']
report={'status':'RUNNING','protocol_sha256':hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest(),'records':[],'rounds':[]}
for round_id in range(protocol['rounds']):
    for version in (('control','candidate') if round_id%2==0 else ('candidate','control')):
        with (out/f'{round_id}-{version}.log').open('x') as log:
            job=subprocess.run([sys.executable,str(root/'tools/bench_nine.py'),'--source',str(root/version/'nine'),'--output',str(out/f'{round_id}-{version}.json')],stdout=log,stderr=subprocess.STDOUT,timeout=900)
        assert job.returncode==0,(round_id,version)
        print(round_id,version,'complete',flush=True)
    a=json.loads((out/f'{round_id}-control.json').read_text());b=json.loads((out/f'{round_id}-candidate.json').read_text())
    assert len(a['cases'])==len(b['cases']);targets=[]
    for x,y in zip(a['cases'],b['cases']):
        assert x['name']==y['name'] and x['event_digest']==y['event_digest']
        speedup=statistics.median(x['times'])/statistics.median(y['times'])
        if x['target']:targets.append(speedup)
        report['records'].append({'round':round_id,'name':x['name'],'target':x['target'],'speedup':speedup,'peak_reduction':1-y['peak_bytes']/x['peak_bytes'],'events_equal':True,'gate_passed':speedup>=1/1.05})
    geomean=math.exp(statistics.mean(math.log(v) for v in targets))
    report['rounds'].append({'round':round_id,'target_geomean_speedup':geomean,'gate_passed':geomean>=protocol['minimum_target_geomean_speedup']})
    (out/'decision.json').write_text(json.dumps(report,indent=2)+'\n')
report['status']='ACCEPT' if all(x['gate_passed'] for x in report['records']+report['rounds']) else 'REJECT'
(out/'decision.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],flush=True)
