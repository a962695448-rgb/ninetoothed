"""Confirm dtype reuse on the preregistered generalization workloads."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess
import sys

root=Path(__file__).resolve().parents[1];out=root/'results/nine-confirmation';out.mkdir()
protocol=json.loads((root/'PROTOCOL.json').read_text())['nine'];records=[];rounds=[]
for round_id in range(3):
    for version in (('control','candidate') if round_id%2==0 else ('candidate','control')):
        command=[sys.executable,str(root/'tools/bench_nine.py'),'--source',str(root/version/'nine'),'--output',str(out/f'{round_id}-{version}.json')]
        with (out/f'{round_id}-{version}.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        assert result.returncode==0,(round_id,version,result.returncode)
        print(round_id,version,'done',flush=True)
    a=json.loads((out/f'{round_id}-control.json').read_text());b=json.loads((out/f'{round_id}-candidate.json').read_text())
    targets=[]
    for x,y in zip(a['cases'],b['cases']):
        assert (x['name'],x['trace'])==(y['name'],y['trace']);assert x['trace_sha256']==y['trace_sha256']
        ratio=statistics.median(x['times'])/statistics.median(y['times'])
        if x['target']:targets.append(ratio)
        records.append({'round':round_id,'name':x['name'],'trace':x['trace'],'target':x['target'],'speedup':ratio,
                        'gate_passed':ratio>=(1.05 if x['target'] else 1/1.05),'trace_equal':True})
    geomean=math.exp(statistics.mean(math.log(v) for v in targets));rounds.append({'round':round_id,'target_geomean':geomean,'gate_passed':geomean>=1.10})
decision={'status':'ACCEPT' if all(r['gate_passed'] for r in records+rounds) else 'REJECT',
          'records':records,'rounds':rounds,'protocol_sha256':hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest(),
          'benchmark_sha256':hashlib.sha256((root/'tools/bench_nine.py').read_bytes()).hexdigest()}
(out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n');print(decision['status'],flush=True)
