"""Three alternating fresh-process pairs; keep failed gates and all samples."""
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'results/nine-confirmation';out.mkdir()
protocol=json.loads((root/'PROTOCOL.json').read_text())['nine']
report={'status':'RUNNING','records':[],'protocol_sha256':hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest()}
for round_id in range(protocol['rounds']):
    for version in (('control','candidate') if round_id%2==0 else ('candidate','control')):
        with (out/f'{round_id}-{version}.log').open('x') as log:
            result=subprocess.run([sys.executable,str(root/'tools/bench_nine.py'),'--source',str(root/version/'nine'),'--output',str(out/f'{round_id}-{version}.json')],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        assert result.returncode==0,(round_id,version,result.returncode)
        print(round_id,version,'finished',flush=True)
    a=json.loads((out/f'{round_id}-control.json').read_text());b=json.loads((out/f'{round_id}-candidate.json').read_text())
    for x,y in zip(a['cases'],b['cases']):
        assert (x['name'],x['trace'])==(y['name'],y['trace'])
        assert x['trace_sha256']==y['trace_sha256']
        ratio=statistics.median(x['times'])/statistics.median(y['times'])
        reductions={mode:1-y['memory'][mode]['peak']/x['memory'][mode]['peak'] for mode in ('warm','cold')}
        report['records'].append({'round':round_id,'name':x['name'],'trace':x['trace'],'target':x['target'],'speedup':ratio,'peak_reduction':reductions,'trace_equal':True,
          'gate_passed':ratio>=1/1.05 and (not x['target'] or min(reductions.values())>=protocol['minimum_target_peak_reduction'])})
    (out/'decision.json').write_text(json.dumps(report,indent=2)+'\n')
report['status']='ACCEPT' if all(r['gate_passed'] for r in report['records']) else 'REJECT'
(out/'decision.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],flush=True)
