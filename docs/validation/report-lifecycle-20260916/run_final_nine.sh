python3 - <<'PY'
from pathlib import Path
import json,hashlib,shutil,subprocess,os,time
p=Path('/data/testing-20260916');assert json.loads((p/'results/summary.json').read_text())['status']=='PASS'
d=p/'nine-final';shutil.copytree(p/'sources/nine',d,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
files={'scripts/verify_interpreter_gpu.py':'af82ea9e87623c0f4d4019905bb68de5bce37916e8103b884c7c2f50a9a1d353','tests/test_interpreter_gpu_report.py':'35941c3a4a2355f429dac7a1fa6da1a068d73def3d52287baf29cab7a74c90ca'}
for name,sha in files.items():
 f=Path('/data')/Path(name).name;assert hashlib.sha256(f.read_bytes()).hexdigest()==sha;shutil.copy2(f,d/name)
manifest={str(f.relative_to(d)):hashlib.sha256(f.read_bytes()).hexdigest() for f in d.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1');env['PATH']='/usr/local/cuda/bin:'+env['PATH']
jobs=[]
for name,args in [('nine-final-gpu',['scripts/verify_interpreter_gpu.py','--report',str(p/'results/nine-final-gpu.json')]),('nine-final-report-tests',['-m','pytest','-q','tests/test_interpreter_gpu_report.py','--junitxml='+str(p/'results/nine-final-report-tests.xml')])]:
 command=['python3',*args];start=time.monotonic()
 with (p/'results'/f'{name}.log').open('w') as log:r=subprocess.run(command,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 jobs.append({'name':name,'returncode':r.returncode,'command':command,'seconds':time.monotonic()-start})
for name,sha in manifest.items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==sha
(p/'results/nine-final-summary.json').write_text(json.dumps({'status':'PASS' if all(j['returncode']==0 for j in jobs) else 'FAIL','jobs':jobs,'source_sha256':manifest,'sources_unchanged_during_run':True},indent=2)+'\n')
export={str(f.relative_to(p)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'text':f.read_bytes().decode('utf8')} for f in sorted((p/'results').iterdir()) if f.is_file()}
payload=json.dumps(export,ensure_ascii=True);(p/'TESTING_FINAL_EXPORT.txt').write_text(payload)
print(json.dumps({'export_bytes':len(payload.encode()),'export_sha256':hashlib.sha256(payload.encode()).hexdigest(),'jobs':jobs}),flush=True)
PY
