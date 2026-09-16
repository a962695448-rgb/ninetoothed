"""Validate uploaded source identities, then run GPU checks in sequence."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time

upload=Path('/data/cuda-profile-20260916')
root=Path('/data/profiling-followup-20260916');root.mkdir()
expected={'confirm_cuda.py':'d0f5dd7edab5dcccb42eaaa42f851262d8d98dff044abfe974b1b88835f73e3f',
 'row_policy_4090d.hpp':'9f4bea3233106a446d650e8d4b9f96dfae22571abc2eed8a1cce0b4b5215253d',
 'CUDA_CONFIRMATION_PROTOCOL.json':'22dfb7be0eb05cfd2b77780867ec7744a2bbed8c8e4335e0dcea1a9951116b69',
 'expressions.py':'93ac38db5ac70f6ec17248b3bb55c8b293c75c0422b0e2806f34d5cbef0bc884',
 'test_interpreter_expression_plans.py':'78b0fc0c940e2c06c4edadff004d7bcc822722b14df3f21e5f5de65767f268df'}
for name,sha in expected.items():assert hashlib.sha256((upload/name).read_bytes()).hexdigest()==sha,name
previous=Path('/data/testing-20260916')
control=previous/'sources/cuda/03_hadamard_tc/a962695448-rgb'
cuda=root/'cuda';nine=root/'nine'
ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.ruff_cache')
shutil.copytree(control,cuda,ignore=ignore)
shutil.copy2(upload/'row_policy_4090d.hpp',cuda/'include/row_policy.hpp')
shutil.copytree(previous/'nine-final',nine,ignore=ignore)
shutil.copy2(upload/'expressions.py',nine/'src/ninetoothed/interpreter/expressions.py')
shutil.copy2(upload/'test_interpreter_expression_plans.py',nine/'tests/test_interpreter_expression_plans.py')
env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONPATH=str(nine/'src'));env['PATH']='/usr/local/cuda/bin:'+env['PATH']
jobs=[('cuda-confirm',root,['python3',str(upload/'confirm_cuda.py'),'--control',str(control),'--candidate',str(cuda),
       '--control-build',str(previous/'build'),'--output',str(root/'cuda-confirm'),'--protocol',str(upload/'CUDA_CONFIRMATION_PROTOCOL.json')],900),
      ('nine-gpu',nine,['python3','scripts/verify_interpreter_gpu.py','--report',str(root/'nine-gpu.json')],300),
      ('nine-expression-tests',nine,['python3','-m','pytest','-q','tests/test_interpreter_expression_plans.py','--junitxml='+str(root/'nine-expression-tests.xml')],120)]
summary={'status':'RUNNING','jobs':[],'upload_sha256':expected}
def save():
 (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
save()
for name,cwd,command,timeout in jobs:
 start=time.monotonic()
 with (root/f'{name}.log').open('x') as log:
  proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=124
 summary['jobs'].append({'name':name,'returncode':code,'seconds':time.monotonic()-start,'command':command});save()
summary['nine_source_sha256']={str(f.relative_to(nine)):hashlib.sha256(f.read_bytes()).hexdigest()
 for f in sorted(nine.rglob('*.py')) if not {'__pycache__','.pytest_cache'}.intersection(f.relative_to(nine).parts)}
summary['status']='PASS' if all(j['returncode']==0 for j in summary['jobs']) else 'FAIL';save()
files=list(root.glob('*.json'))+list(root.glob('*.xml'))+list(root.glob('*.log'))+list((root/'cuda-confirm').glob('*.json'))
export={str(f.relative_to(root)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'text':f.read_bytes().decode()} for f in files}
payload=json.dumps(export,ensure_ascii=True);(root/'FOLLOWUP_EXPORT.txt').write_text(payload)
print(json.dumps({'status':summary['status'],'export_bytes':len(payload.encode()),'export_sha256':hashlib.sha256(payload.encode()).hexdigest()}),flush=True)
