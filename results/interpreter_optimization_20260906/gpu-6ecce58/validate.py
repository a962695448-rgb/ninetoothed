import os,json,subprocess,time,hashlib
from pathlib import Path
root=Path('/data/infinitensor-2026/nine-staging/opt-20260906')
repo=root/'validated-6ecce58'
py='/data/infinitensor-2026/.venv/bin/python'
expected='6ecce58da28bb9709aa35fc6c25c1f361aff736f'
def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
assert git('rev-parse','HEAD')==expected
assert not git('status','--porcelain','--untracked-files=no')
env=dict(os.environ,PYTHONPATH=str(repo/'src')+':'+str(repo),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TRITON_INTERPRET='0',OMP_NUM_THREADS='2',CUDA_HOME='/usr/local/cuda')
env['NINETOOTHED_CACHE_DIR']=str(root/'ninetoothed-cache')
env['TRITON_CACHE_DIR']=str(root/'triton-cache')
env['PATH']='/usr/local/cuda/bin:'+str(Path(py).parent)+':'+env.get('PATH','')
commands=[('triton',[py,'scripts/verify_interpreter_gpu.py','--device','0','--report',str(root/'triton-report.json')]),('cuda',[py,str(root/'cuda_probe.py'),'--repo',str(repo),'--out',str(root/'cuda-dot'),'--device','0'])]
m={'source_commit':expected,'before_head':git('rev-parse','HEAD'),'before_status':git('status','--porcelain','--untracked-files=no'),'status':'RUNNING','stages':{}}
(root/'validation.json').write_text(json.dumps(m,indent=2))
for name,argv in commands:
 started=time.time()
 with (root/(name+'.stdout.log')).open('w') as out,(root/(name+'.stderr.log')).open('w') as err:
  try:r=subprocess.run(argv,cwd=repo,env=env,stdout=out,stderr=err,timeout=180);code=r.returncode
  except subprocess.TimeoutExpired:code=124
 m['stages'][name]={'argv':argv,'exit':code,'seconds':time.time()-started}
 (root/'validation.json').write_text(json.dumps(m,indent=2))
m.update(after_head=git('rev-parse','HEAD'),after_status=git('status','--porcelain','--untracked-files=no'))
m['status']='PASS' if all(x['exit']==0 for x in m['stages'].values()) and m['after_head']==expected and not m['after_status'] else 'FAIL'
(root/'validation.json').write_text(json.dumps(m,indent=2))
print(json.dumps(m))
