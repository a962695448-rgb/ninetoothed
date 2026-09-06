import subprocess,json
from pathlib import Path
p=Path('/data/infinitensor-2026/nine-staging/opt-20260906/ninetoothed')
subprocess.run(['git','clone','--shared','--no-checkout','/data/infinitensor-2026/ninetoothed',str(p)],check=True,timeout=20)
subprocess.run(['git','-c','http.lowSpeedLimit=1000','-c','http.lowSpeedTime=20','fetch','--depth=2','https://github.com/a962695448-rgb/ninetoothed.git','improve-interpreter-matmul-provenance'],cwd=p,check=True,timeout=45)
subprocess.run(['git','checkout','--detach','56f091eb585e94b725a08989e44a63b222b1e3f0'],cwd=p,check=True,timeout=20)
print('BOOT_OK',subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip())
