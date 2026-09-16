"""Run the new correctness checks and export their exact raw text files."""
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=False)
manifest = json.loads((ROOT / 'INPUT_MANIFEST.json').read_text())


def verify():
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


verify()
import torch
import triton
import numpy
import sympy

save('environment.json', {
    'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
    'cuda': torch.version.cuda, 'triton': triton.__version__, 'numpy': numpy.__version__,
    'sympy': sympy.__version__, 'gpu': torch.cuda.get_device_name(),
    'capability': list(torch.cuda.get_device_capability()),
    'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
    'nvcc': subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True),
    'input_manifest_sha256': hashlib.sha256((ROOT / 'INPUT_MANIFEST.json').read_bytes()).hexdigest(),
    'runner_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
})
environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', MAX_JOBS='1')
environment['PATH'] = '/usr/local/cuda/bin:' + environment['PATH']
cuda = ROOT / 'sources/cuda/03_hadamard_tc/a962695448-rgb'
nine = ROOT / 'sources/nine'
jobs = [
    ('cuda-context', cuda, ['scripts/verify_execution_context.py', '--build-directory', str(ROOT/'build'), '--json', str(OUT/'cuda-context.json')], 600),
    ('nine-gpu', nine, ['scripts/verify_interpreter_gpu.py', '--report', str(OUT/'nine-gpu.json')], 600),
    ('nine-report-tests', nine, ['-m', 'pytest', '-q', 'tests/test_interpreter_gpu_report.py', '--junitxml='+str(OUT/'nine-report-tests.xml')], 120),
]
summary = {'status': 'RUNNING', 'jobs': []}
save('summary.json', summary)
for name, directory, arguments, timeout in jobs:
    start = time.monotonic()
    command = [sys.executable, *arguments]
    with (OUT / (name + '.log')).open('w') as log:
        try:
            result = subprocess.run(command, cwd=directory, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    summary['jobs'].append({'name': name, 'command': command, 'cwd': str(directory),
                            'returncode': code, 'seconds': time.monotonic()-start})
    save('summary.json', summary)
verify()
summary['sources_unchanged_during_run'] = True
summary['status'] = 'PASS' if all(x['returncode'] == 0 for x in summary['jobs']) else 'FAIL'
save('summary.json', summary)
export = {str(p.relative_to(ROOT)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                    'text': p.read_bytes().decode('utf8')}
          for p in sorted(OUT.iterdir()) if p.is_file()}
payload = json.dumps(export, ensure_ascii=True)
(ROOT/'TESTING_EXPORT.txt').write_text(payload)
print(json.dumps({'status': summary['status'], 'export_bytes': len(payload.encode()),
                  'export_sha256': hashlib.sha256(payload.encode()).hexdigest()}), flush=True)
