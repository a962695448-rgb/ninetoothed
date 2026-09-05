# Independent fault replay evidence

This is a deliberately injected constant change from `2` to `3`, not a newly
discovered historical compiler bug. The interpreter core remains at source base
`5b377252cc4452b5ccc48c46ff1ae07a4e5e0e8a`; `manifest.json` identifies the exact
modified example, documentation and test files validated on top of that base.

All four recorded processes returned 0 with CUDA hidden on the RTX 4090 host:

- Targeted CPU tests: 39 passed, including the 8 cases for 4 applications across
  the Triton and CUDA default pass pipelines. These are CPU interpretations, not
  8 actual GPU executions.
- Sphinx HTML build succeeded.
- The scripted debugger demo exported both reference and candidate bundles.
- A separate process loaded those bundles, checked the NumPy reference and
  reproduced the difference at `entry:0:arith.constant`.

To replay the archived bundle from a compatible source checkout:

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
python -m zipfile -e \
  results/interpreter_debug_replay_20260905/injected_fault_reproducer.zip \
  /tmp/nine-injected-fault-replay
python /tmp/nine-injected-fault-replay/replay.py
```

Use a new extraction directory. Exit 0 means the expected injected discrepancy
was detected and the reference output passed its independent NumPy check. The
test also substitutes a correct candidate and requires replay to return nonzero.
The saved comparison does not rerun the pass sequence or claim a new A100 result.

Raw logs, JUnit, and the complete replay bundle retain SHA-256 entries in the
manifest. Source paths and command arguments describe the actual recorded run;
adjust paths and preserve fresh evidence for subsequent runs.
