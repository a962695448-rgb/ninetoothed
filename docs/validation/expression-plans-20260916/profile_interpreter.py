"""Measure prepared-program execution separately from frontend compilation."""
import argparse
import cProfile
import gc
import hashlib
import json
import pickle
import platform
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--measure-only', action='store_true')
args = parser.parse_args()
sys.path.insert(0, str(args.source / 'src'))
import numpy as np
import ninetoothed.language as ntl
from ninetoothed import Tensor, interpret
from ninetoothed.interpreter import interpret_program


def vectors(x, out):
    return x.tile((128,)), out.tile((128,))


def repeated_arithmetic(x, out):
    accumulator = x
    for i in range(8):
        accumulator = accumulator * 0.5 + 0.25
    out = accumulator


def rows(x, out):
    return x.tile((1, 128)), out.tile((1, 128))


def softmax(x, out):
    shifted = x - ntl.max(x, 1)[:, None]
    numerator = ntl.exp(shifted)
    out = numerator / ntl.sum(numerator, 1)[:, None]


def matrices(a, b, out):
    return a.tile((4, 16)), b.tile((16, 4)), out.tile((4, 4))


def matmul(a, b, out):
    out = ntl.dot(a, b)


def cases():
    rng = np.random.default_rng(20260916)
    x = rng.uniform(-1, 1, (16384,)).astype(np.float32)
    expected = x.copy()
    for _ in range(8):
        expected = expected * np.float32(.5) + np.float32(.25)
    yield 'elementwise_loop_16384', vectors, repeated_arithmetic, {'x': x}, expected
    x = rng.uniform(-1, 1, (32, 128)).astype(np.float32)
    numerator = np.exp(x-x.max(axis=1, keepdims=True))
    yield 'softmax_32x128', rows, softmax, {'x': x}, numerator/numerator.sum(axis=1, keepdims=True)
    a = rng.integers(-4, 5, (12, 16)).astype(np.float32)/4
    b = rng.integers(-4, 5, (16, 12)).astype(np.float32)/4
    yield 'matmul_12x16x12', matrices, matmul, {'a': a, 'b': b}, a@b


args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('x') as output:
    report = {'status': 'RUNNING', 'python': sys.version, 'platform': platform.platform(),
              'numpy': np.__version__, 'source': str(args.source), 'cases': [],
              'source_sha256': {str(f.relative_to(args.source)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted((args.source/'src').rglob('*.py'))}}

    def save():
        output.seek(0); output.write(json.dumps(report, indent=2)+'\n'); output.truncate(); output.flush()

    save()
    for name, arrangement, application, inputs, expected in cases():
        descriptors = tuple(Tensor(value.ndim, name=key, dtype='float32', other=0) for key, value in inputs.items())
        descriptors += (Tensor(expected.ndim, name='out', dtype='float32'),)
        started = time.perf_counter()
        kernel = interpret(arrangement, application, descriptors, backend='triton')
        preparation_seconds = time.perf_counter()-started
        arrays = dict(inputs, out=np.empty_like(expected))
        before = {key: value.copy() for key, value in inputs.items()}
        for tracing in (False, True):
            def run():
                arrays['out'].fill(-999)
                return interpret_program(kernel.program, arrays, tensors=kernel.tensors,
                                         symbols=kernel.meta, trace=tracing)

            result = run()
            np.testing.assert_allclose(result.outputs['out'], expected, rtol=1e-4, atol=1e-4)
            digest = hashlib.sha256(pickle.dumps(result.trace, protocol=4)).hexdigest()
            event_count = len(result.trace)
            del result
            times = []
            for _ in range(args.repeats):
                gc.collect()
                started = time.perf_counter(); result = run(); times.append(time.perf_counter()-started)
                np.testing.assert_allclose(result.outputs['out'], expected, rtol=1e-4, atol=1e-4)
                assert hashlib.sha256(pickle.dumps(result.trace, protocol=4)).hexdigest() == digest
                del result
            gc.collect(); tracemalloc.start()
            result = run()
            retained, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            trace_serialized_bytes = len(pickle.dumps(result.trace, protocol=4))
            del result
            hot = []
            if not args.measure_only:
                profiler = cProfile.Profile(); profiler.enable(); result = run(); profiler.disable(); del result
                stats = pstats.Stats(profiler)
                for (file, line, function), (primitive, calls, self_seconds, cumulative_seconds, callers) in sorted(stats.stats.items(), key=lambda item:item[1][3], reverse=True)[:35]:
                    hot.append({'file':file,'line':line,'function':function,'calls':calls,'self_seconds':self_seconds,'cumulative_seconds':cumulative_seconds})
            for key, value in before.items():np.testing.assert_array_equal(arrays[key], value)
            report['cases'].append({'name':name,'trace':tracing,'preparation_seconds':preparation_seconds,
                    'runtime_seconds':times,'trace_events':event_count,'trace_pickle_bytes':trace_serialized_bytes,
                    'trace_digest':digest,'tracemalloc_retained_bytes':retained,'tracemalloc_peak_bytes':peak,
                    'hot_functions':hot,'correctness':'PASS'})
            save(); print(name, 'trace='+str(tracing), 'seconds='+str(times), 'peak='+str(peak), flush=True)
    report['status']='PASS';save()
