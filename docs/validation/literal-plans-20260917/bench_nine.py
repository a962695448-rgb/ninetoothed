"""Measure prepared frontend programs without profiling their timed calls."""
import argparse
import gc
import hashlib
import json
import pickle
import platform
import sys
import time
import tracemalloc
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();sys.path.insert(0,str(args.source/'src'))
import numpy as np
import ninetoothed.language as ntl
from ninetoothed import Tensor,interpret
from ninetoothed.interpreter import expressions,interpret_program

protocol=json.loads((Path(__file__).resolve().parents[1]/'PROTOCOL.json').read_text())


def matrix16(a,b,out):return a.tile((4,16)),b.tile((16,4)),out.tile((4,4))
def matrix32(a,b,out):return a.tile((4,32)),b.tile((32,4)),out.tile((4,4))
def matrix8(a,b,out):return a.tile((4,8)),b.tile((8,4)),out.tile((4,4))
def dot(a,b,out):out=ntl.dot(a,b)
def vectors(x,out):return x.tile((128,)),out.tile((128,))
def padded(x,out):return x.pad(((2,7),)).tile((128,)),out.tile((128,))
def affine(x,out):out=x*2+1
def rows(x,out):return x.tile((1,256)),out.tile((1,256))
def softmax(x,out):
    e=ntl.exp(x-ntl.max(x,1)[:,None])
    out=e/ntl.sum(e,1)[:,None]


def workloads():
    rng=np.random.default_rng(9172033)
    for m,k,n,arrange,target in [(7,16,7,matrix16,True),(9,32,11,matrix32,True),(5,17,9,matrix32,True),(3,8,5,matrix8,False)]:
        a=rng.integers(-8,9,(m,k)).astype(np.float32)/8;b=rng.integers(-8,9,(k,n)).astype(np.float32)/8
        yield f'matmul_{m}x{k}x{n}',target,arrange,dot,{'a':a,'b':b},a@b,0
    x=rng.integers(-8,9,4099).astype(np.float32)/8
    yield 'vector_4099',False,vectors,affine,{'x':x},x*2+1,0
    yield 'padded_4099',False,padded,affine,{'x':x},np.pad(x,(2,7))*2+1,0
    x=rng.uniform(-1,1,(13,129)).astype(np.float32);e=np.exp(x-x.max(1,keepdims=True))
    yield 'softmax_13x129',False,rows,softmax,{'x':x},e/e.sum(1,keepdims=True),float('-inf')


def clear_caches():
    expressions._compiled_expression.cache_clear()
    expressions._numpy_dtype_from_name.cache_clear()


def trace_hash(trace):return hashlib.sha256(pickle.dumps(trace,protocol=4)).hexdigest()


report={'status':'RUNNING','source':str(args.source),'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'source_sha256':{str(p.relative_to(args.source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((args.source/'src').rglob('*.py'))},'cases':[],'trace_checks':[]}
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as output:
    def save():
        output.seek(0);output.write(json.dumps(report,indent=2)+'\n');output.truncate();output.flush()
    save()
    for name,target,arrange,apply,inputs,expected,other in workloads():
        tensors=tuple(Tensor(a.ndim,name=n,dtype='float32',other=other) for n,a in inputs.items())+(Tensor(expected.ndim,name='out',dtype='float32'),)
        kernel=interpret(arrange,apply,tensors,backend='triton');data=dict(inputs,out=np.empty_like(expected));before={n:a.copy() for n,a in inputs.items()}
        def check(result):
            np.testing.assert_allclose(result.outputs['out'],expected,rtol=1e-4,atol=1e-4)
            for n,a in before.items():np.testing.assert_array_equal(data[n],a)
        # Every target also gets a full trace check, even where tracing is not timed.
        traced=interpret_program(kernel.program,data,tensors=kernel.tensors,symbols=kernel.meta,trace=True)
        check(traced);full_hash=trace_hash(traced.trace);report['trace_checks'].append({'name':name,'sha256':full_hash,'events':len(traced.trace)});del traced
        for trace in ((False,) if target else (False,True)):
            def run():return interpret_program(kernel.program,data,tensors=kernel.tensors,symbols=kernel.meta,trace=trace)
            result=run();check(result);digest=trace_hash(result.trace);del result
            samples=[]
            for _ in range(protocol['samples']):
                gc.collect();start=time.perf_counter()
                for __ in range(protocol['calls_per_sample']):result=run();del result
                samples.append((time.perf_counter()-start)/protocol['calls_per_sample'])
            cold=[]
            for _ in range(3):
                clear_caches();gc.collect();start=time.perf_counter();result=run();cold.append(time.perf_counter()-start);check(result);del result
            memory={}
            for mode in ('warm','cold'):
                if mode=='cold':clear_caches()
                gc.collect();tracemalloc.start();result=run();retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                check(result);assert trace_hash(result.trace)==digest
                if trace:assert digest==full_hash
                memory[mode]={'retained':retained,'peak':peak};del result
            report['cases'].append({'name':name,'target':target,'trace':trace,'times':samples,'cold_times':cold,'memory':memory,'trace_sha256':digest})
            save();print(name,trace,'PASS',flush=True)
    report['status']='PASS';save()
