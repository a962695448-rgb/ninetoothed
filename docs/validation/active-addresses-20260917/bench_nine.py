"""Measure active-address recording and unchanged public interpreter traces."""
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
from ninetoothed.interpreter import interpret_program,expressions
from ninetoothed.interpreter.access import MemoryRecorder

protocol=json.loads((Path(__file__).resolve().parents[1]/'PROTOCOL.json').read_text())['nine']


def vector_layout(x,out):
    return x.tile((128,)),out.tile((128,))


def affine(x,out):
    out=x*2+1


def row_layout(x,out):
    return x.tile((1,512)),out.tile((1,512))


def softmax(x,out):
    shifted=x-ntl.max(x,1)[:,None]
    numerator=ntl.exp(shifted)
    out=numerator/ntl.sum(numerator,1)[:,None]


def digest(value):
    return hashlib.sha256(pickle.dumps(value,protocol=4)).hexdigest()


def workloads():
    for shape in protocol['target_shapes']:
        array=np.empty(shape,dtype=np.float32)
        coordinates=np.indices(shape,dtype=np.int64,sparse=True)
        recorder=MemoryRecorder({'buffer':array})
        for step in protocol['mask_steps']:
            mask=np.arange(array.size).reshape(shape)%step==0
            for kind in protocol['target_kinds']:
                def run(recorder=recorder,array=array,coordinates=coordinates,mask=mask,kind=kind):
                    with recorder.capture() as events:recorder.access(kind,array,coordinates,mask)
                    return tuple(events)
                yield f'recorder_{"x".join(map(str,shape))}_step{step}_{kind}',True,run,lambda value:None
    for case in ('tiny','empty','unknown'):
        array=np.zeros(32,dtype=np.int32);recorder=MemoryRecorder({'buffer':array})
        source=array.copy() if case=='unknown' else array
        mask=np.zeros(32,dtype=bool) if case=='empty' else np.ones(32,dtype=bool)
        def run(recorder=recorder,source=source,mask=mask):
            with recorder.capture() as events:recorder.access('write',source,(np.arange(32),),mask,linear=True)
            return tuple(events)
        yield 'recorder_'+case,False,run,lambda value:None
    rng=np.random.default_rng(9172031)
    for name,shape,arrange,apply in [('vector',(8193,),vector_layout,affine),('softmax',(17,257),row_layout,softmax)]:
        x=rng.uniform(-1,1,shape).astype(np.float32);out=np.empty_like(x)
        tensors=tuple(Tensor(x.ndim,name=n,dtype='float32',other=float('-inf') if name=='softmax' else 0) for n in ('x','out'))
        kernel=interpret(arrange,apply,tensors,backend='triton')
        exp=np.exp(x-x.max(1,keepdims=True)) if name=='softmax' else None
        expected=exp/exp.sum(1,keepdims=True) if exp is not None else x*2+1
        before=x.copy()
        for trace in (False,True):
            def run(x=x,out=out,kernel=kernel,trace=trace):
                result=interpret_program(kernel.program,{'x':x,'out':out},tensors=kernel.tensors,symbols=kernel.meta,trace=trace)
                return result.trace
            def check(value,out=out,expected=expected,x=x,before=before):
                np.testing.assert_allclose(out,expected,rtol=1e-5,atol=1e-5);np.testing.assert_array_equal(x,before)
            yield f'program_{name}_trace{trace}',False,run,check


report={'status':'RUNNING','source':str(args.source),'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'source_sha256':{str(p.relative_to(args.source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((args.source/'src').rglob('*.py'))},'cases':[]}
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as output:
    def save():
        output.seek(0);output.write(json.dumps(report,indent=2)+'\n');output.truncate();output.flush()
    save()
    for name,target,run,check in workloads():
        result=run();check(result);reference=digest(result);events=len(result);del result
        times=[]
        for _ in range(protocol['samples']):
            gc.collect();start=time.perf_counter()
            for __ in range(protocol['calls_per_sample']):result=run();del result
            times.append((time.perf_counter()-start)/protocol['calls_per_sample'])
        gc.collect();tracemalloc.start();result=run();retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        check(result);assert digest(result)==reference
        report['cases'].append({'name':name,'target':target,'times':times,'event_digest':reference,'events':events,'peak_bytes':peak,'retained_bytes':retained})
        save();print(name,'PASS',flush=True)
    report['status']='PASS';save()
