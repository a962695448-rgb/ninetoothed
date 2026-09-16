"""Prepared SSA runtime on new shapes, layouts, and dtypes; emit exact records."""
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
from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter import expressions


def vectors(x,out):return x.tile((128,)),out.tile((128,))


def float_loop(x,out):
    accumulator=x
    for i in range(8):
        accumulator=accumulator*0.5+0.25
    out=accumulator


def int_loop(x,out):
    accumulator=x
    for i in range(8):
        accumulator=accumulator+3
    out=accumulator


def rows(x,out):return x.tile((1,512)),out.tile((1,512))


def softmax(x,out):
    shifted=x-ntl.max(x,1)[:,None]
    numerator=ntl.exp(shifted)
    out=numerator/ntl.sum(numerator,1)[:,None]


def matrices(a,b,out):return a.tile((4,16)),b.tile((16,4)),out.tile((4,4))


def matmul(a,b,out):out=ntl.dot(a,b)


def cases():
    rng=np.random.default_rng(93712026)
    for size,dtype,layout in [(4096,np.float32,'contiguous'),(16387,np.float32,'strided'),(8193,np.int32,'reversed')]:
        data=rng.integers(-8,9,size=size).astype(dtype)
        if dtype==np.float32:data/=8
        if layout=='strided':
            storage=np.full(size*2+1,-111,dtype=dtype);x=storage[1::2];x[:]=data
        elif layout=='reversed':x=data[::-1]
        else:x=data
        expected=x.copy()
        for _ in range(8):expected=expected*np.float32(.5)+np.float32(.25) if dtype==np.float32 else expected+3
        yield f'loop_{size}_{np.dtype(dtype).name}_{layout}',True,vectors,float_loop if dtype==np.float32 else int_loop,{'x':x},expected,0
    x=rng.uniform(-1,1,(17,257)).astype(np.float32);v=np.exp(x-x.max(1,keepdims=True))
    yield 'softmax_17x257_tail',False,rows,softmax,{'x':x},v/v.sum(1,keepdims=True),float('-inf')
    a=rng.integers(-4,5,(9,16)).astype(np.float32)/4;b=rng.integers(-4,5,(16,7)).astype(np.float32)/4
    yield 'matmul_9x16x7_tail',False,matrices,matmul,{'a':a,'b':b},a@b,0


report={'status':'RUNNING','source':str(args.source),'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
        'source_sha256':{str(f.relative_to(args.source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((args.source/'src').rglob('*.py'))},'cases':[]}
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as output:
    def save():output.seek(0);output.write(json.dumps(report,indent=2)+'\n');output.truncate();output.flush()
    save()
    for name,target,arrange,apply,inputs,expected,other in cases():
        dtype=expected.dtype
        tensors=tuple(Tensor(x.ndim,name=n,dtype=str(x.dtype),other=other) for n,x in inputs.items())+(Tensor(expected.ndim,name='out',dtype=str(dtype)),)
        kernel=interpret(arrange,apply,tensors,backend='triton')
        arrays=dict(inputs,out=np.empty_like(expected));before={n:x.copy() for n,x in inputs.items()}
        for trace in [False,True]:
            def run():
                arrays['out'].fill(-999)
                return interpret_program(kernel.program,arrays,tensors=kernel.tensors,symbols=kernel.meta,trace=trace)
            result=run();digest=hashlib.sha256(pickle.dumps(result.trace,protocol=4)).hexdigest();events=len(result.trace);del result
            timings=[]
            for _ in range(7):
                gc.collect();start=time.perf_counter();result=run();timings.append(time.perf_counter()-start)
                np.testing.assert_allclose(result.outputs['out'],expected,rtol=1e-4,atol=1e-4)
                assert hashlib.sha256(pickle.dumps(result.trace,protocol=4)).hexdigest()==digest
                del result
            memory={}
            for mode in ['warm','cold']:
                if mode=='cold':
                    expressions._compiled_expression.cache_clear()
                    fn=getattr(expressions,'_numpy_dtype_from_name',None)
                    if fn is not None:fn.cache_clear()
                gc.collect();tracemalloc.start();result=run();retained,peak=tracemalloc.get_traced_memory()
                del result;gc.collect();post_result=tracemalloc.get_traced_memory()[0]
                expressions._compiled_expression.cache_clear();fn=getattr(expressions,'_numpy_dtype_from_name',None)
                if fn is not None:fn.cache_clear()
                gc.collect();after_clear=tracemalloc.get_traced_memory()[0];tracemalloc.stop()
                memory[mode]={'retained':retained,'peak':peak,'after_result_release':post_result,'after_cache_clear':after_clear}
            for n,x in before.items():np.testing.assert_array_equal(arrays[n],x)
            report['cases'].append({'name':name,'target':target and not trace,'trace':trace,'times':timings,'events':events,'trace_sha256':digest,'memory':memory,'correctness':'PASS'})
            save();print(name,trace,timings,flush=True)
    report['status']='PASS';save()
