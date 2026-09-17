"""Run identical public interpreter workloads in one isolated source process."""
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

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source / 'src'))
import numpy as np
import ninetoothed.language as ntl
from ninetoothed import Tensor, interpret
from ninetoothed.interpreter import interpret_program, expressions
from ninetoothed.ir import ssa

protocol = json.loads((Path(__file__).resolve().parents[1] / 'PROTOCOL.json').read_text())['nine']


def vector_layout(x, out):
    return x.tile((128,)), out.tile((128,))


def affine(x, out):
    out = x + x


def row_layout(x, out):
    return x.tile((1, 512)), out.tile((1, 512))


def softmax(x, out):
    shifted = x - ntl.max(x, 1)[:, None]
    exp = ntl.exp(shifted)
    out = exp / ntl.sum(exp, 1)[:, None]


def workloads():
    rng = np.random.default_rng(9172026)
    for shape, layout in zip(protocol['target_shapes'], protocol['target_layouts']):
        x = rng.integers(-32, 33, size=shape).astype(np.float32) / 16
        if layout == 'fortran':
            x = np.asfortranarray(x)
        if layout == 'reversed':
            x = x[::-1, ::-1]
        dtype = ssa.Type(kind='tensor', shape=tuple(map(str, shape)), dtype='float32')
        inp, out, added = [ssa.Value(name=n, type=dtype) for n in ('x', 'out', '%added')]
        program = ssa.Program(kind='identity_affine', inputs=(inp,out), outputs=(out,), blocks=(ssa.Block(operations=(
            ssa.Operation(opcode='arith.add', operands=('x','x'), results=(added,)),
            ssa.Operation(opcode='mem.store', operands=('%added','out')),
        )),))
        yield 'store_'+'x'.join(map(str,shape))+'_'+layout, False, x, x+x, program, {}, 'out'
        program = ssa.Program(kind='identity_readonly', inputs=(inp,), outputs=(added,), blocks=(ssa.Block(operations=(ssa.Operation(opcode='arith.add', operands=('x','x'), results=(added,)),)),))
        yield 'read_'+'x'.join(map(str,shape))+'_'+layout, True, x, x+x, program, {}, '%added'
    for name, shape, arrange, apply in [('vector', (4096,), vector_layout, affine), ('softmax',(17,257),row_layout,softmax)]:
        x = rng.uniform(-1,1,shape).astype(np.float32)
        tensors = tuple(Tensor(x.ndim, name=n, dtype='float32', other=float('-inf') if name=='softmax' else 0) for n in ('x','out'))
        kernel = interpret(arrange,apply,tensors,backend='triton')
        exp = np.exp(x-x.max(1,keepdims=True)) if name=='softmax' else None
        expected = exp/exp.sum(1,keepdims=True) if name=='softmax' else x+x
        yield name, False, x, expected, kernel.program, {'tensors':kernel.tensors,'symbols':kernel.meta}, 'out'


def clear_caches():
    expressions._compiled_expression.cache_clear()
    expressions._numpy_dtype_from_name.cache_clear()


report = {'status':'RUNNING','source':str(args.source),'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
          'source_sha256':{str(f.relative_to(args.source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((args.source/'src').rglob('*.py'))},'cases':[]}
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as output:
    def save():
        output.seek(0); output.write(json.dumps(report,indent=2)+'\n'); output.truncate(); output.flush()
    save()
    for name,target,x,expected,program,options,result_key in workloads():
        out = np.empty_like(x)
        original = x.copy()
        for trace in (False,True):
            def run():
                return interpret_program(program,{'x':x} if result_key=='%added' else {'x':x,'out':out},trace=trace,**options)
            result=run();np.testing.assert_allclose(result.outputs[result_key],expected,rtol=1e-5,atol=1e-5)
            digest=hashlib.sha256(pickle.dumps(result.trace,protocol=4)).hexdigest();events=len(result.trace);del result
            times=[]
            for _ in range(protocol['samples']):
                gc.collect();start=time.perf_counter()
                for __ in range(protocol['calls_per_sample']):
                    result=run();del result
                times.append((time.perf_counter()-start)/protocol['calls_per_sample'])
            memory={}
            for mode in ('warm','cold'):
                if mode=='cold':clear_caches()
                gc.collect();tracemalloc.start();result=run();retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                np.testing.assert_allclose(result.outputs[result_key],expected,rtol=1e-5,atol=1e-5)
                assert hashlib.sha256(pickle.dumps(result.trace,protocol=4)).hexdigest()==digest
                memory[mode]={'peak':peak,'retained':retained};del result
            np.testing.assert_array_equal(x,original)
            report['cases'].append({'name':name,'target':target and not trace,'trace':trace,'times':times,'memory':memory,'trace_sha256':digest,'events':events})
            save();print(name,trace,'PASS',flush=True)
    report['status']='PASS';save()
