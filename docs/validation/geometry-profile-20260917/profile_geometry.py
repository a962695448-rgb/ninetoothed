"""Count repeated view geometry in actual frontend workloads; no speed claim."""
from collections import Counter
import argparse
import platform
import hashlib
import json
import sys
import weakref
from pathlib import Path

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();SOURCE=args.source.resolve()
manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(SOURCE/'src'))
import numpy as np
import sympy
import ninetoothed.language as ntl
from ninetoothed import Tensor,interpret
from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.memory import TensorRef


def vector(x,out):return x.tile((128,)),out.tile((128,))
def affine(x,out):out=x*2+1
def matrix(a,b,out):return a.tile((4,32)),b.tile((32,4)),out.tile((4,4))
def dot(a,b,out):out=ntl.dot(a,b)
def rows(x,out):return x.tile((1,256)),out.tile((1,256))
def softmax(x,out):
    exponent=ntl.exp(x-ntl.max(x,1)[:,None])
    out=exponent/ntl.sum(exponent,1)[:,None]


original=TensorRef._access
signatures=Counter();object_counts=Counter();symbol_types=Counter();identities={};serial=0


def observed(self,extra_mask=True):
    global serial
    address=id(self)
    previous=identities.get(address)
    if previous is None or previous[0]() is not self:
        serial+=1;identities[address]=(weakref.ref(self),serial)
    object_counts[identities[address][1]]+=1
    scalar_items=[];eligible=True
    for name,value in self.symbols.items():
        symbol_types[type(value).__module__+'.'+type(value).__qualname__]+=1
        if type(name) is str and (type(value) is int or type(value) is bool):
            scalar_items.append((name,type(value).__name__,value))
        else:
            eligible=False
    if eligible and type(extra_mask) is bool:
        key=(id(self.array),id(self.spec),tuple(self.array.shape),self.outer_index,self.level,
             self.extracted,tuple(sorted(scalar_items)),extra_mask)
        signatures[key]+=1
    return original(self,extra_mask)


TensorRef._access=observed
rng=np.random.default_rng(9172050)
x=rng.normal(size=4099).astype(np.float32)
v=rng.uniform(-1,1,(13,129)).astype(np.float32);e=np.exp(v-v.max(1,keepdims=True))
a=rng.integers(-8,9,(9,32)).astype(np.float32)/8;b=rng.integers(-8,9,(32,11)).astype(np.float32)/8
cases=[('vector',vector,affine,{'x':x},x*2+1,0),('softmax',rows,softmax,{'x':v},e/e.sum(1,keepdims=True),float('-inf')),
       ('matmul',matrix,dot,{'a':a,'b':b},a@b,0)]
report={'purpose':'Instrumentation counts only, not timed performance. Primitive binding signatures do not prove caching is semantically safe.',
        'source_commit':manifest['commit'],'python':sys.version,'numpy':np.__version__,'sympy':sympy.__version__,'platform':platform.platform(),
        'source_sha256':{str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SOURCE/'src').rglob('*.py'))},'cases':[]}
try:
    for name,arrange,apply,inputs,expected,other in cases:
        tensors=tuple(Tensor(value.ndim,name=key,dtype='float32',other=other) for key,value in inputs.items())+(Tensor(expected.ndim,name='out',dtype='float32'),)
        kernel=interpret(arrange,apply,tensors,backend='triton');data=dict(inputs,out=np.empty_like(expected))
        signatures.clear();object_counts.clear();symbol_types.clear();identities.clear();serial=0
        result=interpret_program(kernel.program,data,tensors=kernel.tensors,symbols=kernel.meta)
        np.testing.assert_allclose(result.outputs['out'],expected,rtol=1e-4,atol=1e-4)
        item={'name':name,'access_calls':sum(object_counts.values()),'ref_objects':len(object_counts),
              'objects_accessed_more_than_once':sum(n>1 for n in object_counts.values()),
              'primitive_binding_accesses':sum(signatures.values()),'distinct_primitive_geometry_signatures':len(signatures),
              'maximum_signature_reuse':max(signatures.values(),default=0),'symbol_types':dict(symbol_types)}
        report['cases'].append(item);print(item)
finally:
    TensorRef._access=original
with args.output.open('x') as output:
    output.write(json.dumps(report,indent=2)+'\n')
