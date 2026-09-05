CPU reference interpreter
=========================

``ninetoothed.interpret`` executes an application on NumPy arrays using the
same arrangement-to-layout and application-to-SSA frontend as the compiler.
It does not compile GPU source, initialize a GPU runtime, or run the original
application as ordinary Python. Its purpose is semantic validation and debugging,
not measuring GPU performance.

Quick start
-----------

Save the following in a Python file: the existing frontend inspects function
source, so a function entered only in an interactive ``python -c`` string may
not have retrievable source.

.. code-block:: python

   import numpy as np
   from ninetoothed import Tensor, interpret

   def arrangement(x, out):
       return x.tile((4,)), out.tile((4,))

   def application(x, out):
       out = x * 2 + 1

   kernel = interpret(
       arrangement, application,
       (Tensor(1, name="x", dtype="float32"),
        Tensor(1, name="out", dtype="float32")),
       trace=True,
   )
   x = np.arange(11, dtype=np.float32)
   out = np.empty_like(x)
   result = kernel(x, out)
   np.testing.assert_array_equal(out, x * 2 + 1)
   print(result.outputs["out"])

Output arrays are modified in place. Inputs are NumPy arrays, CPU Python/NumPy
scalars, or already-loaded PyTorch CPU tensors. A strided CPU Torch tensor is
adapted with ``tensor.numpy()`` without copying, so writes reach the original
storage and ``result.outputs["out"] is out`` remains true for a Torch output.
Non-contiguous CPU tensor views retain their strides. CUDA/non-CPU tensors,
``requires_grad=True``, sparse layouts, conjugate/negative view bits and dtypes
that cannot be represented without copying are rejected explicitly. Detaching
or moving tensors is the caller's decision; the interpreter never does it
implicitly. The interpreter itself never imports PyTorch or Triton for NumPy
execution.
Array dtypes must match declared SSA types. The acceptance-tested arithmetic
types are float32, int32, and bool; index values use int64 internally.
Object, complex, string, structured, and unsupported dtypes are rejected.
Signed integer ``floordiv`` rounds toward negative infinity, and ``mod`` has
the divisor's sign, following the SSA/Python contract. The Triton emitter
corrects its native signed division/remainder when needed. Division by zero
and the signed overflow case ``INT_MIN / -1`` are outside the validated inputs.

The package's existing installation metadata still lists Triton as a dependency.
For CPU-only contributor testing, use an isolated environment containing NumPy,
SymPy, and pytest, and run from this checkout with ``PYTHONPATH=src``. This is a
source-checkout testing route, not a newly published CPU-only wheel.

Frontend versus backend SSA
---------------------------

By default, ``kernel.program`` is the original frontend SSA. To execute the SSA
pass pipeline used by a backend, select it explicitly:

.. code-block:: python

   kernel = interpret(arrangement, application, descriptors, backend="triton")
   # These are separate, inspectable programs.
   original = kernel.frontend_program
   transformed = kernel.program

``backend`` calls the existing ``lower_for_target`` pass pipeline, without
source emission or GPU materialization. It is therefore useful on a CPU-only
machine. ``pipeline`` and ``pass_options`` select the same pipeline options as
the compiler. There is no fallback to the original program if an operation in
the transformed program is unsupported.

To evaluate an already available program, use:

.. code-block:: python

   from ninetoothed.interpreter import interpret_program

   result = interpret_program(
       transformed, {"x": x, "out": out}, tensors=kernel.tensors,
       symbols=kernel.meta, trace=True,
   )
   # Or retain an existing handle's layouts and binding order:
   result = kernel.with_program(transformed)(x, out)

The lower-level entry point accepts the actual ``ssa.Program`` object, not a
string containing generated Triton/CUDA code. ``TensorSpec.layout`` provides
the same structured coordinate maps and predicates used by emitters.

Memory and control flow
-----------------------

Arranged blocks are mapped to source coordinates with the existing ``IndexExpr``
and ``AccessMap`` records. They support broadcast program domains, strided source
arrays, nested layout levels and non-divisible tails. Masked-out coordinates
are never used to index the source allocation. ``Tensor(other=...)`` controls
the masked load value; use zero for sums/dots and negative infinity for a
softmax/max input where that is the intended kernel semantics.

The explicit pointer subset has these operand contracts:

* ``mem.data_ptr(tensor)`` returns a checked element pointer.
* ``arith.add(pointer, offsets)`` creates element-offset addresses.
* ``mem.load(pointer, mask?, other?)`` loads only active addresses.
* ``mem.store(value, destination, mask?)`` follows the frontend's value-first
  order. ``attrs["other"]`` supplies a load fill value when no third operand is
  given.

Raw pointer operations currently require C-contiguous storage. Arranged tensor
accesses use source coordinates and support NumPy strides, including negative
strides. An active out-of-bounds lane raises ``InterpretationError`` containing
the operation location and program ID; inactive lanes may have invalid addresses.

``scf.if`` requires a scalar condition and executes only the selected region.
``scf.for`` obeys Python range bounds and steps, including negative steps and
zero iterations. Region arguments and ``scf.yield`` carry values explicitly.

The default arranged launch uses a flat program ID, as current emitters do.
The lower-level API also accepts explicit one-to-three-dimensional grids for
program-ID operations. Trace IDs always contain three coordinates; unused
coordinates are zero. An explicit arranged grid must cover the same number of
programs as its layout domain.

Tracing, extension and pass debugging
-------------------------------------

``TraceEvent`` records ``program_id``, ``location``, ``opcode``, execution-before
``inputs`` snapshots, ``results`` snapshots, ``mask``, loop ``iteration``, and
``watched`` values. A snapshot stores shape, dtype and values. Memory destination
and pointer snapshots store reference metadata instead of dereferencing arbitrary
addresses. Lazy tensor references in results and watch lists likewise retain
metadata; observing them never introduces an unmasked memory read. Numeric
source snapshots are copied, so later mutations do not alter the recorded trace.

The lower-level API accepts ``program_ids``, ``opcodes`` and ``watch`` filters.
``callback(event)`` receives each selected event synchronously. ``StepDebugger``
offers terminal and deterministic scripted-command interfaces without a
background thread. Pauses occur after an operation completes. Writes performed
before a pause or quit remain in the output buffer.

.. code-block:: python

   from ninetoothed.interpreter.debugger import StepDebugger

   debugger = StepDebugger(
       commands=["watch %0", "print %0", "step", "break mem.store", "continue"],
       output=print,
   )
   kernel = interpret(arrangement, application, descriptors, callback=debugger)
   result = kernel(x, out)
   print([event.location for event in debugger.pauses])

Omit ``commands`` to read commands interactively. Use ``step``/``s`` to stop at
the next completed operation, ``continue``/``c`` to run to the next breakpoint,
``break LOCATION``/``b LOCATION`` to add an exact location or prefix,
``break OPCODE`` to break on an operation type, ``delete``/``d`` to remove it,
``watch NAME``/``w NAME`` to track an SSA value, and ``print NAME``/``p NAME`` to
show its latest observed snapshot. ``quit``/``q`` raises ``DebuggerQuit``.
An empty command steps; an exhausted scripted command stream continues.

``debugger.inspect(name)`` returns the latest observed snapshot. The ``values``
cache resets for each program ID and scalar output lane; it does not claim that a previously observed
region-local name remains in scope. Dynamic watch names are requested from the
execution engine at each event. Avoid trace filters during full debugging if
all intermediate definitions should remain observable.

The runnable example accepts ``--debug`` for scripted stepping and breakpoints,
or ``--interactive-debug`` to read commands from the terminal:

.. code-block:: console

   PYTHONPATH=src python docs/cpu_interpreter_demo.py --debug

The demo checks the correct ``x * 2 + 1`` result against NumPy, then deliberately
changes the SSA constant ``2`` to ``3``. ``check_passes`` identifies
``injected_bad_constant`` and the first different ``arith.constant`` operation.
This is fault injection for teaching the debugger, not a historical compiler bug.
Eleven inputs with tiles of four also demonstrate three program IDs and a
masked final lane.

Use a new directory to save both programs and replay the same difference in a
separate process, without rerunning the frontend or the injected pass:

.. code-block:: console

   PYTHONPATH=src python docs/cpu_interpreter_demo.py --debug --export /tmp/nine-demo
   PYTHONPATH=src python /tmp/nine-demo/replay.py

The directory contains ``reference/`` and ``candidate/`` replay bundles plus a
top-level ``replay.py``. Both bundles preserve the original inputs, layouts,
dtype and seed. The top-level script checks the reference result against NumPy,
verifies the injected candidate's result, and prints ``Different outputs:
('out',)`` and the first different operation. Exit code zero means the expected
injected fault was reproduced; if the saved candidate no longer differs, the
script fails. The saved-SSA comparison does not claim to rerun or independently
identify the pass. Reusing an existing export directory is rejected.

An extension can register a handler for an individual operation without changing
the frontend or matching an entire application:

.. code-block:: python

   result = interpret_program(
       program, inputs,
       handlers={"math.my_op": lambda operation, operands: operands[0] + 2},
   )

Unregistered unsupported operations fail with ``UnsupportedOperationError`` and
an operation location. Handlers return one NumPy result, or a tuple for multiple
SSA results, and remain responsible for their operation's semantics.

.. code-block:: python

   from ninetoothed.backends.core import Target
   from ninetoothed.compiler.passes import Context, default_pipeline
   from ninetoothed.interpreter.debugger import check_passes

   context = Context(
       backend=Target.TRITON, compiler_options={}, kernel_metadata={},
       tensors=kernel.tensors,
   )
   pipeline = default_pipeline(Target.TRITON)
   checks = tuple(
       (pass_.name, lambda program, pass_=pass_: pass_.run(program, context))
       for pass_ in pipeline.passes
   )
   report = check_passes(
       kernel.frontend_program, checks, {"x": x, "out": out},
       tensors=kernel.tensors, symbols=kernel.meta,
   )

``check_passes`` compares every intermediate program to the original reference
using independent input copies, stopping at the first bad pass. Integer and
boolean outputs require exact equality. Floating outputs default to
``rtol=1e-3, atol=1e-3``. ``compare_programs`` additionally identifies the first
different corresponding operation when SSA structures and execution order can
be aligned. If a pass changes structure, output comparison remains valid but an
operation location is not guessed without provenance information.

Replay bundles
--------------

``export_reproducer(directory, program, inputs, tensors=..., symbols=..., seed=...)``
exports structured JSON, readable SSA, numeric NPZ inputs, shape/dtype/seed metadata,
array strides and write permissions, and a replay script. Differential copies
and loaded bundles preserve positive, negative and zero strides in independent
storage. ``load_reproducer`` loads JSON and NumPy data with
``allow_pickle=False``. Existing bundle files are never overwritten.
The supplied case is preserved; automatic shape/operation minimization is not
implemented. Multiple input names bound to the same array object retain that
alias relationship in differential copies and loaded bundles. Overlapping
distinct NumPy views are rejected for differential
copying/export because their shared-storage relationship is not serialized.

Current boundaries
------------------

The basic subset includes arithmetic, comparison, cast, selection, broadcasting,
masked memory, sum/max/min reductions, structured control flow and program IDs.
Direct ``linalg.dot``/``linalg.matmul`` and the exp/max/sum composition used for
softmax are also implemented. Scalar matmul/transpose decompositions execute the
actual SSA operations and region loops once per valid rank-2 output lane. This
supports one output store and a single logical program, either a whole untiled
matrix or one arranged tile with masked tails. The ``TraceEvent.lane`` field
identifies the output coordinate; normal vector execution has ``lane=None``.
Inactive output lanes are never executed or written. This is a diagnostic CPU
path, not a fast matrix multiplication implementation.

Multi-program tiled scalar decompositions and combinations with other output
stores remain explicitly unsupported, because their global-versus-local lane
domain must first be defined. Direct/undecomposed linalg operations retain their
normal vector semantics. Atomics, random-number operations, jagged layouts and
arbitrary external calls are not part of the validated subset.

The decomposition pass now derives a fixed or compound reduction extent through
``shape.dim(lhs, -1)`` from the arranged operand, instead of referring to an
undefined fallback ``k`` symbol. The symbolic-K path is preserved. Invalid SSA
from any pipeline still reports the original verification cause, with no
fallback to the frontend program.

CPU tests cannot establish agreement with A100 hardware or measure GPU speed.
That requires the separate real-GPU differential run on matching kernels,
inputs, layouts, dtypes, seeds and pass settings.
