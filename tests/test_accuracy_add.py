"""Accuracy tests for the element-wise add kernel via MLIR pipeline.

Generates Triton MLIR from ninetoothed, compiles with ``triton.compiler``,
launches on GPU, and verifies results against PyTorch reference.

.. note::

    Due to a Triton compiler limitation, this file and
    ``test_accuracy_matmul.py`` should not be collected in the same pytest
    session.  Run them separately or use ``pytest-xdist`` (``-n auto``) to
    isolate processes.

Usage::

    pytest tests/test_accuracy_add.py -v
"""

import shutil

import pytest
import torch

import ninetoothed
from ninetoothed import Symbol, Tensor
from tests.utils import get_available_devices

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def _add_arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
    input_arranged = input.tile((BLOCK_SIZE,))
    other_arranged = other.tile((BLOCK_SIZE,))
    output_arranged = output.tile((BLOCK_SIZE,))
    return input_arranged, other_arranged, output_arranged


def _add_application(input, other, output):
    output = input + other


def _compile(mlir_file):
    from triton.compiler import compile as triton_compile

    ttir_path = mlir_file.replace(".mlir", ".ttir")
    shutil.copy(mlir_file, ttir_path)
    return triton_compile(ttir_path, options={"num_warps": 4, "num_stages": 3})


# -- module-level kernel compilation (runs once at import time) ----------------

_tensors = tuple(Tensor(1) for _ in range(3))
_kernel = ninetoothed.make(
    _add_arrangement, _add_application, _tensors, use_mlir=True
)
_ck = _compile(_kernel.mlir_file)

# Generated MLIR function signature:
#   %arg0: i32          -> BLOCK_SIZE
#   %arg1: i32          -> next_power_of_2(BLOCK_SIZE)
#   %arg2: !tt.ptr<f32> -> input_pointer
#   %arg3: i32          -> input_size_0
#   %arg4: i32          -> input_stride_0
#   %arg5: !tt.ptr<f32> -> other_pointer
#   %arg6: i32          -> other_size_0
#   %arg7: i32          -> other_stride_0
#   %arg8: !tt.ptr<f32> -> output_pointer
#   %arg9: i32          -> output_size_0
#   %arg10: i32         -> output_stride_0

BLOCK = 1024  # Must match the hardcoded tile size in the generated MLIR


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("N", (1024, 2048, 512, 1234))
def test_add_accuracy(N, device):
    torch.manual_seed(42)

    a = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)
    c = torch.empty_like(a)
    c_ref = a + b

    num_blocks = (N + BLOCK - 1) // BLOCK
    grid = (num_blocks, 1, 1)
    launcher = _ck[grid]
    launcher(BLOCK, BLOCK, a, N, 1, b, N, 1, c, N, 1)
    torch.cuda.synchronize()

    max_diff = (c - c_ref).abs().max().item()
    mean_diff = (c - c_ref).abs().mean().item()

    assert max_diff < 1e-6, (
        f"N={N}: max_diff={max_diff:.2e} >= 1e-6, "
        f"mean_diff={mean_diff:.2e}"
    )
