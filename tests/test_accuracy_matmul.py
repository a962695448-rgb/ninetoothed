"""Accuracy tests for the matmul kernel via MLIR pipeline.

Generates Triton MLIR from ninetoothed, compiles with ``triton.compiler``,
launches on GPU, and verifies results against PyTorch reference.

.. note::

    Due to a Triton compiler limitation, this file and
    ``test_accuracy_add.py`` should not be collected in the same pytest
    session.  Run them separately or use ``pytest-xdist`` (``-n auto``) to
    isolate processes.

Usage::

    pytest tests/test_accuracy_matmul.py -v
"""

import shutil

import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from tests.utils import get_available_devices

BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64


def _matmul_arrangement(input, other, output,
    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K):
    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    input_arranged = input.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    other_arranged = other.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)

    return input_arranged, other_arranged, output_arranged


def _matmul_application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])
    output = accumulator


def _compile(mlir_file):
    from triton.compiler import compile as triton_compile

    ttir_path = mlir_file.replace(".mlir", ".ttir")
    shutil.copy(mlir_file, ttir_path)
    return triton_compile(ttir_path, options={"num_warps": 4, "num_stages": 3})


# -- module-level kernel compilation (runs once at import time) ----------------

_tensors = (Tensor(2), Tensor(2), Tensor(2))
_kernel = ninetoothed.make(
    _matmul_arrangement, _matmul_application, _tensors, use_mlir=True
)
_ck = _compile(_kernel.mlir_file)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "M, K, N",
    ((256, 256, 256), (128, 64, 256), (64, 128, 256), (123, 57, 89)),
)
def test_matmul_accuracy(M, K, N, device):
    torch.manual_seed(42)

    a = torch.randn(M, K, device=device, dtype=torch.float32)
    b = torch.randn(K, N, device=device, dtype=torch.float32)
    c = torch.empty(M, N, device=device, dtype=torch.float32)
    c_ref = torch.mm(a, b)

    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m * grid_n, 1, 1)

    launcher = _ck[grid]
    launcher(a, M, K, K, 1, b, K, N, N, 1, c, M, N, N, 1)
    torch.cuda.synchronize()

    max_diff = (c - c_ref).abs().max().item()
    ref_max = c_ref.abs().max().item()
    rel_err = max_diff / (ref_max + 1e-30)
    mean_abs = (c - c_ref).abs().mean().item()

    # tf32 matmul has lower precision
    tol = 1e-1
    assert rel_err < tol, (
        f"M={M}, K={K}, N={N}: rel_err={rel_err:.2e} >= {tol}, "
        f"max_diff={max_diff:.2e}, mean_abs={mean_abs:.2e}"
    )
