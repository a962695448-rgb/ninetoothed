"""Mid IR to Triton MLIR conversion pass.

This module converts MidFunction IR nodes into valid Triton MLIR text using
the ``triton._C.libtriton.ir`` builder API.

The resulting MLIR can be compiled to PTX / GPU binaries by Triton's compiler
backend.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ninetoothed import naming
from ninetoothed.symbol import Symbol
from ninetoothed.ir.mid_ir import (
    MidArange,
    MidAssign,
    MidBinOp,
    MidBoolOp,
    MidCall,
    MidCompare,
    MidConstant,
    MidDataPtr,
    MidDtypeAttr,
    MidExprStmt,
    MidFor,
    MidFunction,
    MidIf,
    MidIfExp,
    MidInvariant,
    MidLoad,
    MidMaskExpr,
    MidName,
    MidOffsets,
    MidParam,
    MidPointerExpr,
    MidProgramId,
    MidReturn,
    MidStore,
    MidStride,
    MidSubscript,
    MidTile,
    MidTuple,
    MidUnaryOp,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AXIS_NAMES = {0: "x", 1: "y", 2: "z"}

# Matches 'ninetoothed_tensor_{idx}_pointers' to extract the tensor index
_TENSOR_PTR_INDEX_RE = re.compile(r"ninetoothed_tensor_(\d+)_pointers")
_TENSOR_BARE_NAME_RE = re.compile(r"ninetoothed_tensor_(\d+)$")

# ninetoothed dtype string -> canonical MLIR element type string
_DTYPE_TO_MLIR: Dict[str, str] = {
    "float32": "f32",
    "fp32": "f32",
    "float16": "f16",
    "fp16": "f16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float64": "f64",
    "fp64": "f64",
    "int32": "i32",
    "i32": "i32",
    "int64": "i64",
    "i64": "i64",
    "int8": "i8",
    "i8": "i8",
    "int16": "i16",
    "i16": "i16",
    "int1": "i1",
    "bool": "i1",
}

# Integer MLIR element types
_INT_TYPES = frozenset({"i1", "i8", "i16", "i32", "i64"})

# Float MLIR element types
_FLOAT_TYPES = frozenset({"f16", "bf16", "f32", "f64"})


# Type helpers (pure -- no builder dependency)
# ---------------------------------------------------------------------------

def _mlir_elem_type(dtype_str: Optional[str]) -> str:
    """Return the canonical MLIR element type string for a ninetoothed dtype."""
    if dtype_str is None:
        return "f32"
    return _DTYPE_TO_MLIR.get(dtype_str, "f32")


def _is_int_type(mlir_type: Optional[str]) -> bool:
    """Check if an MLIR element type string is an integer type."""
    if mlir_type is None:
        return False
    # Handle both bare types ("i32") and tensor types ("tensor<1024xi32>")
    # and pointer types ("!tt.ptr<f32>")
    if mlir_type.startswith("tensor<"):
        inner = mlir_type[len("tensor<"):-1]  # strip "tensor<" and ">"
        # inner is like "1024xi32"
        elem = inner.split("x")[-1]
        return elem in _INT_TYPES
    if mlir_type.startswith("!tt.ptr<"):
        # Pointers are address-space integers, treat as int for op dispatch
        return True
    return mlir_type in _INT_TYPES


def _is_float_type(mlir_type: Optional[str]) -> bool:
    """Check if an MLIR element type string is a float type."""
    if mlir_type is None:
        return False
    if mlir_type.startswith("tensor<"):
        inner = mlir_type[len("tensor<"):-1]
        elem = inner.split("x")[-1]
        return elem in _FLOAT_TYPES
    return mlir_type in _FLOAT_TYPES


def _is_tensor_type(mlir_type: Optional[str]) -> bool:
    """Check if an MLIR type is a tensor type."""
    return mlir_type is not None and mlir_type.startswith("tensor<")


def _is_scalar_type(mlir_type: Optional[str]) -> bool:
    """Check if an MLIR type is a scalar (non-tensor, non-pointer) type."""
    if mlir_type is None:
        return True
    return not mlir_type.startswith("tensor<") and not mlir_type.startswith("!")


def _elem_of(mlir_type: str) -> str:
    """Extract the element type from a potentially compound MLIR type."""
    if mlir_type.startswith("tensor<"):
        inner = mlir_type[len("tensor<"):-1]
        return inner.split("x")[-1]
    if mlir_type.startswith("!tt.ptr<"):
        inner = mlir_type[len("!tt.ptr<"):-1]
        return inner
    return mlir_type


def _shape_of(mlir_type: str) -> Optional[List[int]]:
    """Extract the shape dimensions from a tensor type, or None for non-tensors."""
    if not mlir_type.startswith("tensor<"):
        return None
    inner = mlir_type[len("tensor<"):-1]
    # inner is like "1024xi32" or "16x16xf32"
    parts = inner.split("x")
    if len(parts) < 2:
        return None
    # Last part is the element type; everything before is shape
    shape_parts = parts[:-1]
    try:
        return [int(s) for s in shape_parts]
    except ValueError:
        return None


def _tensor_type(shape: List[int], elem_type: str) -> str:
    """Build an MLIR tensor type string, e.g. ``tensor<1024xf32>``."""
    shape_str = "x".join(str(d) for d in shape)
    return f"tensor<{shape_str}x{elem_type}>"


def _ptr_type(elem_type: str) -> str:
    """Build an MLIR Triton pointer type string, e.g. ``!tt.ptr<f32>``."""
    return f"!tt.ptr<{elem_type}>"


def _ptr_tensor_type(elem_type: str, shape: List[int]) -> str:
    """Build a tensor-of-pointers type string, e.g. ``tensor<1024x!tt.ptr<f32>>``."""
    shape_str = "x".join(str(d) for d in shape)
    return f"tensor<{shape_str}x!tt.ptr<{elem_type}>>"


def _broadcast_type(ty_a: Optional[str], ty_b: Optional[str]) -> str:
    """Compute the result type of a binary operation following MLIR broadcast rules.

    If one operand is a tensor and the other is a scalar, the result is the
    tensor type.  If both are tensors with the same shape, the result is that
    tensor type.  If both are scalars, the result is the first operand's type.
    """
    if ty_a is None:
        return ty_b or "i32"
    if ty_b is None:
        return ty_a
    a_is_tensor = _is_tensor_type(ty_a)
    b_is_tensor = _is_tensor_type(ty_b)
    if a_is_tensor and not b_is_tensor:
        return ty_a
    if b_is_tensor and not a_is_tensor:
        return ty_b
    if a_is_tensor and b_is_tensor:
        a_dims = _shape_of(ty_a) or []
        b_dims = _shape_of(ty_b) or []
        if a_dims == b_dims:
            return ty_a
        # Broadcasting: element-wise max of dimensions
        if len(a_dims) == len(b_dims):
            elem = _elem_of(ty_a) or _elem_of(ty_b) or "i32"
            broadcast_dims = [max(d1, d2) for d1, d2 in zip(a_dims, b_dims)]
            return _tensor_type(broadcast_dims, elem)
        return ty_a
    return ty_a


def _find_arange_in_expr(node):
    """Find the first ``MidSubscript`` (arange) node in an expression tree."""
    if isinstance(node, MidSubscript):
        return node
    if isinstance(node, MidBinOp):
        return _find_arange_in_expr(node.lhs) or _find_arange_in_expr(node.rhs)
    return None


def _strip_arange_from_expr(node):
    """Remove the arange term from ``dim_off = start_part + arange``.

    Returns *start_part* (the non-arange child of the top-level ``+``),
    or ``MidConstant(0)`` if the entire expression is just an arange.
    """
    if isinstance(node, MidSubscript):
        return MidConstant(0)
    if isinstance(node, MidBinOp) and node.op == "+":
        if isinstance(node.rhs, MidSubscript):
            return node.lhs
        if isinstance(node.lhs, MidSubscript):
            return node.rhs
    return None


def _replace_tensor_elem(tensor_type: str, new_elem: str) -> str:
    """Replace the element type in a tensor type string.

    ``tensor<1024xi32>`` with ``new_elem="i1"`` becomes ``tensor<1024xi1>``.
    """
    if not tensor_type.startswith("tensor<"):
        return new_elem
    inner = tensor_type[len("tensor<"):-1]
    parts = inner.split("x")
    if len(parts) < 2:
        return new_elem
    parts[-1] = new_elem
    return f"tensor<{('x'.join(parts))}>"


@dataclass
class _BuilderValue:
    """Bookkeeping for a builder API value, pairing the native Triton value with its MLIR type string."""

    value: object      # triton._C.libtriton.ir.value or block_argument
    type_str: str      # MLIR type string like "i32", "tensor<64x64xf32>"


# ---------------------------------------------------------------------------
# Builder API abstraction layer
# ---------------------------------------------------------------------------

class _BuilderAPI:
    """Thin wrapper around the Triton builder API (``triton._C.libtriton.ir``).

    The :attr:`available` flag is set during :meth:`_init_builder` and
    checked once at the pipeline entry point.  All methods below assume
    the builder is available.
    """

    def __init__(self) -> None:
        self.available = False
        self._init_builder()

    # -- initialisation -----------------------------------------------------

    def _init_builder(self) -> None:
        try:
            from triton._C.libtriton import ir as tl_ir  # type: ignore[import-untyped]

            self._tl_ir = tl_ir
            self.ctx = tl_ir.context()
            tl_ir.load_dialects(self.ctx)
            self.builder = tl_ir.builder(self.ctx)
            self.mod = self.builder.create_module()
            self.available = True
            logger.info("Using triton._C.libtriton.ir builder API")
        except Exception as exc:
            logger.debug("Triton builder API unavailable (%s); using text fallback", exc)
            self.available = False

    # -- module / function --------------------------------------------------

    def create_function(
        self,
        module,
        name: str,
        arg_ir_types: list,
        ret_ir_types: list,
    ):
        func_type = self.builder.get_function_ty(arg_ir_types, ret_ir_types)
        func = self.builder.get_or_insert_function(module, name, func_type, "public", True)
        module.push_back(func)
        entry = func.add_entry_block()
        self.builder.set_insertion_point_to_end(entry)
        return func

    def create_block(self, func) -> None:
        # Block is created automatically by create_function
        pass

    # -- constants ----------------------------------------------------------

    def create_int_constant(self, value: int, bits: int = 32):
        if bits == 1:
            return self.builder.get_int1(bool(value))
        if bits == 64:
            return self.builder.get_int64(value)
        if bits == 32:
            return self.builder.get_int32(value)
        if bits == 16:
            return self.builder.get_int16(value)
        if bits == 8:
            return self.builder.get_int8(value)
        return self.builder.get_int32(value)

    def create_float_constant(self, value: float, ty_str: str = "f32"):
        if ty_str == "f16":
            return self.builder.get_fp16(value)
        if ty_str == "bf16":
            return self.builder.get_bf16(value)
        if ty_str == "f64":
            return self.builder.get_fp64(value)
        return self.builder.get_fp32(value)

    # -- arithmetic ---------------------------------------------------------

    def create_add(self, lhs, rhs, is_float: bool):
        if is_float:
            return self.builder.create_fadd(lhs, rhs)
        return self.builder.create_add(lhs, rhs)

    def create_sub(self, lhs, rhs, is_float: bool):
        if is_float:
            return self.builder.create_fsub(lhs, rhs)
        return self.builder.create_sub(lhs, rhs)

    def create_mul(self, lhs, rhs, is_float: bool):
        if is_float:
            return self.builder.create_fmul(lhs, rhs)
        return self.builder.create_mul(lhs, rhs)

    def create_div(self, lhs, rhs, is_float: bool):
        if is_float:
            return self.builder.create_fdiv(lhs, rhs)
        return self.builder.create_sdiv(lhs, rhs)

    def create_rem(self, lhs, rhs, is_float: bool):
        if is_float:
            return self.builder.create_frem(lhs, rhs)
        return self.builder.create_srem(lhs, rhs)

    def create_and(self, lhs, rhs):
        return self.builder.create_and(lhs, rhs)

    def create_or(self, lhs, rhs):
        return self.builder.create_or(lhs, rhs)

    def create_xor(self, lhs, rhs):
        return self.builder.create_xor(lhs, rhs)

    def create_shl(self, lhs, rhs):
        return self.builder.create_shl(lhs, rhs)

    def create_ashr(self, lhs, rhs):
        return self.builder.create_ashr(lhs, rhs)

    # -- comparison ---------------------------------------------------------

    def create_icmp(self, pred: str, lhs, rhs):
        method = getattr(self.builder, f"create_icmp_{pred}", None)
        if method is not None:
            return method(lhs, rhs)
        return None

    def create_fcmp(self, pred: str, lhs, rhs):
        method = getattr(self.builder, f"create_fcmp_{pred}", None)
        if method is not None:
            return method(lhs, rhs)
        return None

    # -- memory -------------------------------------------------------------

    def create_load(self, ptr, mask, other, ptr_type: str, mask_type: str, elem_type: str):
        return self.builder.create_load(ptr, mask, other)

    def create_store(self, ptr, val, mask, ptr_type: str, val_type: str, mask_type: str):
        return self.builder.create_store(ptr, val, mask)

    def create_addptr(self, base, offset, ptr_type: str, offset_type: str):
        return self.builder.create_addptr(base, offset)

    # -- pointer conversion --------------------------------------------------

    def create_ptr_to_int(self, ptr_val):
        return self.builder.create_ptr_to_int(ptr_val, self.builder.get_int64_ty())

    def create_int_to_ptr(self, int_val, result_type):
        return self.builder.create_int_to_ptr(int_val, result_type)

    # -- splat / broadcast --------------------------------------------------

    def create_splat(self, scalar, ir_tensor_type):
        return self.builder.create_splat(ir_tensor_type, scalar)

    # -- SPMD ---------------------------------------------------------------

    def create_get_program_id(self, axis: int):
        return self.builder.create_get_program_id(axis)

    def create_make_range(self, start: int, end: int):
        size = end - start
        tensor_ty = self.get_ir_type(_tensor_type([size], "i32"))
        return self.builder.create_make_range(tensor_ty, start, end)

    # -- math ---------------------------------------------------------------

    def create_exp(self, x):
        return self.builder.create_exp(x)

    def create_exp2(self, x):
        return self.builder.create_exp2(x)

    def create_log(self, x):
        return self.builder.create_log(x)

    def create_log2(self, x):
        return self.builder.create_log2(x)

    def create_sqrt(self, x):
        return self.builder.create_sqrt(x)

    def create_fma(self, a, b, c):
        return self.builder.create_fma(a, b, c)

    def create_pow(self, x, y):
        return self.builder.create_pow(x, y)

    def create_abs(self, x):
        return self.builder.create_abs(x)

    def create_floor(self, x):
        return self.builder.create_floor(x)

    def create_ceil(self, x):
        return self.builder.create_ceil(x)

    def create_rsqrt(self, x):
        return self.builder.create_rsqrt(x)

    def create_maximumf(self, lhs, rhs):
        return self.builder.create_maximumf(lhs, rhs)

    def create_minimumf(self, lhs, rhs):
        return self.builder.create_minimumf(lhs, rhs)

    def create_select(self, cond, true_val, false_val):
        return self.builder.create_select(cond, true_val, false_val)

    def create_fp_to_fp(self, value, type_str: str):
        ir_type = self.get_ir_type(type_str)
        if ir_type is None:
            return None
        return self.builder.create_fp_to_fp(value, ir_type, None)

    def create_int_cast(self, value, type_str: str, is_signed: bool = True):
        ir_type = self.get_ir_type(type_str)
        if ir_type is None:
            return None
        return self.builder.create_int_cast(value, ir_type, is_signed)

    def create_si_to_fp(self, value, type_str: str = "f32"):
        ir_type = self.get_ir_type(type_str)
        if ir_type is None:
            return None
        return self.builder.create_si_to_fp(value, ir_type)

    # -- control flow -------------------------------------------------------

    def create_for(self, lower, upper, step):
        return self.builder.create_for(lower, upper, step)

    def create_yield(self, values: list):
        return self.builder.create_yield_op(values)

    def create_if(self, cond):
        return self.builder.create_if(cond)

    def ret(self, values: list):
        return self.builder.ret(values)

    # -- module string ------------------------------------------------------

    def module_to_string(self) -> str:
        return self.mod.str_nodebug()

    # -- type helpers -------------------------------------------------------

    def get_ir_type(self, type_str: str):
        """Convert an MLIR type string to a Triton ir.type object."""
        if type_str.startswith("tensor<"):
            shape, elem = self._parse_tensor_type(type_str)
            elem_ty = self._resolve_elem_ir_type(elem)
            if elem_ty is not None and shape is not None:
                return self.builder.get_block_ty(elem_ty, shape)
        if type_str.startswith("!tt.ptr<"):
            inner = type_str[len("!tt.ptr<"):-1]
            elem_ty = self._elem_to_ir_type(inner)
            if elem_ty is not None:
                return self.builder.get_ptr_ty(elem_ty, 1)
        return self._elem_to_ir_type(type_str)

    def _resolve_elem_ir_type(self, elem: str):
        """Resolve an element type string, handling pointer types.

        Regular types like ``"f32"`` go through :meth:`_elem_to_ir_type`.
        Pointer types like ``"!tt.ptr<f32>"`` are resolved via
        :meth:`get_ptr_ty`.
        """
        if elem.startswith("!tt.ptr<"):
            inner = elem[len("!tt.ptr<"):-1]
            inner_ty = self._elem_to_ir_type(inner)
            if inner_ty is not None:
                return self.builder.get_ptr_ty(inner_ty, 1)
        return self._elem_to_ir_type(elem)

    def _elem_to_ir_type(self, elem: str):
        """Convert an element type string to ir.type."""
        mapping = {
            "i1": self.builder.get_int1_ty,
            "i8": self.builder.get_int8_ty,
            "i16": self.builder.get_int16_ty,
            "i32": self.builder.get_int32_ty,
            "i64": self.builder.get_int64_ty,
            "f16": self.builder.get_half_ty,
            "bf16": self.builder.get_bf16_ty,
            "f32": self.builder.get_float_ty,
            "f64": self.builder.get_double_ty,
        }
        fn = mapping.get(elem)
        return fn() if fn else None

    @staticmethod
    def _parse_tensor_type(type_str: str):
        """Parse 'tensor<64x64xf32>' into ([64, 64], 'f32')."""
        inner = type_str[len("tensor<"):-1]
        parts = inner.split("x")
        if len(parts) < 2:
            return None, None
        try:
            shape = [int(p) for p in parts[:-1]]
            return shape, parts[-1]
        except ValueError:
            return None, None


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

class MidIRToMLIRPass:
    """Convert a :class:`MidFunction` into Triton MLIR text.

    The pass uses a **hybrid** strategy:

    * It first attempts to use the ``triton._C.libtriton.ir`` builder API
      to construct an MLIR module programmatically.
    * If the builder API is unavailable or any call fails, it falls back
      to direct MLIR text generation which produces the same output.

    Both code paths produce valid Triton MLIR that the Triton compiler can
    parse and lower to PTX / GPU binaries.

    Usage::

        from ninetoothed.ir.passes import MidIRToMLIRPass

        mid_func = ...  # a MidFunction instance
        mlir_text = MidIRToMLIRPass().transform(mid_func)
    """

    def __init__(
        self,
        num_warps: int = 4,
        num_ctas: int = 1,
    ) -> None:
        self.num_warps = num_warps
        self.num_ctas = num_ctas

        # Shared state used by the builder path
        self._values: Dict[str, _BuilderValue] = {}
        self._param_types: Dict[str, str] = {}
        self._tile_size: Optional[int] = None
        self._mid_func: Optional[MidFunction] = None
        self._tensor_shapes: Dict[str, List[str]] = {}  # tensor_name -> [dim0, dim1, ...]
        self._concrete_shapes: Dict[str, List[int]] = {}  # tensor_name -> [dim0, dim1, ...] (concrete ints)
        self._tensor_name_to_idx: Dict[str, int] = {}  # tensor param name -> flat index

    def _reset_state(self) -> None:
        self._values.clear()
        self._param_types.clear()
        self._tile_size = None
        self._tensor_shapes.clear()
        self._concrete_shapes.clear()
        self._tensor_name_to_idx.clear()

    def transform(self, mid_func: MidFunction) -> str:
        """Convert *mid_func* to an MLIR module string.

        Parameters
        ----------
        mid_func:
            A :class:`MidFunction` node representing the kernel.

        Returns
        -------
        str
            The full MLIR module as text.
        """
        self._reset_state()

        builder_api = _BuilderAPI()
        if not builder_api.available:
            raise RuntimeError(
                "Triton builder API is not available. "
                "Ensure triton is installed with C++ bindings."
            )
        result = self._transform_with_builder(mid_func, builder_api)
        if result is None:
            raise RuntimeError(
                "Builder API conversion failed. "
                "Check debug logs for details (enable logging.DEBUG for 'ninetoothed.ir.passes')."
            )
        return result

    def _transform_with_builder(self, mid_func: MidFunction, api: _BuilderAPI) -> Optional[str]:
        """Attempt conversion using the Triton builder API.

        Builds the IR using the C++ builder API, verifies it, then generates
        the MLIR text using the text-generation path (which produces
        parseable pretty-printed output).

        Returns ``None`` on any failure so the caller can fall back to text
        generation.
        """
        try:
            self._mid_func = mid_func

            # -- build tensor name → flat index mapping -----------------------
            for i, tp in enumerate(mid_func.tensor_params):
                if isinstance(tp.name, str):
                    self._tensor_name_to_idx[tp.name] = i

            # -- build function signature (type strings) ---------------------
            arg_type_strs: List[str] = []
            ret_type_strs: List[str] = []

            for idx, param in enumerate(mid_func.params):
                if param.is_constexpr:
                    mlir_ty = _mlir_elem_type(param.dtype) if param.dtype else "i32"
                    self._param_types[param.name] = mlir_ty
                else:
                    mlir_ty = self._classify_param_type(param)
                    self._param_types[param.name] = mlir_ty
                arg_type_strs.append(mlir_ty)

            # -- resolve concrete shapes from constexpr symbols -------------------
            constexpr_defaults = self._resolve_constexpr_defaults(mid_func)
            self._concrete_shapes = self._detect_concrete_tensor_shapes(
                mid_func, constexpr_defaults
            )
            self._tile_size = self._detect_tile_size_from_shapes(self._concrete_shapes)

            # Backward-compatible string shapes for legacy consumers
            self._tensor_shapes = {
                name: [str(d) for d in shape]
                for name, shape in self._concrete_shapes.items()
            }

            # -- convert type strings to ir.type objects ----------------------
            arg_ir_types = []
            for ts in arg_type_strs:
                ir_ty = api.get_ir_type(ts)
                if ir_ty is None:
                    return None
                arg_ir_types.append(ir_ty)

            ret_ir_types = [api.get_ir_type(ts) for ts in ret_type_strs if ts]
            ret_ir_types = [t for t in ret_ir_types if t is not None]

            # -- create function via builder --------------------------------
            func = api.create_function(api.mod, mid_func.name, arg_ir_types, ret_ir_types)
            if func is None:
                return None

            # -- map function args into value table --------------------------
            for idx, param in enumerate(mid_func.params):
                block_arg = func.args(idx)
                self._values[param.name] = _BuilderValue(block_arg, arg_type_strs[idx])

            # -- visit invariants then body (builder API) -------------------
            for inv in mid_func.invariants:
                self._visit_stmt_builder(inv, api)

            for stmt in mid_func.body:
                self._visit_stmt_builder(stmt, api)

            # -- return from function (builder API) --------------------------
            api.ret([])

            # -- verify the built IR ----------------------------------------
            try:
                api.mod.verify()
            except Exception:
                logger.debug("Builder API: IR verification failed")
                return None

            logger.info("Builder API: IR construction and verification succeeded")

            # -- serialize the builder's IR directly -----------------------
            # With 'public' function visibility, str_nodebug() returns
            # pretty-printed MLIR that Triton can parse.
            mlir_text = api.module_to_string()
            if mlir_text is None:
                logger.debug("Builder API: module_to_string returned None")
                return None

            return mlir_text

        except Exception as exc:
            logger.warning("Builder API failed: %s", exc, exc_info=True)
            return None

    def _detect_tile_size(self, param: MidParam) -> None:
        """Heuristic: detect the tile size from a param's tile history."""
        for tile_op in param.tile_history:
            if isinstance(tile_op.args, (list, tuple)):
                for a in tile_op.args:
                    if isinstance(a, int) and a > 0:
                        self._tile_size = a
                    elif not isinstance(a, int):
                        # Symbol or other non-int: try string conversion
                        try:
                            val = int(a)
                            if val > 0:
                                self._tile_size = val
                        except (ValueError, TypeError):
                            pass
            elif isinstance(tile_op.args, int):
                self._tile_size = tile_op.args

    def _detect_tensor_shapes(self, mid_func: MidFunction) -> None:
        """Build per-tensor multi-dimensional shape map from tensor_params."""
        for tp in mid_func.tensor_params:
            if tp.tensor is not None:
                innermost = tp.tensor.innermost()
                self._tensor_shapes[tp.name] = [str(s) for s in innermost.shape]

    def _resolve_constexpr_defaults(self, mid_func: MidFunction) -> Dict[str, int]:
        """Map constexpr param names to concrete integer defaults.

        Keys are the *prefixed* names (e.g. ``ninetoothed_constexpr_prefix_BLOCK_SIZE``)
        which match ``str(Symbol(...))`` exactly.
        """
        defaults: Dict[str, int] = {}
        for param in mid_func.params:
            if param.is_constexpr and param.name:
                defaults[param.name] = 1024
        return defaults

    def _resolve_concrete_shape(
        self, shape_tuple: tuple, constexpr_defaults: Dict[str, int]
    ) -> List[int]:
        """Convert a shape tuple of (int | Symbol) into a list of concrete ints."""
        concrete: List[int] = []
        for dim in shape_tuple:
            if isinstance(dim, int):
                concrete.append(dim)
            elif isinstance(dim, Symbol):
                concrete.append(constexpr_defaults.get(str(dim), 1024))
            else:
                try:
                    concrete.append(int(dim))
                except (ValueError, TypeError):
                    concrete.append(1024)
        return concrete

    def _detect_concrete_tensor_shapes(
        self, mid_func: MidFunction, constexpr_defaults: Dict[str, int]
    ) -> Dict[str, List[int]]:
        """Build per-tensor concrete shape map from tensor_params."""
        shapes: Dict[str, List[int]] = {}
        for tp in mid_func.tensor_params:
            if tp.tensor is not None:
                innermost = tp.tensor.innermost()
                shapes[tp.name] = self._resolve_concrete_shape(
                    innermost.shape, constexpr_defaults
                )
        return shapes

    def _detect_tile_size_from_shapes(
        self, concrete_shapes: Dict[str, List[int]]
    ) -> Optional[int]:
        """Derive the global tile size as the largest non-trivial dimension."""
        max_dim: Optional[int] = None
        for shape in concrete_shapes.values():
            for dim in shape:
                if dim > 1 and (max_dim is None or dim > max_dim):
                    max_dim = dim
        return max_dim

    def _classify_param_type(self, param: MidParam) -> str:
        """Determine the MLIR type for a non-constexpr parameter by name pattern.

        * ``*_pointer`` → ``!tt.ptr<elem_type>``
        * ``*_size_*`` / ``*_stride_*`` → ``i32``
        * fallback → ``!tt.ptr<elem_type>``
        """
        name = param.name
        elem = _mlir_elem_type(param.dtype)
        if name.endswith("_pointer"):
            return _ptr_type(elem)
        if "_size_" in name or "_stride_" in name:
            return "i32"
        return _ptr_type(elem)

    # -- builder value helpers ----------------------------------------------

    def _bv(self, name: str) -> Optional[_BuilderValue]:
        """Look up a builder value by name."""
        val = self._values.get(name)
        if isinstance(val, _BuilderValue):
            return val
        return None

    def _native_val(self, name: str):
        """Get the native Triton value for a name."""
        bv = self._bv(name)
        return bv.value if bv is not None else None

    def _type_of(self, name: str) -> Optional[str]:
        """Get the MLIR type string for a name."""
        bv = self._bv(name)
        return bv.type_str if bv is not None else None

    def _store_bv(self, name: str, native_val, type_str: str) -> None:
        """Store a builder value with type tracking."""
        self._values[name] = _BuilderValue(native_val, type_str)

    # -- statement visitors (builder) --------------------------------------

    def _visit_stmt_builder(self, stmt, api: _BuilderAPI) -> None:
        """Visit a single statement using the builder API."""
        if isinstance(stmt, (MidAssign, MidInvariant)):
            val, ty = self._visit_expr_builder_typed(stmt.value, api)
            if val is not None:
                self._store_bv(stmt.target, val, ty or "i32")
                # If target is a tensor param, also emit a store operation
                if isinstance(stmt, MidAssign) and isinstance(stmt.target, str):
                    if stmt.target in self._tensor_name_to_idx:
                        self._try_store_tensor(stmt.target, val, api)
        elif isinstance(stmt, MidStore):
            self._visit_store_builder(stmt, api)
        elif isinstance(stmt, MidReturn):
            api.ret([])
        elif isinstance(stmt, MidExprStmt):
            self._visit_expr_builder(stmt.value, api)
        elif isinstance(stmt, MidFor):
            self._stmt_for_builder(stmt, api)
        elif isinstance(stmt, MidIf):
            self._stmt_if_builder(stmt, api)

    def _stmt_for_builder(self, stmt: MidFor, api: _BuilderAPI) -> None:
        """Emit ``scf.for`` with iter_args via the builder API."""
        # -- resolve upper bound -------------------------------------------
        upper_val, upper_ty = self._visit_expr_builder_typed(stmt.iter_expr, api)
        if upper_val is None:
            return

        # -- detect loop-carried variables (iter_args) ---------------------
        loop_var = stmt.target.name if isinstance(stmt.target, MidName) else str(stmt.target)
        iter_arg_names: List[str] = []
        for body_stmt in stmt.body:
            if isinstance(body_stmt, MidAssign) and isinstance(body_stmt.target, str):
                if body_stmt.target in self._values and body_stmt.target != loop_var:
                    if body_stmt.target not in iter_arg_names:
                        iter_arg_names.append(body_stmt.target)

        # -- create constants ----------------------------------------------
        c0 = api.create_int_constant(0, 32)
        c1 = api.create_int_constant(1, 32)
        if c0 is None or c1 is None:
            return

        # -- build init_values and their ir types ---------------------------
        init_values = []
        for name in iter_arg_names:
            bv = self._bv(name)
            if bv is not None:
                init_values.append(bv.value)
            else:
                return

        # -- create scf.for -------------------------------------------------
        for_op = api.builder.create_for_op(c0, upper_val, c1, init_values)
        if for_op is None:
            return

        body = for_op.get_body(0)
        iv = for_op.get_induction_var()

        # -- set insertion point to loop body ------------------------------
        ip_before = api.builder.get_insertion_block()
        api.builder.set_insertion_point_to_end(body)

        # -- register loop variable and iter_args in body scope ------------
        self._store_bv(loop_var, iv, "i32")
        for idx, name in enumerate(iter_arg_names):
            iter_arg_val = body.arg(idx + 1)  # arg(0) is IV, arg(1+) are iter_args
            old_bv = self._bv(name)
            ty = old_bv.type_str if old_bv else "tensor<64x64xf32>"
            self._store_bv(name, iter_arg_val, ty)

        # -- visit loop body -----------------------------------------------
        for body_stmt in stmt.body:
            self._visit_stmt_builder(body_stmt, api)

        # -- yield iter_args -----------------------------------------------
        yield_values = []
        for name in iter_arg_names:
            v = self._native_val(name)
            if v is not None:
                yield_values.append(v)
            else:
                yield_values = []
                break

        if yield_values:
            api.create_yield(yield_values)

        # -- restore insertion point to after the loop ---------------------
        api.builder.set_insertion_point_to_end(ip_before)

        # -- map iter_arg names to for_op results --------------------------
        for idx, name in enumerate(iter_arg_names):
            result_val = for_op.get_result(idx)
            old_bv = self._bv(name)
            ty = old_bv.type_str if old_bv else "tensor<64x64xf32>"
            self._store_bv(name, result_val, ty)

    def _stmt_if_builder(self, stmt: MidIf, api: _BuilderAPI) -> None:
        """Emit ``scf.if`` via the builder API."""
        cond_val, cond_ty = self._visit_expr_builder_typed(stmt.test, api)
        if cond_val is None:
            return

        has_else = bool(stmt.orelse)

        # Determine result types (empty for void scf.if)
        if_op = api.builder.create_if_op([], cond_val, has_else)
        if if_op is None:
            return

        ip_before = api.builder.get_insertion_block()

        # -- then block -----------------------------------------------------
        then_body = if_op.get_body(0)
        api.builder.set_insertion_point_to_end(then_body)
        for body_stmt in stmt.body:
            self._visit_stmt_builder(body_stmt, api)
        api.create_yield([])

        # -- else block -----------------------------------------------------
        if has_else:
            else_body = if_op.get_body(1)
            api.builder.set_insertion_point_to_end(else_body)
            for else_stmt in stmt.orelse:
                self._visit_stmt_builder(else_stmt, api)
            api.create_yield([])

        api.builder.set_insertion_point_to_end(ip_before)

    def _visit_store_builder(self, stmt: MidStore, api: _BuilderAPI) -> None:
        # Evaluate value first so we can check its rank.
        val_val, val_ty = self._visit_expr_builder_typed(stmt.value, api)

        # Detect 1D-pointer / 2D-value shape mismatch early.
        needs_2d_ptr = False
        if val_val is not None and val_ty is not None:
            val_dims = _shape_of(val_ty)
            if val_dims is not None and len(val_dims) >= 2:
                needs_2d_ptr = True

        if needs_2d_ptr:
            # Skip the original 1D pointer/mask evaluation entirely to avoid
            # emitting invalid ops into the module.
            ptr_val, mask_val = self._fix_2d_store_ptr(
                stmt, None, None, api, val_dims
            )
            ptr_ty = None
        else:
            ptr_val, ptr_ty = self._visit_expr_builder_typed(
                stmt.pointer, api
            )
            mask_val = None
            if stmt.mask is not None:
                mask_val, _ = self._visit_expr_builder_typed(stmt.mask, api)

        if ptr_val is not None and val_val is not None:
            if mask_val is not None:
                api.builder.create_masked_store(
                    ptr_val, val_val, mask_val,
                    api._tl_ir.CACHE_MODIFIER.NONE,
                    api._tl_ir.EVICTION_POLICY.NORMAL,
                )
            else:
                api.builder.create_store(
                    ptr_val, val_val,
                    api._tl_ir.CACHE_MODIFIER.NONE,
                    api._tl_ir.EVICTION_POLICY.NORMAL,
                )

    # -- expression visitors (builder) ------------------------------------

    def _visit_expr_builder(self, expr, api: _BuilderAPI):  # noqa: C901
        """Visit an expression, returning the native Triton value (or None)."""
        val, _ = self._visit_expr_builder_typed(expr, api)
        return val

    def _visit_expr_builder_typed(self, expr, api: _BuilderAPI):  # noqa: C901
        """Visit an expression, returning (native_value, type_string) or (None, None)."""
        if isinstance(expr, MidCall):
            pass  # handled by _call_builder below
        if isinstance(expr, MidName):
            bv = self._bv(expr.name)
            if bv is not None:
                # For tensor params stored as pointers, emit a load.
                # 0D tensors need a simple scalar load; ND tensors use
                # the full offset computation in _try_load_tensor.
                if bv.type_str and bv.type_str.startswith("!tt.ptr<"):
                    # Check original tensor name (e.g. "eps")
                    is_tensor = expr.name in self._tensor_name_to_idx
                    # Check generated tensor name (e.g. "ninetoothed_tensor_1")
                    is_0d_ref = False
                    if not is_tensor:
                        m = _TENSOR_BARE_NAME_RE.match(expr.name)
                        if m:
                            idx = int(m.group(1))
                            # 0D tensors have no size/stride invariants
                            has_size = self._bv(
                                f"ninetoothed_ninetoothed_tensor_{idx}_size_0"
                            ) is not None
                            has_stride = self._bv(
                                f"ninetoothed_ninetoothed_tensor_{idx}_stride_0"
                            ) is not None
                            if not has_size and not has_stride:
                                is_0d_ref = True
                    if is_tensor:
                        shape = self._tensor_shapes.get(expr.name, [])
                        if len(shape) == 0:
                            loaded = api.builder.create_load(
                                bv.value,
                                api._tl_ir.CACHE_MODIFIER.NONE,
                                api._tl_ir.EVICTION_POLICY.NORMAL, False,
                            )
                            if loaded is not None:
                                return loaded, _elem_of(bv.type_str)
                        try:
                            result = self._try_load_tensor(expr.name, api)
                            if result[0] is not None:
                                return result
                        except Exception:
                            pass
                    elif is_0d_ref:
                        # 0D tensor referenced by generated name: scalar load
                        loaded = api.builder.create_load(
                            bv.value,
                            api._tl_ir.CACHE_MODIFIER.NONE,
                            api._tl_ir.EVICTION_POLICY.NORMAL, False,
                        )
                        if loaded is not None:
                            return loaded, _elem_of(bv.type_str)
                return bv.value, bv.type_str
            # Try loading a tensor by name (for tensors used in intermediate
            # computations where the AST pass generates MidName instead of
            # MidLoad).
            try:
                result = self._try_load_tensor(expr.name, api)
            except Exception:
                result = None, None
            return result

        if isinstance(expr, MidConstant):
            val = self._constant_builder(expr, api)
            if val is not None:
                ty = self._constant_type(expr)
                return val, ty
            return None, None

        if isinstance(expr, MidBinOp):
            return self._binop_builder(expr, api)

        if isinstance(expr, MidCompare):
            return self._compare_builder(expr, api)

        if isinstance(expr, MidBoolOp):
            return self._boolop_builder(expr, api)

        if isinstance(expr, MidUnaryOp):
            return self._unaryop_builder(expr, api)

        if isinstance(expr, MidLoad):
            return self._load_builder(expr, api)

        if isinstance(expr, MidPointerExpr):
            return self._pointer_builder(expr, api)

        if isinstance(expr, MidMaskExpr):
            return self._mask_builder(expr, api)

        if isinstance(expr, MidProgramId):
            val = api.create_get_program_id(expr.axis)
            return val, "i32"

        if isinstance(expr, MidArange):
            return self._arange_builder(expr, api)

        if isinstance(expr, MidCall):
            return self._call_builder(expr, api)

        if isinstance(expr, MidSubscript):
            return self._subscript_builder(expr, api)

        if isinstance(expr, MidTuple):
            # Visit all elements; tuple itself produces no single value
            for elt in expr.elts:
                self._visit_expr_builder(elt, api)
            return None, None

        # Metadata nodes that produce no MLIR instructions
        if isinstance(expr, (MidTile, MidTensorAccess, MidDataPtr,
                            MidOffsets, MidStride, MidDtypeAttr)):
            return None, None

        return None, None

    def _resolve_int(self, node) -> int:
        """Resolve a Mid IR node to an integer."""
        if isinstance(node, MidConstant):
            return int(node.value)
        if isinstance(node, MidName):
            return 0
        if isinstance(node, int):
            return node
        return 0

    def _constant_type(self, expr: MidConstant) -> str:
        """Return the MLIR type string for a constant expression."""
        if isinstance(expr.value, bool):
            return "i1"
        if isinstance(expr.value, int):
            return "i32"
        if isinstance(expr.value, float):
            return "f32"
        return "i32"

    def _constant_builder(self, expr: MidConstant, api: _BuilderAPI):
        # Reuse constants to avoid duplicates
        if isinstance(expr.value, bool):
            return api.create_int_constant(int(expr.value), 1)
        if isinstance(expr.value, int):
            return api.create_int_constant(expr.value, 32)
        if isinstance(expr.value, float):
            return api.create_float_constant(expr.value, "f32")
        return None

    # -- broadcasting helpers (builder) ------------------------------------

    def _splat_builder(self, api: _BuilderAPI, scalar_val, scalar_type: str,
                        tensor_type: str) -> Tuple:
        """Splat a scalar value to a tensor type via the builder API.

        For pointer element types (e.g. ``tensor<64x64x!tt.ptr<f32>>``),
        uses the pattern: ``ptr_to_int → splat → int_to_ptr`` because the
        Triton builder's ``create_splat`` does not support pointer elements.
        """
        elem = _elem_of(tensor_type)
        is_ptr_elem = elem.startswith("!tt.ptr<")

        if is_ptr_elem and scalar_type and scalar_type.startswith("!tt.ptr<"):
            # Pointer splat: ptr_to_int → splat i64 → int_to_ptr
            i64_tensor_type = _replace_tensor_elem(tensor_type, "i64")
            ir_i64_tensor_ty = api.get_ir_type(i64_tensor_type)
            if ir_i64_tensor_ty is None:
                return scalar_val, tensor_type
            int_val = api.create_ptr_to_int(scalar_val)
            if int_val is None:
                return scalar_val, tensor_type
            int_tensor = api.builder.create_splat(ir_i64_tensor_ty, int_val)
            ir_ptr_tensor_ty = api.get_ir_type(tensor_type)
            if ir_ptr_tensor_ty is None:
                return scalar_val, tensor_type
            ptr_tensor = api.create_int_to_ptr(int_tensor, ir_ptr_tensor_ty)
            return ptr_tensor, tensor_type

        ir_tensor_ty = api.get_ir_type(tensor_type)
        if ir_tensor_ty is None:
            return scalar_val, tensor_type
        splatted = api.builder.create_splat(ir_tensor_ty, scalar_val)
        return splatted, tensor_type

    def _reshape_1d_to_2d_builder(self, api: _BuilderAPI, val, type_str: str,
                                   target_dims: List[int]) -> Tuple:
        """Reshape a 1D tensor to 2D via the builder API."""
        shape_1d = _shape_of(type_str)
        if shape_1d is None or len(shape_1d) != 1 or len(target_dims) != 2:
            return val, type_str
        dim0 = shape_1d[0]
        elem = _elem_of(type_str)
        if target_dims[0] == 1 and target_dims[1] == dim0:
            new_shape = [1, dim0]
        elif target_dims[0] == dim0 and target_dims[1] == 1:
            new_shape = [dim0, 1]
        elif dim0 == 1:
            # Single-element 1D tensor: reshape to [1, 1] so that the
            # subsequent 2D-to-2D broadcast can expand it to the target
            # shape (e.g. tensor<1xf32> → tensor<1x1xf32> → broadcast
            # to tensor<1xBLOCK_SIZExf32>).
            new_shape = [1, 1]
        else:
            return val, type_str
        new_type = _tensor_type(new_shape, elem)
        reshaped = api.builder.create_reshape(val, new_shape, False)
        return reshaped, new_type

    def _broadcast_2d_to_2d_builder(self, api: _BuilderAPI, val, type_str: str,
                                     other_dims: List[int]) -> Tuple:
        """Broadcast a 2D tensor to match another 2D tensor's shape."""
        my_dims = _shape_of(type_str)
        if my_dims is None or len(my_dims) != 2 or len(other_dims) != 2:
            return val, type_str
        if my_dims == other_dims:
            return val, type_str
        elem = _elem_of(type_str)
        bc_dims = [max(d1, d2) for d1, d2 in zip(my_dims, other_dims)]
        if bc_dims == my_dims:
            return val, type_str
        new_type = _tensor_type(bc_dims, elem)
        broadcasted = api.builder.create_broadcast(val, bc_dims)
        return broadcasted, new_type

    def _expand_dims_builder(self, api: _BuilderAPI, val, type_str: str,
                              axis: int, target_dims: List[int]) -> Tuple:
        """Expand dims of a tensor via the builder API."""
        my_dims = _shape_of(type_str)
        if my_dims is None or len(my_dims) != 2:
            return val, type_str
        elem = _elem_of(type_str)
        new_type = _tensor_type(target_dims, elem)
        expanded = api.builder.create_expand_dims(val, axis)
        return expanded, new_type

    def _ensure_matching_types_builder(self, api: _BuilderAPI, lhs, rhs,
                                        lhs_type, rhs_type) -> Tuple:
        """Ensure lhs and rhs have matching types via splat/reshape/broadcast."""
        if lhs_type is None:
            lhs_type = "i32"
        if rhs_type is None:
            rhs_type = "i32"

        # Promote int to float when one operand is float
        if _is_float_type(lhs_type) and _is_int_type(rhs_type):
            if _is_tensor_type(rhs_type):
                cast_type = _replace_tensor_elem(rhs_type, _elem_of(lhs_type))
            else:
                cast_type = _elem_of(lhs_type)
            rhs = api.create_si_to_fp(rhs)
            rhs_type = cast_type
        elif _is_float_type(rhs_type) and _is_int_type(lhs_type):
            if _is_tensor_type(lhs_type):
                cast_type = _replace_tensor_elem(lhs_type, _elem_of(rhs_type))
            else:
                cast_type = _elem_of(rhs_type)
            lhs = api.create_si_to_fp(lhs)
            lhs_type = cast_type

        lhs_is_tensor = _is_tensor_type(lhs_type)
        rhs_is_tensor = _is_tensor_type(rhs_type)

        # Splat scalar to tensor
        if lhs_is_tensor and not rhs_is_tensor:
            tensor_ty = _replace_tensor_elem(lhs_type, _elem_of(rhs_type))
            rhs, rhs_type = self._splat_builder(api, rhs, rhs_type, tensor_ty)
        elif rhs_is_tensor and not lhs_is_tensor:
            tensor_ty = _replace_tensor_elem(rhs_type, _elem_of(lhs_type))
            lhs, lhs_type = self._splat_builder(api, lhs, lhs_type, tensor_ty)

        lhs_is_tensor = _is_tensor_type(lhs_type)
        rhs_is_tensor = _is_tensor_type(rhs_type)

        # Broadcast 1D tensors: [1] → [N] when the other is [N]
        if lhs_is_tensor and rhs_is_tensor:
            lhs_dims = _shape_of(lhs_type) or []
            rhs_dims = _shape_of(rhs_type) or []
            if len(lhs_dims) == 1 and len(rhs_dims) == 1 and lhs_dims != rhs_dims:
                if lhs_dims[0] == 1 and rhs_dims[0] > 1:
                    new_type = rhs_type
                    lhs = api.builder.create_broadcast(lhs, rhs_dims)
                    lhs_type = new_type
                elif rhs_dims[0] == 1 and lhs_dims[0] > 1:
                    new_type = lhs_type
                    rhs = api.builder.create_broadcast(rhs, lhs_dims)
                    rhs_type = new_type

        # Reshape 1D to 2D
        if lhs_is_tensor and rhs_is_tensor:
            lhs_dims = _shape_of(lhs_type) or []
            rhs_dims = _shape_of(rhs_type) or []
            if len(lhs_dims) == 1 and len(rhs_dims) == 2:
                lhs, lhs_type = self._reshape_1d_to_2d_builder(api, lhs, lhs_type, rhs_dims)
            elif len(rhs_dims) == 1 and len(lhs_dims) == 2:
                rhs, rhs_type = self._reshape_1d_to_2d_builder(api, rhs, rhs_type, lhs_dims)

        # Broadcast 2D to 2D
        if lhs_is_tensor and rhs_is_tensor:
            lhs_dims = _shape_of(lhs_type) or []
            rhs_dims = _shape_of(rhs_type) or []
            if (len(lhs_dims) == 2 and len(rhs_dims) == 2
                    and lhs_dims != rhs_dims):
                lhs, lhs_type = self._broadcast_2d_to_2d_builder(api, lhs, lhs_type, rhs_dims)
                rhs, rhs_type = self._broadcast_2d_to_2d_builder(api, rhs, rhs_type, lhs_dims)

        return lhs, rhs, lhs_type, rhs_type

    # -- binary operation builder ------------------------------------------

    def _binop_builder(self, expr: MidBinOp, api: _BuilderAPI) -> Tuple:
        op = expr.op

        # -- dot+acc fusion ------------------------------------------------
        if op == "+":
            fused = self._try_dot_acc_fusion(expr, api)
            if fused is not None:
                return fused

        lhs, lhs_type = self._visit_expr_builder_typed(expr.lhs, api)
        rhs, rhs_type = self._visit_expr_builder_typed(expr.rhs, api)
        if lhs is None or rhs is None:
            return None, None

        # -- pointer arithmetic: splat + addptr -----------------------------
        if op == "+":
            ptr_result = self._try_ptr_addptr_builder(lhs, lhs_type, rhs, rhs_type, api)
            if ptr_result is not None:
                return ptr_result
            ptr_result = self._try_ptr_addptr_builder(rhs, rhs_type, lhs, lhs_type, api)
            if ptr_result is not None:
                return ptr_result

        # -- ensure matching types -----------------------------------------
        lhs, rhs, lhs_type, rhs_type = self._ensure_matching_types_builder(
            api, lhs, rhs, lhs_type, rhs_type
        )

        result_type = _broadcast_type(lhs_type, rhs_type)
        is_float = _is_float_type(result_type)

        # -- promote int operands to float for float operations ----------
        if is_float:
            if _is_int_type(lhs_type) and _is_scalar_type(lhs_type):
                lhs = api.create_si_to_fp(lhs)
                lhs_type = "f32"
            if _is_int_type(rhs_type) and _is_scalar_type(rhs_type):
                rhs = api.create_si_to_fp(rhs)
                rhs_type = "f32"
            # Splat scalars to match tensor operands for fdiv/fmul etc.
            if _is_tensor_type(lhs_type) and _is_scalar_type(rhs_type):
                ir_rhs_ty = api.get_ir_type(lhs_type)
                if ir_rhs_ty is not None:
                    rhs = api.builder.create_splat(ir_rhs_ty, rhs)
                    rhs_type = lhs_type
            elif _is_tensor_type(rhs_type) and _is_scalar_type(lhs_type):
                ir_lhs_ty = api.get_ir_type(rhs_type)
                if ir_lhs_ty is not None:
                    lhs = api.builder.create_splat(ir_lhs_ty, lhs)
                    lhs_type = rhs_type
            result_type = _broadcast_type(lhs_type, rhs_type)
            is_float = _is_float_type(result_type)

        if op == "+":
            return api.create_add(lhs, rhs, is_float), result_type
        if op == "-":
            return api.create_sub(lhs, rhs, is_float), result_type
        if op == "*":
            return api.create_mul(lhs, rhs, is_float), result_type
            return api.create_mul(lhs, rhs, is_float), result_type
        if op == "/":
            if is_float:
                return api.create_div(lhs, rhs, True), result_type
            return api.create_div(lhs, rhs, False), result_type
        if op == "//":
            return api.create_div(lhs, rhs, False), result_type
        if op == "%":
            if is_float:
                return api.create_rem(lhs, rhs, True), result_type
            return api.create_rem(lhs, rhs, False), result_type
        if op == "&":
            return api.create_and(lhs, rhs), result_type
        if op == "|":
            return api.create_or(lhs, rhs), result_type
        if op == "^":
            return api.create_xor(lhs, rhs), result_type
        if op == "<<":
            return api.create_shl(lhs, rhs), result_type
        if op == ">>":
            return api.create_ashr(lhs, rhs), result_type

        return None, None

    def _try_ptr_addptr_builder(self, ptr_val, ptr_type, offset_val, offset_type,
                                 api: _BuilderAPI) -> Optional[Tuple]:
        """Try to emit ptr + offset as splat + addptr."""
        if ptr_type is None or not ptr_type.startswith("!tt.ptr<"):
            return None
        # addptr requires an integer offset — reject float offsets
        if offset_type is not None and _is_float_type(offset_type):
            return None
        elem = _elem_of(ptr_type)
        offset_dims = _shape_of(offset_type) or [self._tile_size or 1024]
        ptr_tensor_ty = _ptr_tensor_type(elem, offset_dims)
        splatted_ptr, _ = self._splat_builder(api, ptr_val, ptr_type, ptr_tensor_ty)
        result = api.builder.create_addptr(splatted_ptr, offset_val)
        return result, ptr_tensor_ty

    def _try_dot_acc_fusion(self, expr: MidBinOp, api: _BuilderAPI) -> Optional[Tuple]:
        """Detect acc + dot(a, b) and fuse into tt.dot a, b, acc."""
        dot_side = None
        acc_side = None
        if isinstance(expr.rhs, MidCall):
            rhs_base = expr.rhs.func.rsplit(".", 1)[-1] if "." in expr.rhs.func else expr.rhs.func
            if rhs_base == "dot" and len(expr.rhs.args) >= 2:
                dot_side, acc_side = expr.rhs, expr.lhs
        elif isinstance(expr.lhs, MidCall):
            lhs_base = expr.lhs.func.rsplit(".", 1)[-1] if "." in expr.lhs.func else expr.lhs.func
            if lhs_base == "dot" and len(expr.lhs.args) >= 2:
                dot_side, acc_side = expr.lhs, expr.rhs
        if dot_side is None:
            return None

        a_val, a_type = self._visit_expr_builder_typed(dot_side.args[0], api)
        b_val, b_type = self._visit_expr_builder_typed(dot_side.args[1], api)
        acc_val, acc_type = self._visit_expr_builder_typed(acc_side, api)
        if a_val is None or b_val is None or acc_val is None:
            return None

        # Expand 1D operands to 2D for tt.dot (requires 2D+ operands).
        # [K] -> [K, 1] and [K] -> [1, K] so dot produces [K, K] outer product.
        a_dims = _shape_of(a_type) if a_type else None
        b_dims = _shape_of(b_type) if b_type else None
        if (a_dims is not None and len(a_dims) == 1
                and b_dims is not None and len(b_dims) == 1):
            k_dim = a_dims[0]
            a_elem = _elem_of(a_type)
            b_elem = _elem_of(b_type)
            a_val = api.builder.create_reshape(a_val, [k_dim, 1], False)
            a_type = _tensor_type([k_dim, 1], a_elem)
            b_val = api.builder.create_reshape(b_val, [1, k_dim], False)
            b_type = _tensor_type([1, k_dim], b_elem)

        result_type = self._resolve_dot_result_type()
        dot_result = api.builder.create_dot(
            a_val, b_val, acc_val,
            api._tl_ir.INPUT_PRECISION.TF32, 0,
        )
        return dot_result, result_type

    # -- comparison / boolop / unary builders ------------------------------

    def _compare_builder(self, expr: MidCompare, api: _BuilderAPI) -> Tuple:
        lhs, lhs_type = self._visit_expr_builder_typed(expr.left, api)
        rhs, rhs_type = self._visit_expr_builder_typed(expr.right, api)
        if lhs is None or rhs is None:
            return None, None

        # Splat for scalar-tensor comparisons
        lhs, rhs, lhs_type, rhs_type = self._ensure_matching_types_builder(
            api, lhs, rhs, lhs_type, rhs_type
        )

        operand_type = _broadcast_type(lhs_type, rhs_type)
        is_float = _is_float_type(operand_type)

        cmp_map_int = {
            "==": "EQ", "!=": "NE",
            "<": "SLT", "<=": "SLE",
            ">": "SGT", ">=": "SGE",
        }
        cmp_map_float = {
            "==": "OEQ", "!=": "UNE",
            "<": "OLT", "<=": "OLE",
            ">": "OGT", ">=": "OGE",
        }
        pred = (cmp_map_float if is_float else cmp_map_int).get(expr.op)
        if pred is None:
            return None, None

        if is_float:
            method = getattr(api.builder, f"create_fcmp{pred}", None)
        else:
            method = getattr(api.builder, f"create_icmp{pred}", None)
        if method is None:
            return None, None

        result = method(lhs, rhs)
        if _is_tensor_type(operand_type):
            res_type = _replace_tensor_elem(operand_type, "i1")
        else:
            res_type = "i1"
        return result, res_type

    def _boolop_builder(self, expr: MidBoolOp, api: _BuilderAPI) -> Tuple:
        dialect = "andi" if expr.op == "and" else "ori"
        result, res_type = self._visit_expr_builder_typed(expr.values[0], api)
        if result is None:
            return None, None

        for cond in expr.values[1:]:
            result, res_type = self._fold_cond_with_splat(result, res_type, cond, dialect, api)
            if result is None:
                return None, None
        return result, res_type

    def _fold_cond_with_splat(self, result, res_type, cond_node, op_name, api):
        """Fold a condition into an accumulated result with splat type matching."""
        rhs, rhs_type = self._visit_expr_builder_typed(cond_node, api)
        if rhs is None:
            return None, None

        # Splat scalar to tensor when types don't match
        if _is_tensor_type(res_type) and not _is_tensor_type(rhs_type) and rhs_type is not None:
            tensor_ty = _replace_tensor_elem(res_type, _elem_of(rhs_type))
            rhs, rhs_type = self._splat_builder(api, rhs, rhs_type, tensor_ty)
        elif not _is_tensor_type(res_type) and _is_tensor_type(rhs_type):
            res_type = rhs_type

        method = getattr(api.builder, f"create_{op_name}", None)
        if method is None:
            return None, None
        return method(result, rhs), res_type

    def _unaryop_builder(self, expr: MidUnaryOp, api: _BuilderAPI) -> Tuple:
        operand, op_type = self._visit_expr_builder_typed(expr.operand, api)
        if operand is None:
            return None, None
        if expr.op == "-":
            if _is_float_type(op_type):
                return None, None  # negf not directly available
            zero = api.create_int_constant(0, 32)
            if zero is None:
                return None, None
            return api.create_sub(zero, operand, False), op_type
        return None, None

    # -- load / pointer / mask builders -----------------------------------

    def _load_builder(self, expr: MidLoad, api: _BuilderAPI) -> Tuple:
        # Try 2D decomposition from the pointer expression structure.
        # This correctly handles loop variables (e.g. k in matmul).
        decomp = self._decompose_2d_pointer(expr.pointer)
        if decomp is not None and len(decomp[1]) == 2:
            result = self._load_2d_builder(expr, api, decomp)
            if result[0] is not None:
                return result

        ptr_val, ptr_type = self._visit_expr_builder_typed(expr.pointer, api)
        if ptr_val is None:
            return None, None

        mask_val = None
        if expr.mask is not None:
            mask_val, _ = self._visit_expr_builder_typed(expr.mask, api)

        other_val = None
        if expr.other is not None:
            other_val, _ = self._visit_expr_builder_typed(expr.other, api)

        # Use masked_load when mask is present
        if mask_val is not None:
            loaded = api.builder.create_masked_load(
                ptr_val, mask_val, other_val,
                api._tl_ir.CACHE_MODIFIER.NONE,
                api._tl_ir.EVICTION_POLICY.NORMAL, False,
            )
        else:
            loaded = api.builder.create_load(
                ptr_val,
                api._tl_ir.CACHE_MODIFIER.NONE,
                api._tl_ir.EVICTION_POLICY.NORMAL, False,
            )

        if loaded is None:
            return None, None

        # Determine result type from pointer
        result_type = self._resolve_load_type(expr.pointer)
        return loaded, result_type

    # -- 2D pointer decomposition helpers --------------------------------

    def _decompose_2d_pointer(self, pointer_expr):
        """Decompose a 2D pointer expression into per-dimension components.

        Expected structure::

            base_pointers + (dim0_off * stride_0) + (dim1_off * stride_1)

        where each ``dim_off = start_expr * block_size + arange(block_size)``.

        Returns ``(base_name, [(start_expr, block_size, stride_name), ...])``
        or ``None`` if the expression cannot be decomposed.
        """
        # Top level: base + overall_offsets
        if not isinstance(pointer_expr, MidBinOp) or pointer_expr.op != "+":
            return None

        lhs, rhs = pointer_expr.lhs, pointer_expr.rhs
        base_name = None
        offsets_expr = None
        if isinstance(lhs, MidName) and "pointers" in lhs.name:
            base_name = lhs.name
            offsets_expr = rhs
        elif isinstance(rhs, MidName) and "pointers" in rhs.name:
            base_name = rhs.name
            offsets_expr = lhs
        if base_name is None or offsets_expr is None:
            return None

        # overall_offsets = dim0_term + dim1_term
        if not isinstance(offsets_expr, MidBinOp) or offsets_expr.op != "+":
            return None

        dims = []
        for term in [offsets_expr.lhs, offsets_expr.rhs]:
            if not isinstance(term, MidBinOp) or term.op != "*":
                return None
            # term = dim_off * stride  (or stride * dim_off)
            if isinstance(term.rhs, MidName) and "stride" in term.rhs.name:
                dim_off, stride_name = term.lhs, term.rhs.name
            elif isinstance(term.lhs, MidName) and "stride" in term.lhs.name:
                dim_off, stride_name = term.rhs, term.lhs.name
            else:
                return None

            # dim_off = start_part + arange  (top-level '+')
            arange = _find_arange_in_expr(dim_off)
            if arange is None:
                return None

            # Get block_size from arange value
            block_size = self._tile_size or 1024
            if isinstance(arange.value, MidName):
                bv = self._bv(arange.value.name)
                if bv is not None:
                    # Try to resolve from constexpr defaults
                    block_size = self._resolve_int_const(arange.value) or block_size
            elif isinstance(arange.value, MidConstant) and isinstance(arange.value.value, int):
                block_size = arange.value.value

            # Strip arange to get start_expr (the non-arange part)
            start_expr = _strip_arange_from_expr(dim_off)
            if start_expr is None:
                return None

            dims.append((start_expr, block_size, stride_name))

        return base_name, dims

    def _load_2d_builder(self, expr: MidLoad, api: _BuilderAPI,
                         decomp: tuple) -> Tuple:
        """Generate a 2D block load using decomposed pointer components.

        *decomp* is the result of :meth:`_decompose_2d_pointer`:
        ``(base_name, [(start_expr, block_size, stride_name), ...])``.
        """
        base_name, dims = decomp
        tile_m = dims[0][1]
        tile_n = dims[1][1]
        stride_0_name = dims[0][2]
        stride_1_name = dims[1][2]

        # Look up base pointer and strides
        ptr_bv = self._bv(base_name)
        stride_0_bv = self._bv(stride_0_name)
        stride_1_bv = self._bv(stride_1_name)

        # Resolve tensor index for size lookups
        target_idx = None
        for inv in (self._mid_func.invariants or []):
            inv_name = inv.target if isinstance(inv.target, str) else str(inv.target)
            if inv_name == base_name:
                m = _TENSOR_BARE_NAME_RE.match(base_name.split("_pointers")[0].rstrip("_"))
                if m:
                    candidate = int(m.group(1))
                    if candidate in self._tensor_name_to_idx.values():
                        target_idx = candidate
                break

        size_0_bv = None
        size_1_bv = None
        if target_idx is not None:
            size_0_bv = self._bv(
                f"ninetoothed_ninetoothed_tensor_{target_idx}_size_0"
            )
            size_1_bv = self._bv(
                f"ninetoothed_ninetoothed_tensor_{target_idx}_size_1"
            )

        if any(v is None for v in [ptr_bv, stride_0_bv, stride_1_bv]):
            return None, None

        elem_type = "f32"
        if ptr_bv.type_str and "!tt.ptr<" in ptr_bv.type_str:
            elem_type = ptr_bv.type_str.split("<")[1].rstrip(">")

        # Evaluate per-dimension start expressions as scalars.
        # These naturally include loop variables (e.g. k) when inside a loop.
        dim0_start_val, _ = self._visit_expr_builder_typed(dims[0][0], api)
        dim1_start_val, _ = self._visit_expr_builder_typed(dims[1][0], api)
        if dim0_start_val is None or dim1_start_val is None:
            return None, None

        # Row indices: (dim0_start + arange(0, tile_m)) → [tile_m, 1]
        row_ir_ty = api.get_ir_type(f"tensor<{tile_m}xi32>")
        row_arange = api.create_make_range(0, tile_m)
        row_splat = api.builder.create_splat(row_ir_ty, dim0_start_val)
        row_idx_1d = api.builder.create_add(row_splat, row_arange)
        row_idx_2d = api.builder.create_expand_dims(row_idx_1d, 1)

        # Col indices: (dim1_start + arange(0, tile_n)) → [1, tile_n]
        col_ir_ty = api.get_ir_type(f"tensor<{tile_n}xi32>")
        col_arange = api.create_make_range(0, tile_n)
        col_splat = api.builder.create_splat(col_ir_ty, dim1_start_val)
        col_idx_1d = api.builder.create_add(col_splat, col_arange)
        col_idx_2d = api.builder.create_expand_dims(col_idx_1d, 0)

        # 2D offsets: stride splatted to match 2D index shapes, then broadcast
        row_ty = api.get_ir_type(f"tensor<{tile_m}x1xi32>")
        col_ty = api.get_ir_type(f"tensor<1x{tile_n}xi32>")
        s0_splat = api.builder.create_splat(row_ty, stride_0_bv.value)
        s1_splat = api.builder.create_splat(col_ty, stride_1_bv.value)
        row_off = api.builder.create_mul(row_idx_2d, s0_splat)
        col_off = api.builder.create_mul(col_idx_2d, s1_splat)
        row_off_bc = api.builder.create_broadcast(row_off, [tile_m, tile_n])
        col_off_bc = api.builder.create_broadcast(col_off, [tile_m, tile_n])
        total_off = api.builder.create_add(row_off_bc, col_off_bc)

        # 2D pointer tensor
        ptr_tensor_type = f"tensor<{tile_m}x{tile_n}x!tt.ptr<{elem_type}>>"
        ptr_tensor, _ = self._splat_builder(
            api, ptr_bv.value, ptr_bv.type_str or f"!tt.ptr<{elem_type}>",
            ptr_tensor_type,
        )
        final_ptr = api.builder.create_addptr(ptr_tensor, total_off)

        # 2D boundary mask: check actual element indices (start + arange) < size
        mask_val = None
        if size_0_bv is not None or size_1_bv is not None:
            row_mask = None
            if size_0_bv is not None:
                sz_ty = api.get_ir_type(f"tensor<{tile_m}xi32>")
                sz_splat = api.builder.create_splat(sz_ty, size_0_bv.value)
                row_mask_1d = api.builder.create_icmpSLT(row_idx_1d, sz_splat)
                row_mask = api.builder.create_expand_dims(row_mask_1d, 1)

            col_mask = None
            if size_1_bv is not None:
                sz_ty = api.get_ir_type(f"tensor<{tile_n}xi32>")
                sz_splat = api.builder.create_splat(sz_ty, size_1_bv.value)
                col_mask_1d = api.builder.create_icmpSLT(col_idx_1d, sz_splat)
                col_mask = api.builder.create_expand_dims(col_mask_1d, 0)

            if row_mask is not None and col_mask is not None:
                row_mask_bc = api.builder.create_broadcast(row_mask, [tile_m, tile_n])
                col_mask_bc = api.builder.create_broadcast(col_mask, [tile_m, tile_n])
                mask_val = api.builder.create_and(row_mask_bc, col_mask_bc)
            elif row_mask is not None:
                mask_val = api.builder.create_broadcast(row_mask, [tile_m, tile_n])
            elif col_mask is not None:
                mask_val = api.builder.create_broadcast(col_mask, [tile_m, tile_n])

        # Load
        result_type = f"tensor<{tile_m}x{tile_n}x{elem_type}>"
        if mask_val is not None:
            other_ir_ty = api.get_ir_type(result_type)
            zero = api.create_float_constant(0.0, elem_type)
            other_splat = api.builder.create_splat(other_ir_ty, zero)
            loaded = api.builder.create_masked_load(
                final_ptr, mask_val, other_splat,
                api._tl_ir.CACHE_MODIFIER.NONE,
                api._tl_ir.EVICTION_POLICY.NORMAL, False,
            )
        else:
            loaded = api.builder.create_load(
                final_ptr,
                api._tl_ir.CACHE_MODIFIER.NONE,
                api._tl_ir.EVICTION_POLICY.NORMAL, False,
            )

        if loaded is None:
            return None, None
        return loaded, result_type

    # -- tensor name resolution (builder) ----------------------------------

    def _try_load_tensor(self, name: str, api: _BuilderAPI) -> Tuple:
        """Try to generate a load for a tensor param referenced by *name*.

        When the AST pass generates ``MidName('input')`` instead of a
        ``MidLoad`` (e.g. for intermediate computations like
        ``ntl.cast(input, dtype)``), this method synthesises a load using
        the invariant-computed pointer, indices, and strides.
        """
        idx = self._tensor_name_to_idx.get(name)
        if idx is None:
            return None, None

        tile_size = self._tile_size or 1024

        # Gather invariant-computed values
        ptr_bv = self._bv(f"ninetoothed_tensor_{idx}_pointers")
        idx_0_bv = self._bv(f"ninetoothed_tensor_{idx}_index_0")
        idx_1_bv = self._bv(f"ninetoothed_tensor_{idx}_index_1")
        stride_0_bv = self._bv(f"ninetoothed_ninetoothed_tensor_{idx}_stride_0")
        stride_1_bv = self._bv(f"ninetoothed_ninetoothed_tensor_{idx}_stride_1")
        size_1_bv = self._bv(f"ninetoothed_ninetoothed_tensor_{idx}_size_1")

        if ptr_bv is None or idx_0_bv is None:
            return None, None

        # Determine element type from pointer type
        elem_type = "f32"
        if ptr_bv.type_str and "!tt.ptr<" in ptr_bv.type_str:
            elem_type = ptr_bv.type_str.split("<")[1].rstrip(">")

        # Detect dimensionality and determine result shape
        is_2d = stride_1_bv is not None
        # For a 2D tensor tiled as (1, BLOCK_SIZE), the load produces
        # a 1D tensor of shape (BLOCK_SIZE,).  We use tile_size for the
        # single dimension.
        result_type = f"tensor<{tile_size}x{elem_type}>"

        # Create arange
        arange = api.create_make_range(0, tile_size)
        arange_type = f"tensor<{tile_size}xi32>"

        if is_2d and stride_0_bv is not None and stride_1_bv is not None:
            # 2D tensor: offset = index_0 * stride_0 + arange * stride_1
            row_off = api.builder.create_mul(idx_0_bv.value, stride_0_bv.value)
            row_ir_ty = api.get_ir_type(arange_type)
            row_splat = api.builder.create_splat(row_ir_ty, row_off)

            s1_splat = api.builder.create_splat(row_ir_ty, stride_1_bv.value)
            col_off = api.builder.create_mul(arange, s1_splat)

            total_off = api.builder.create_add(row_splat, col_off)
        elif stride_0_bv is not None:
            # 1D tensor: offset = (index_0 * tile_size + arange) * stride_0
            block_val = api.create_int_constant(tile_size, 32)
            idx_t = api.builder.create_mul(idx_0_bv.value, block_val)
            ir_ty = api.get_ir_type(arange_type)
            idx_splat = api.builder.create_splat(ir_ty, idx_t)
            off = api.builder.create_add(idx_splat, arange)
            s0_splat = api.builder.create_splat(ir_ty, stride_0_bv.value)
            total_off = api.builder.create_mul(off, s0_splat)
        else:
            return None, None

        # Splat base pointer to tensor-of-pointers (ptr_to_int → splat → int_to_ptr)
        ptr_tensor_type = f"tensor<{tile_size}x!tt.ptr<{elem_type}>>"
        ptr_tensor, _ = self._splat_builder(
            api, ptr_bv.value, ptr_bv.type_str or "!tt.ptr<f32>", ptr_tensor_type
        )

        # addptr: tensor<ptr> + tensor<i32>
        final_ptr = api.builder.create_addptr(ptr_tensor, total_off)

        # Optional mask: arange < size_1
        mask_val = None
        if is_2d and size_1_bv is not None:
            ir_ty = api.get_ir_type(arange_type)
            sz_splat = api.builder.create_splat(ir_ty, size_1_bv.value)
            mask_val = api.builder.create_icmpSLT(arange, sz_splat)

        # Load
        if mask_val is not None:
            other_ir_ty = api.get_ir_type(result_type)
            zero = api.create_float_constant(0.0, elem_type)
            other_splat = api.builder.create_splat(other_ir_ty, zero)
            loaded = api.builder.create_masked_load(
                final_ptr, mask_val, other_splat,
                api._tl_ir.CACHE_MODIFIER.NONE,
                api._tl_ir.EVICTION_POLICY.NORMAL, False,
            )
        else:
            loaded = api.builder.create_load(
                final_ptr,
                api._tl_ir.CACHE_MODIFIER.NONE,
                api._tl_ir.EVICTION_POLICY.NORMAL, False,
            )

        if loaded is None:
            return None, None

        return loaded, result_type

    def _fix_2d_store_ptr(self, stmt, old_ptr, old_ptr_ty, api: _BuilderAPI,
                           val_dims=None):
        """Recompute the store pointer and mask for a 2D value.

        Returns (pointer, mask) where mask is a 2D boundary mask (or None
        if no boundary check is needed).
        """
        # Determine which tensor this store targets from the pointer expression.
        target_idx = None
        ptr_expr = stmt.pointer if hasattr(stmt, "pointer") else None
        if isinstance(ptr_expr, MidBinOp) and ptr_expr.op == "+":
            for child in [ptr_expr.lhs, ptr_expr.rhs]:
                if isinstance(child, MidName) and "pointers" in child.name:
                    m = _TENSOR_BARE_NAME_RE.match(
                        child.name.split("_pointers")[0].rstrip("_")
                    )
                    if m:
                        candidate = int(m.group(1))
                        if candidate in self._tensor_name_to_idx.values():
                            target_idx = candidate
                            break

        if target_idx is None:
            return None, None

        tile_size = self._tile_size or 1024
        idx_0_bv = self._bv(f"ninetoothed_tensor_{target_idx}_index_0")
        idx_1_bv = self._bv(f"ninetoothed_tensor_{target_idx}_index_1")
        stride_0_bv = self._bv(
            f"ninetoothed_ninetoothed_tensor_{target_idx}_stride_0"
        )
        stride_1_bv = self._bv(
            f"ninetoothed_ninetoothed_tensor_{target_idx}_stride_1"
        )
        ptr_bv = self._bv(f"ninetoothed_tensor_{target_idx}_pointers")
        size_0_bv = self._bv(
            f"ninetoothed_ninetoothed_tensor_{target_idx}_size_0"
        )
        size_1_bv = self._bv(
            f"ninetoothed_ninetoothed_tensor_{target_idx}_size_1"
        )

        if any(v is None for v in [ptr_bv, idx_0_bv, idx_1_bv, stride_0_bv, stride_1_bv]):
            return None, None

        elem_type = "f32"
        if ptr_bv.type_str and "!tt.ptr<" in ptr_bv.type_str:
            elem_type = ptr_bv.type_str.split("<")[1].rstrip(">")

        # Use actual value dimensions for the 2D pointer shape.
        tile_m = val_dims[0] if val_dims and len(val_dims) >= 2 else tile_size
        tile_n = val_dims[1] if val_dims and len(val_dims) >= 2 else tile_size

        row_ir_ty = api.get_ir_type(f"tensor<{tile_m}xi32>")
        col_ir_ty = api.get_ir_type(f"tensor<{tile_n}xi32>")
        row_arange = api.create_make_range(0, tile_m)
        col_arange = api.create_make_range(0, tile_n)
        row_block = api.create_int_constant(tile_m, 32)
        col_block = api.create_int_constant(tile_n, 32)

        # Row indices: (row_block * tile_m + arange) expanded to [tile_m, 1]
        row_start = api.builder.create_mul(idx_0_bv.value, row_block)
        row_idx_1d = api.builder.create_add(
            api.builder.create_splat(row_ir_ty, row_start), row_arange
        )
        row_idx_2d = api.builder.create_expand_dims(row_idx_1d, 1)

        # Col indices: (col_block * tile_n + arange) expanded to [1, tile_n]
        col_start = api.builder.create_mul(idx_1_bv.value, col_block)
        col_idx_1d = api.builder.create_add(
            api.builder.create_splat(col_ir_ty, col_start), col_arange
        )
        col_idx_2d = api.builder.create_expand_dims(col_idx_1d, 0)

        # 2D offsets: splat strides to match 2D index shapes for arith.muli
        row_ty = api.get_ir_type(f"tensor<{tile_m}x1xi32>")
        col_ty = api.get_ir_type(f"tensor<1x{tile_n}xi32>")
        s0_splat = api.builder.create_splat(row_ty, stride_0_bv.value)
        s1_splat = api.builder.create_splat(col_ty, stride_1_bv.value)
        row_off = api.builder.create_mul(row_idx_2d, s0_splat)
        col_off = api.builder.create_mul(col_idx_2d, s1_splat)
        # Broadcast both to full tile_m x tile_n before adding
        row_off_bc = api.builder.create_broadcast(row_off, [tile_m, tile_n])
        col_off_bc = api.builder.create_broadcast(col_off, [tile_m, tile_n])
        total_off = api.builder.create_add(row_off_bc, col_off_bc)

        # 2D pointer tensor
        ptr_tensor_type = f"tensor<{tile_m}x{tile_n}x!tt.ptr<{elem_type}>>"
        ptr_tensor, _ = self._splat_builder(
            api, ptr_bv.value, ptr_bv.type_str or f"!tt.ptr<{elem_type}>",
            ptr_tensor_type,
        )
        ptr_val = api.builder.create_addptr(ptr_tensor, total_off)

        # 2D boundary mask: check actual element indices (start + arange) < size
        mask_val = None
        if size_0_bv is not None or size_1_bv is not None:
            row_mask = None
            if size_0_bv is not None:
                s0_sz = api.builder.create_splat(row_ir_ty, size_0_bv.value)
                row_mask_1d = api.builder.create_icmpSLT(row_idx_1d, s0_sz)
                row_mask = api.builder.create_expand_dims(row_mask_1d, 1)

            col_mask = None
            if size_1_bv is not None:
                s1_sz = api.builder.create_splat(col_ir_ty, size_1_bv.value)
                col_mask_1d = api.builder.create_icmpSLT(col_idx_1d, s1_sz)
                col_mask = api.builder.create_expand_dims(col_mask_1d, 0)

            if row_mask is not None and col_mask is not None:
                # Broadcast to full 2D before AND
                row_mask_bc = api.builder.create_broadcast(row_mask, [tile_m, tile_n])
                col_mask_bc = api.builder.create_broadcast(col_mask, [tile_m, tile_n])
                mask_val = api.builder.create_and(row_mask_bc, col_mask_bc)
            elif row_mask is not None:
                mask_val = api.builder.create_broadcast(row_mask, [tile_m, tile_n])
            elif col_mask is not None:
                mask_val = api.builder.create_broadcast(col_mask, [tile_m, tile_n])

        return ptr_val, mask_val

    def _try_store_tensor(self, name: str, val, api: _BuilderAPI) -> None:
        """Emit a store for a tensor param referenced by *name*."""
        idx = self._tensor_name_to_idx.get(name)
        if idx is None:
            return

        tile_size = self._tile_size or 1024

        ptr_bv = self._bv(f"ninetoothed_tensor_{idx}_pointers")
        idx_0_bv = self._bv(f"ninetoothed_tensor_{idx}_index_0")
        idx_1_bv = self._bv(f"ninetoothed_tensor_{idx}_index_1")
        stride_0_bv = self._bv(f"ninetoothed_ninetoothed_tensor_{idx}_stride_0")
        stride_1_bv = self._bv(f"ninetoothed_ninetoothed_tensor_{idx}_stride_1")
        size_0_bv = self._bv(f"ninetoothed_ninetoothed_tensor_{idx}_size_0")
        size_1_bv = self._bv(f"ninetoothed_ninetoothed_tensor_{idx}_size_1")

        if ptr_bv is None or idx_0_bv is None:
            return

        elem_type = "f32"
        if ptr_bv.type_str and "!tt.ptr<" in ptr_bv.type_str:
            elem_type = ptr_bv.type_str.split("<")[1].rstrip(">")

        is_2d = stride_1_bv is not None
        arange = api.create_make_range(0, tile_size)
        arange_type = f"tensor<{tile_size}xi32>"
        ir_ty = api.get_ir_type(arange_type)

        # Check if the output is truly 2D (both tile dims > 1)
        is_truly_2d = is_2d and idx_1_bv is not None

        if is_truly_2d and stride_0_bv is not None and stride_1_bv is not None:
            # --- 2D store: compute 2D offset grid --------------------------
            tile_m = tile_size
            tile_n = tile_size

            # Row indices: (row_block * tile_m + arange(0, tile_m))
            block_m_val = api.create_int_constant(tile_m, 32)
            row_start = api.builder.create_mul(idx_0_bv.value, block_m_val)
            row_idx_1d = api.builder.create_add(
                api.builder.create_splat(ir_ty, row_start), arange
            )  # tensor<tile_m xi32>
            row_idx_2d = api.builder.create_expand_dims(row_idx_1d, 1)
            # tensor<tile_m x 1 xi32>

            # Col indices: (col_block * tile_n + arange(0, tile_n))
            block_n_val = api.create_int_constant(tile_n, 32)
            col_start = api.builder.create_mul(idx_1_bv.value, block_n_val)
            col_idx_1d = api.builder.create_add(
                api.builder.create_splat(ir_ty, col_start), arange
            )  # tensor<tile_n xi32>
            col_idx_2d = api.builder.create_expand_dims(col_idx_1d, 0)
            # tensor<1 x tile_n xi32>

            # Offsets: splat strides to match 2D index shapes for arith.muli
            row_ty = api.get_ir_type(f"tensor<{tile_m}x1xi32>")
            col_ty = api.get_ir_type(f"tensor<1x{tile_n}xi32>")
            s0_splat = api.builder.create_splat(row_ty, stride_0_bv.value)
            s1_splat = api.builder.create_splat(col_ty, stride_1_bv.value)
            row_off_2d = api.builder.create_mul(row_idx_2d, s0_splat)
            # tensor<tile_m x 1 xi32>
            col_off_2d = api.builder.create_mul(col_idx_2d, s1_splat)
            # tensor<1 x tile_n xi32>
            # Broadcast both to full tile_m x tile_n before adding
            row_off_bc = api.builder.create_broadcast(row_off_2d, [tile_m, tile_n])
            col_off_bc = api.builder.create_broadcast(col_off_2d, [tile_m, tile_n])
            total_off = api.builder.create_add(row_off_bc, col_off_bc)
            # tensor<tile_m x tile_n xi32>

            # 2D pointer tensor
            ptr_tensor_type = (
                f"tensor<{tile_m}x{tile_n}x!tt.ptr<{elem_type}>>"
            )
            ptr_tensor, _ = self._splat_builder(
                api,
                ptr_bv.value,
                ptr_bv.type_str or f"!tt.ptr<{elem_type}>",
                ptr_tensor_type,
            )
            final_ptr = api.builder.create_addptr(ptr_tensor, total_off)

            # 2D mask
            mask_val = None
            if size_0_bv is not None or size_1_bv is not None:
                row_arange_2d = api.builder.create_expand_dims(arange, 1)
                col_arange_2d = api.builder.create_expand_dims(arange, 0)

                row_mask = None
                if size_0_bv is not None:
                    s0_sz = api.builder.create_splat(ir_ty, size_0_bv.value)
                    row_mask_1d = api.builder.create_icmpSLT(arange, s0_sz)
                    row_mask = api.builder.create_expand_dims(row_mask_1d, 1)

                col_mask = None
                if size_1_bv is not None:
                    s1_sz = api.builder.create_splat(ir_ty, size_1_bv.value)
                    col_mask_1d = api.builder.create_icmpSLT(arange, s1_sz)
                    col_mask = api.builder.create_expand_dims(col_mask_1d, 0)

                if row_mask is not None and col_mask is not None:
                    row_mask_bc = api.builder.create_broadcast(row_mask, [tile_m, tile_n])
                    col_mask_bc = api.builder.create_broadcast(col_mask, [tile_m, tile_n])
                    mask_val = api.builder.create_and(row_mask_bc, col_mask_bc)
                elif row_mask is not None:
                    mask_val = api.builder.create_broadcast(row_mask, [tile_m, tile_n])
                elif col_mask is not None:
                    mask_val = api.builder.create_broadcast(col_mask, [tile_m, tile_n])

        elif is_2d and stride_0_bv is not None and stride_1_bv is not None:
            # --- Effectively 1D (first dim is 1): 1D offsets ---------------
            row_off = api.builder.create_mul(idx_0_bv.value, stride_0_bv.value)
            row_splat = api.builder.create_splat(ir_ty, row_off)
            s1_splat = api.builder.create_splat(ir_ty, stride_1_bv.value)
            col_off = api.builder.create_mul(arange, s1_splat)
            total_off = api.builder.create_add(row_splat, col_off)

            ptr_tensor_type = f"tensor<{tile_size}x!tt.ptr<{elem_type}>>"
            ptr_tensor, _ = self._splat_builder(
                api,
                ptr_bv.value,
                ptr_bv.type_str or f"!tt.ptr<{elem_type}>",
                ptr_tensor_type,
            )
            final_ptr = api.builder.create_addptr(ptr_tensor, total_off)

            mask_val = None
            if size_1_bv is not None:
                sz_splat = api.builder.create_splat(ir_ty, size_1_bv.value)
                mask_val = api.builder.create_icmpSLT(arange, sz_splat)

        elif stride_0_bv is not None:
            block_val = api.create_int_constant(tile_size, 32)
            idx_t = api.builder.create_mul(idx_0_bv.value, block_val)
            idx_splat = api.builder.create_splat(ir_ty, idx_t)
            off = api.builder.create_add(idx_splat, arange)
            s0_splat = api.builder.create_splat(ir_ty, stride_0_bv.value)
            total_off = api.builder.create_mul(off, s0_splat)

            ptr_tensor_type = f"tensor<{tile_size}x!tt.ptr<{elem_type}>>"
            ptr_tensor, _ = self._splat_builder(
                api,
                ptr_bv.value,
                ptr_bv.type_str or f"!tt.ptr<{elem_type}>",
                ptr_tensor_type,
            )
            final_ptr = api.builder.create_addptr(ptr_tensor, total_off)

            mask_val = None
        else:
            return

        if mask_val is not None:
            api.builder.create_masked_store(
                final_ptr, val, mask_val,
                api._tl_ir.CACHE_MODIFIER.NONE,
                api._tl_ir.EVICTION_POLICY.NORMAL,
            )
        else:
            api.builder.create_store(
                final_ptr, val,
                api._tl_ir.CACHE_MODIFIER.NONE,
                api._tl_ir.EVICTION_POLICY.NORMAL,
            )

    def _pointer_builder(self, expr: MidPointerExpr, api: _BuilderAPI) -> Tuple:
        base_val, base_type = self._visit_expr_builder_typed(expr.base, api)
        offset_val, offset_type = self._visit_expr_builder_typed(expr.offsets, api)
        if base_val is None or offset_val is None:
            return None, None

        base_type = base_type or _ptr_type("f32")
        offset_type = offset_type or _tensor_type([self._tile_size or 1024], "i32")

        # Splat base pointer to tensor-of-pointers
        elem = _elem_of(base_type)
        offset_dims = _shape_of(offset_type) or [self._tile_size or 1024]
        ptr_tensor_ty = _ptr_tensor_type(elem, offset_dims)
        splatted, _ = self._splat_builder(api, base_val, base_type, ptr_tensor_ty)

        result = api.builder.create_addptr(splatted, offset_val)
        return result, ptr_tensor_ty

    def _mask_builder(self, expr: MidMaskExpr, api: _BuilderAPI) -> Tuple:
        result, res_type = self._visit_expr_builder_typed(expr.conditions[0], api)
        if result is None:
            return None, None

        for cond in expr.conditions[1:]:
            result, res_type = self._fold_cond_with_splat(result, res_type, cond, "andi", api)
            if result is None:
                return None, None
        return result, res_type or f"tensor<{self._tile_size or 1024}xi1>"

    # -- arange / subscript builders ---------------------------------------

    def _arange_builder(self, expr: MidArange, api: _BuilderAPI) -> Tuple:
        start = self._resolve_int(expr.start)
        end = self._resolve_int(expr.end)
        size = end - start
        tile = size if size > 0 else (self._tile_size or 1024)

        tensor_ty = api.get_ir_type(_tensor_type([tile], "i32"))
        if tensor_ty is None:
            return None, None
        val = api.builder.create_make_range(tensor_ty, start, end)
        return val, f"tensor<{tile}xi32>"

    def _subscript_builder(self, expr: MidSubscript, api: _BuilderAPI) -> Tuple:
        if self._is_arange_subscript(expr):
            end = self._arange_end(expr)
            shape = self._arange_shape(expr)

            # For tiled kernels (e.g. tile (1, BLOCK_SIZE)), the AST pass
            # generates 2D arange shapes like [1, N] or [1, 1].  Triton
            # processes tiles as flat 1D blocks, so flatten: keep only the
            # non-trivial dimension.  [1,1] → scalar 0, [1,N] → [N], etc.
            non_trivial = [d for d in shape if d > 1]
            if len(non_trivial) == 0:
                # arange(1) = [0] — return scalar constant 0
                return api.create_int_constant(0, 32), "i32"
            if len(non_trivial) == 1:
                end = non_trivial[0]
                shape = [end]

            tensor_ty = api.get_ir_type(_tensor_type([end], "i32"))
            if tensor_ty is None:
                return None, None
            val = api.builder.create_make_range(tensor_ty, 0, end)
            type_str = f"tensor<{end}xi32>"

            if len(shape) == 2:
                new_type = _tensor_type(shape, "i32")
                reshaped = api.builder.create_reshape(val, shape, False)
                return reshaped, new_type
            return val, type_str

        # Handle tensor.shape[dim] patterns:
        #   input.shape[-1]   → MidSubscript(MidTuple(shape_nodes), MidUnaryOp(-, 1))
        #   input.shape[0]    → MidSubscript(MidTuple(shape_nodes), MidConstant(0))
        #
        # When the value is a MidTuple of shape expressions, we can directly
        # resolve the dimension to the corresponding element.
        dim = self._resolve_subscript_dim(expr.slice)
        if dim is not None and isinstance(expr.value, MidTuple):
            elts = expr.value.elts
            # Normalise negative index
            idx = dim
            if idx < 0:
                idx = len(elts) + idx
            if 0 <= idx < len(elts):
                elt = elts[idx]
                if isinstance(elt, MidName):
                    bv = self._bv(elt.name)
                    if bv is not None:
                        return bv.value, "i32"
                if isinstance(elt, MidConstant) and isinstance(elt.value, int):
                    val = api.create_int_constant(elt.value, 32)
                    return val, "i32"

        # Non-arange subscript: visit value and slice
        val, ty = self._visit_expr_builder_typed(expr.value, api)
        if expr.slice is not None:
            _, _ = self._visit_expr_builder_typed(expr.slice, api)
        return val, ty

    @staticmethod
    def _resolve_subscript_dim(slice_node) -> Optional[int]:
        """Resolve a subscript slice to an integer dimension.

        Handles ``MidConstant(n)`` and ``MidUnaryOp('-', MidConstant(n))``.
        """
        if isinstance(slice_node, MidConstant) and isinstance(slice_node.value, int):
            return slice_node.value
        if isinstance(slice_node, MidUnaryOp) and slice_node.op == "-":
            if isinstance(slice_node.operand, MidConstant) and isinstance(slice_node.operand.value, int):
                return -slice_node.operand.value
        return None

    def _try_resolve_tensor_shape_dim(self, tensor_name: str, dim: int,
                                       api: _BuilderAPI):
        """Resolve ``tensor.shape[dim]`` to the corresponding size parameter.

        For a 2D tensor tiled as ``(1, BLOCK_SIZE)``, ``shape[0]`` maps to
        the grid dimension (number of rows = program_id range) and
        ``shape[-1]`` (or ``shape[1]``) maps to the last tile dimension
        (``BLOCK_SIZE`` if it is a constexpr, otherwise the corresponding
        ``ninetoothed_ninetoothed_tensor_{idx}_size_1`` parameter).
        """
        idx = self._tensor_name_to_idx.get(tensor_name)
        if idx is None:
            return None

        # Normalise negative dimension index
        shape_list = self._tensor_shapes.get(tensor_name, [])
        ndim = len(shape_list)
        if ndim == 0:
            ndim = 1  # conservative default
        if dim < 0:
            dim = ndim + dim
        if dim < 0 or dim >= ndim:
            return None

        # Try to find the size parameter for this dimension
        size_key = f"ninetoothed_ninetoothed_tensor_{idx}_size_{dim}"
        bv = self._bv(size_key)
        if bv is not None:
            return bv.value

        # For constexpr dimensions (e.g. BLOCK_SIZE), look up the param
        if dim < ndim:
            dim_str = shape_list[dim]
            bv = self._bv(dim_str)
            if bv is not None:
                return bv.value

        return None

    # -- call builder -----------------------------------------------------

    def _call_builder(self, expr: MidCall, api: _BuilderAPI) -> Tuple:
        func = expr.func
        # Extract base name: "ninetoothed.language.dot" -> "dot"
        base = func.rsplit(".", 1)[-1] if "." in func else func

        # -- range ---------------------------------------------------------
        if base == "range" and len(expr.args) == 1:
            upper = expr.args[0]
            if isinstance(upper, MidSubscript) and isinstance(upper.slice, MidConstant):
                idx = int(upper.slice.value)
                if isinstance(upper.value, MidTuple) and idx < len(upper.value.elts):
                    return self._visit_expr_builder_typed(upper.value.elts[idx], api)
            return self._visit_expr_builder_typed(upper, api)

        # -- zeros ---------------------------------------------------------
        if base == "zeros":
            dtype_str = "f32"
            if "dtype" in expr.kwargs:
                kw = expr.kwargs["dtype"]
                if isinstance(kw, MidName):
                    dtype_str = self._resolve_dtype_name(kw.name)
                elif isinstance(kw, str):
                    dtype_str = self._resolve_dtype_name(kw)
            shape_str = self._resolve_shape_from_expr(expr.args[0]) if expr.args else str(self._tile_size or 1024)
            shape = [int(s) for s in shape_str.split("x")] if "x" in shape_str else [int(shape_str)]
            tensor_type_str = _tensor_type(shape, dtype_str)
            ir_ty = api.get_ir_type(tensor_type_str)
            if ir_ty is None:
                return None, None
            zero_val = api.create_float_constant(0.0, dtype_str)
            if zero_val is None:
                return None, None
            result = api.builder.create_splat(ir_ty, zero_val)
            return result, tensor_type_str

        # -- dot -----------------------------------------------------------
        if base == "dot" and len(expr.args) >= 2:
            a_val, a_type = self._visit_expr_builder_typed(expr.args[0], api)
            b_val, b_type = self._visit_expr_builder_typed(expr.args[1], api)
            if a_val is None or b_val is None:
                return None, None

            # Expand 1D operands to 2D for tt.dot (requires 2D+ operands).
            a_dims = _shape_of(a_type) if a_type else None
            b_dims = _shape_of(b_type) if b_type else None
            if (a_dims is not None and len(a_dims) == 1
                    and b_dims is not None and len(b_dims) == 1):
                k_dim = a_dims[0]
                a_elem = _elem_of(a_type)
                b_elem = _elem_of(b_type)
                a_val = api.builder.create_expand_dims(a_val, 1)
                a_type = _tensor_type([k_dim, 1], a_elem)
                b_val = api.builder.create_expand_dims(b_val, 0)
                b_type = _tensor_type([1, k_dim], b_elem)

            result_type = self._resolve_dot_result_type()
            # Create zero accumulator for dot without explicit accumulator
            zero_acc = api.create_float_constant(0.0, "f32")
            if zero_acc is None:
                return None, None
            acc_ir_ty = api.get_ir_type(result_type)
            if acc_ir_ty is None:
                return None, None
            acc_splat = api.builder.create_splat(acc_ir_ty, zero_acc)
            result = api.builder.create_dot(
                a_val, b_val, acc_splat,
                api._tl_ir.INPUT_PRECISION.TF32, 0,
            )
            return result, result_type

        # -- reduce operations (sum, max, min) -------------------------------
        # These must be handled before the general arg visit below because
        # they require a special insertion-point dance for the combiner region.
        if base in ("sum", "max", "min") and len(expr.args) == 1:
            return self._reduce_call_builder(base, expr, api)

        # -- cast (must be before general arg visit) -----------------------
        # The second positional arg is a dtype reference (e.g. ntl.float32),
        # not a builder value, so it must not be visited as a regular arg.
        if base == "cast" and len(expr.args) >= 1:
            val, val_type = self._visit_expr_builder_typed(expr.args[0], api)
            if val is not None:
                return self._cast_call_builder(val, val_type, expr, api)

        # -- math intrinsics -----------------------------------------------
        typed_args = [self._visit_expr_builder_typed(a, api) for a in expr.args]
        if any(a[0] is None for a in typed_args):
            return None, None
        args = [a[0] for a in typed_args]

        math_map = {
            "exp": (api.create_exp, 1),
            "exp2": (api.create_exp2, 1),
            "log": (api.create_log, 1),
            "log2": (api.create_log2, 1),
            "sqrt": (api.create_sqrt, 1),
            "abs": (api.create_abs, 1),
            "floor": (api.create_floor, 1),
            "ceil": (api.create_ceil, 1),
            "rsqrt": (api.create_rsqrt, 1),
        }
        entry = math_map.get(base)
        if entry and len(args) >= entry[1]:
            fn = entry[0]
            result = fn(args[0])
            arg_type = typed_args[0][1] if typed_args else "f32"
            return result, arg_type

        # -- 2-arg math (maximum / minimum) ---------------------------------
        if base in ("maximum", "max") and len(args) >= 2:
            result = api.create_maximumf(args[0], args[1])
            arg_type = typed_args[0][1] if typed_args else "f32"
            return result, arg_type
        if base in ("minimum", "min") and len(args) >= 2:
            result = api.create_minimumf(args[0], args[1])
            arg_type = typed_args[0][1] if typed_args else "f32"
            return result, arg_type

        if base == "fma" and len(args) >= 3:
            result = api.create_fma(args[0], args[1], args[2])
            return result, "f32"
        if base == "pow" and len(args) >= 2:
            result = api.create_pow(args[0], args[1])
            return result, "f32"

        # -- where (select) -------------------------------------------------
        if base == "where" and len(args) >= 3:
            result = api.create_select(args[0], args[1], args[2])
            true_type = typed_args[1][1] if len(typed_args) > 1 else "f32"
            return result, true_type

        # -- generic / unhandled: visit all args ---------------------------
        for a in expr.args:
            self._visit_expr_builder(a, api)
        return None, None

    def _reduce_call_builder(self, op_name: str, expr: MidCall,
                              api: _BuilderAPI) -> Tuple:
        """Emit a ``tt.reduce`` for *sum*, *max*, or *min*.

        Pattern (e.g. sum)::

            %reduce = tt.reduce %tensor, axis <axis> : tensor<MxNxf32>
            ^bb0(%arg0: f32, %arg1: f32):
              %combined = math.fadd %arg0, %arg1 : f32
              tt.reduce_ret %combined : f32
            %result = ... : tensor<Mxf32>
        """
        # Visit the single argument to get its value and type
        val, val_type = self._visit_expr_builder_typed(expr.args[0], api)
        if val is None or val_type is None:
            return None, None

        # Determine axis: check kwargs for explicit axis, default to -1
        axis = -1
        if "axis" in expr.kwargs:
            axis_kw = expr.kwargs["axis"]
            if isinstance(axis_kw, MidConstant) and isinstance(axis_kw.value, int):
                axis = axis_kw.value
            elif isinstance(axis_kw, int):
                axis = axis_kw

        # Normalise negative axis using the tracked type shape
        if axis < 0 and _is_tensor_type(val_type):
            shape = _shape_of(val_type)
            if shape is not None:
                axis = len(shape) + axis

        # Compute result type: dropping the reduced dimension
        result_type = self._compute_reduce_result_type(val_type, axis)

        # Choose the combiner builder
        if op_name == "sum":
            combiner_fn = api.builder.create_fadd
        elif op_name in ("max", "maximum"):
            combiner_fn = api.builder.create_maximumf
        elif op_name in ("min", "minimum"):
            combiner_fn = api.builder.create_minimumf
        else:
            return None, None

        # Save the current insertion point before entering the combiner region
        ip_before = api.builder.get_insertion_block()

        # Create the reduce operation and save result before building combiner
        reduce_op = api.builder.create_reduce([val], axis)
        result_val = reduce_op.get_result(0)

        # Build the combiner region using create_block_with_parent
        elem_type_str = _elem_of(val_type)
        elem_ir_type = api.get_ir_type(elem_type_str)
        region = reduce_op.get_region(0)
        body = api.builder.create_block_with_parent(region, [elem_ir_type, elem_ir_type])
        api.builder.set_insertion_point_to_end(body)

        # body.arg(0) and body.arg(1) are the two element values to combine
        combined = combiner_fn(body.arg(0), body.arg(1))
        api.builder.create_reduce_ret(combined)

        # Restore insertion point to the main function body
        if ip_before is not None:
            api.builder.set_insertion_point_to_end(ip_before)

        return result_val, result_type

    def _compute_reduce_result_type(self, val_type: str, axis: int) -> str:
        """Compute the result type of a reduce along *axis*.

        For ``axis=-1`` on a 2D tensor ``tensor<MxNxf32>`` the result is
        ``tensor<Mxf32>``.  For a 1D tensor ``tensor<Nxf32>`` the result is
        the scalar element type ``f32``.
        """
        if not _is_tensor_type(val_type):
            return val_type
        shape = _shape_of(val_type)
        if shape is None:
            return val_type
        elem = _elem_of(val_type)
        if len(shape) == 0:
            return elem
        # Normalise negative axis
        if axis < 0:
            axis = len(shape) + axis
        if axis < 0 or axis >= len(shape):
            return val_type
        new_shape = [d for i, d in enumerate(shape) if i != axis]
        if len(new_shape) == 0:
            return elem
        return _tensor_type(new_shape, elem)

    def _cast_call_builder(self, value, val_type: Optional[str], expr: MidCall,
                            api: _BuilderAPI) -> Tuple:
        """Emit a cast operation (fp-to-fp or int-to-int).

        Dispatches to :meth:`_BuilderAPI.create_fp_to_fp` for float types and
        :meth:`_BuilderAPI.create_int_cast` for integer types.

        *value* is the native Triton value and *val_type* is the type string
        of that value (as tracked by the value table).
        """
        # Determine target dtype from kwargs or second positional arg
        target_dtype = None
        dtype_source = expr.kwargs.get("dtype")
        if dtype_source is None and len(expr.args) >= 2:
            dtype_source = expr.args[1]
        if dtype_source is not None:
            if isinstance(dtype_source, MidName):
                target_dtype = self._resolve_dtype_name(dtype_source.name)
            elif isinstance(dtype_source, str):
                target_dtype = self._resolve_dtype_name(dtype_source)
            elif isinstance(dtype_source, MidConstant) and isinstance(dtype_source.value, str):
                target_dtype = self._resolve_dtype_name(dtype_source.value)

        if target_dtype is None:
            return None, None

        mlir_type = _mlir_elem_type(target_dtype)
        source_type = val_type

        if mlir_type in _FLOAT_TYPES and (source_type is None or _is_float_type(source_type)):
            # Skip identity casts (same source and target element type)
            source_elem = _elem_of(source_type) if source_type else None
            if source_elem == mlir_type:
                return value, source_type or mlir_type
            if source_type and _is_tensor_type(source_type):
                cast_type = _replace_tensor_elem(source_type, mlir_type)
            else:
                cast_type = mlir_type
            result = api.create_fp_to_fp(value, cast_type)
            if source_type and _is_tensor_type(source_type):
                result_type = _replace_tensor_elem(source_type, mlir_type)
            else:
                result_type = mlir_type
            return result, result_type

        if mlir_type in _INT_TYPES:
            is_signed = True
            if source_type and _is_tensor_type(source_type):
                cast_type = _replace_tensor_elem(source_type, mlir_type)
            else:
                cast_type = mlir_type
            result = api.create_int_cast(value, cast_type, is_signed)
            if source_type and _is_tensor_type(source_type):
                result_type = _replace_tensor_elem(source_type, mlir_type)
            else:
                result_type = mlir_type
            return result, result_type

        return None, None

    # -- type resolution for builder path ----------------------------------

    def _resolve_load_type(self, expr) -> str:
        """Resolve the result type of a load expression."""
        ptr_base = self._extract_pointer_base(expr)
        if ptr_base is not None:
            ptr_name = ptr_base.name
            tensor_name = self._resolve_pointer_to_tensor_name(ptr_name)
            if tensor_name and tensor_name in self._tensor_shapes:
                shape = self._tensor_shapes[tensor_name]
                base_type = self._param_types.get(ptr_name)
                elem = _elem_of(base_type) if base_type else "f32"
                return _tensor_type(shape, elem)
        tile = self._tile_size or 1024
        return _tensor_type([tile], "f32")

    # -- type / pointer resolution helpers (shared) -------------------

    def _resolve_dtype_name(self, dtype_str: str) -> str:
        """Resolve a dtype name (e.g. 'float32', 'ntl.float32') to MLIR element type."""
        # Strip common dialect prefixes
        for prefix in ("ninetoothed.language.", "triton.language.", "ntl."):
            if dtype_str.startswith(prefix):
                dtype_str = dtype_str[len(prefix):]
                break
        return _mlir_elem_type(dtype_str)

    def _resolve_shape_from_expr(self, expr) -> str:
        """Resolve a shape expression to a dimension string like '64' or '64x64'."""
        if isinstance(expr, MidConstant) and isinstance(expr.value, int):
            return str(expr.value)
        if isinstance(expr, MidTuple):
            # Shape tuple like (64, 64) -> "64x64"
            dims = []
            for elt in expr.elts:
                if isinstance(elt, MidConstant) and isinstance(elt.value, int):
                    dims.append(str(elt.value))
                else:
                    # Non-constant element (e.g. Symbol from block_size()):
                    # use tile size as fallback per dimension.
                    dims.append(str(self._tile_size or 1024))
            return "x".join(dims)
        if isinstance(expr, MidName):
            # Try tensor shapes from _tensor_shapes
            if expr.name in self._tensor_shapes:
                dims = self._tensor_shapes[expr.name]
                return "x".join(dims)
        tile = self._tile_size or 1024
        return str(tile)

    def _resolve_dot_result_type(self) -> str:
        """Determine the result type of a dot (matmul) operation.

        Defaults to a 2D tensor of f32 with dimensions based on tensor shapes.
        """
        # Use the largest tensor shape dimensions for the result
        shapes = list(self._tensor_shapes.values())
        if len(shapes) >= 2:
            m = shapes[0][0] if shapes[0] else str(self._tile_size or 64)
            n = shapes[1][1] if len(shapes[1]) > 1 else shapes[1][0] if shapes[1] else str(self._tile_size or 64)
            return _tensor_type([int(m), int(n)], "f32")
        tile = self._tile_size or 64
        return _tensor_type([tile, tile], "f32")

    def _is_arange_subscript(self, expr: MidSubscript) -> bool:
        """Check if a MidSubscript represents an arange pattern."""
        def _all_none(node):
            if isinstance(node, MidConstant) and node.value is None:
                return True
            if isinstance(node, MidTuple):
                return all(_all_none(e) for e in node.elts)
            return False

        if not isinstance(expr.slice, MidTuple):
            return False
        if not all(_all_none(e) for e in expr.slice.elts):
            return False
        return isinstance(expr.value, (MidName, MidConstant))

    def _arange_end(self, expr: MidSubscript) -> int:
        """Return the end value for an arange subscript."""
        if isinstance(expr.value, MidConstant) and isinstance(expr.value.value, int):
            return expr.value.value
        return self._tile_size or 1024

    def _arange_shape(self, expr: MidSubscript) -> List[int]:
        """Determine the shape of a multi-dimensional arange subscript."""
        end = self._arange_end(expr)
        if not isinstance(expr.slice, MidTuple) or len(expr.slice.elts) != 2:
            return [end]

        def _is_slice_all_none(node):
            if isinstance(node, MidTuple):
                return all(isinstance(e, MidConstant) and e.value is None for e in node.elts)
            return False

        elt0, elt1 = expr.slice.elts
        if _is_slice_all_none(elt0) and isinstance(elt1, MidConstant) and elt1.value is None:
            return [end, 1]
        if isinstance(elt0, MidConstant) and elt0.value is None and _is_slice_all_none(elt1):
            return [1, end]
        return [end]

    def _resolve_pointer_to_tensor_name(self, ptr_name: str) -> Optional[str]:
        """Map pointer invariant name to tensor_param name."""
        if self._mid_func is None:
            return None
        m = _TENSOR_PTR_INDEX_RE.match(ptr_name)
        if m:
            idx = int(m.group(1))
            for tp in self._mid_func.tensor_params:
                if tp.tensor is not None and tp.tensor.index == idx:
                    return tp.name
        return None

    def _extract_pointer_base(self, expr) -> Optional[MidName]:
        """Extract the base pointer name from a pointer expression."""
        if isinstance(expr, MidPointerExpr):
            return expr.base if isinstance(expr.base, MidName) else None
        if isinstance(expr, MidSubscript):
            return self._extract_pointer_base(expr.value)
        if isinstance(expr, MidBinOp) and expr.op == "+":
            lhs_result = self._extract_pointer_base(expr.lhs)
            if lhs_result is not None:
                return lhs_result
            return self._extract_pointer_base(expr.rhs)
        return None

    def _extract_type(line: str) -> str:
        """Extract the result type from an MLIR assignment line."""
        stripped = line.strip()
        if "=" not in stripped:
            return ""
        after_eq = stripped.split("=", 1)[1].strip()
        if ":" in after_eq:
            after_colon = after_eq.split(":", 1)[1].strip()
            if " -> " in after_colon:
                return after_colon.rsplit(" -> ", 1)[-1].strip()
            return after_colon
        return ""

    def _resolve_type(self, expr, lines: List[str]) -> Optional[str]:
        """Resolve the type of an expression.

        Used by the builder path to determine operand/result types.
        """
        if isinstance(expr, MidName):
            bv = self._values.get(expr.name)
            if bv is not None:
                return bv.type_str
            ty = self._param_types.get(expr.name)
            if ty is not None:
                return ty
            return None
        if isinstance(expr, MidConstant):
            if isinstance(expr.value, bool):
                return "i1"
            if isinstance(expr.value, int):
                return "i32"
            if isinstance(expr.value, float):
                return "f32"
            return None
        if isinstance(expr, MidProgramId):
            return "i32"
        if isinstance(expr, MidArange):
            start = self._resolve_int_const(expr.start)
            end = self._resolve_int_const(expr.end)
            size = end - start
            tile = size if size > 0 else (self._tile_size or 1024)
            return f"tensor<{tile}xi32>"
        if isinstance(expr, MidPointerExpr):
            tile = self._tile_size or 1024
            if isinstance(expr.base, MidName):
                base_type = self._param_types.get(expr.base.name)
                if base_type and base_type.startswith("!tt.ptr<"):
                    elem = _elem_of(base_type)
                    return _ptr_tensor_type(elem, [tile])
                return base_type
            return _ptr_tensor_type("f32", [tile])
        if isinstance(expr, MidBinOp) and expr.op == "+":
            tile = self._tile_size or 1024
            lhs_type = self._resolve_type(expr.lhs, [])
            rhs_type = self._resolve_type(expr.rhs, [])
            if lhs_type is not None and lhs_type.startswith("!tt.ptr<"):
                elem = _elem_of(lhs_type)
                return _ptr_tensor_type(elem, [tile])
            if rhs_type is not None and rhs_type.startswith("!tt.ptr<"):
                elem = _elem_of(rhs_type)
                return _ptr_tensor_type(elem, [tile])
        if isinstance(expr, MidLoad):
            ptr_base = self._extract_pointer_base(expr.pointer)
            if ptr_base is not None:
                ptr_name = ptr_base.name
                tensor_name = self._resolve_pointer_to_tensor_name(ptr_name)
                if tensor_name and tensor_name in self._tensor_shapes:
                    shape = self._tensor_shapes[tensor_name]
                    base_type = self._param_types.get(ptr_name)
                    elem = _elem_of(base_type) if base_type else "f32"
                    return _tensor_type(shape, elem)
                base_type = self._param_types.get(ptr_name)
                if base_type:
                    elem = _elem_of(base_type)
                    tile = self._tile_size or 1024
                    return _tensor_type([tile], elem)
            tile = self._tile_size or 1024
            return _tensor_type([tile], "f32")
        if isinstance(expr, MidMaskExpr):
            tile = self._tile_size or 1024
            return f"tensor<{tile}xi1>"
        if isinstance(expr, MidSubscript):
            if self._is_arange_subscript(expr):
                shape = self._arange_shape(expr)
                return _tensor_type(shape, "i32")
        if isinstance(expr, MidCall):
            base = expr.func.name if isinstance(expr.func, MidName) else str(expr.func)
            if base == "load" and expr.args:
                arg = expr.args[0]
                if isinstance(arg, MidBinOp) and arg.op == "+":
                    ptr = arg.lhs if isinstance(arg.lhs, MidPointerExpr) else None
                    if ptr is None and isinstance(arg.rhs, MidPointerExpr):
                        ptr = arg.rhs
                    if ptr is not None and isinstance(ptr.base, MidName):
                        ptr_name = ptr.base.name
                        tensor_name = self._resolve_pointer_to_tensor_name(ptr_name)
                        if tensor_name and tensor_name in self._tensor_shapes:
                            shape = self._tensor_shapes[tensor_name]
                            param_type = self._param_types.get(ptr_name)
                            elem = _elem_of(param_type) if param_type else "f32"
                            return _tensor_type(shape, elem)
        if isinstance(expr, MidCompare):
            left_type = self._resolve_type(expr.left, lines)
            if _is_tensor_type(left_type):
                return _replace_tensor_elem(left_type, "i1")
            return "i1"

        return None

    def _resolve_int_const(self, node) -> int:
        """Resolve a node to a concrete integer for arange/make_range."""
        if isinstance(node, MidConstant):
            return int(node.value)
        if isinstance(node, MidName):
            bv = self._values.get(node.name)
            if bv is not None and bv.type_str in _INT_TYPES:
                # bv.value is a native Triton C++ object, not a string.
                # We cannot extract the integer from it directly.
                pass
            return 0
        if isinstance(node, int):
            return node
        return 0


    # -- type / pointer resolution helpers (shared) -----------


