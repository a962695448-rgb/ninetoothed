"""Mid IR node definitions.

This module defines the complete Mid IR node type hierarchy used as an
intermediate representation between Python AST and Triton MLIR dialects.
"""

from ninetoothed.ir.base import IRNode, NativeFormatter


class MidFunction(IRNode):
    """Top-level kernel function in Mid IR.

    Attributes:
        name: Function name.
        params: Flat parameters for function signature (pointers, sizes, strides, constexprs).
        tensor_params: Original tensor metadata from arrangement (with tile history).
        invariants: Pre-computed expressions (program_id, offsets, masks).
        body: Kernel body statements.
        grid_expr: Grid expression for kernel launch (optional).
    """

    def __init__(self, name, params=None, tensor_params=None, invariants=None, body=None, grid_expr=None):
        self.name = name
        self.params = params or []
        self.tensor_params = tensor_params or []
        self.invariants = invariants or []
        self.body = body or []
        self.grid_expr = grid_expr

    def dump(self):
        """Native Mid IR dump -- shows tile semantics, no dialect prefixes."""
        formatter = NativeFormatter()
        return formatter.format(self)


class MidParam(IRNode):
    """Function parameter in Mid IR.

    Attributes:
        name: Parameter name.
        dtype: Element type string (e.g., "float32", "int32").
        ndim: Number of dimensions (None for scalar/constexpr).
        shape: Shape symbols (for tensor params).
        is_constexpr: Whether this is a compile-time constant.
        tensor: Reference to original Tensor object (for tensor_params).
        tile_history: List of TileOp descriptors from Tensor._history.
    """

    def __init__(self, name, dtype=None, ndim=None, shape=None, is_constexpr=False, tensor=None, tile_history=None):
        self.name = name
        self.dtype = dtype
        self.ndim = ndim
        self.shape = shape or ()
        self.is_constexpr = is_constexpr
        self.tensor = tensor
        self.tile_history = tile_history or []


class TileOp(IRNode):
    """Descriptor for a tile operation extracted from Tensor._history."""

    def __init__(self, kind, args, kwargs=None):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs or {}

    def __str__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.kind}({args_str})"


# --- Statements ---

class MidAssign(IRNode):
    """Assignment statement: target = value."""

    def __init__(self, target, value):
        self.target = target
        self.value = value


class MidExprStmt(IRNode):
    """Expression statement (expression with side effects)."""

    def __init__(self, value):
        self.value = value


class MidReturn(IRNode):
    """Return statement."""

    def __init__(self, value=None):
        self.value = value


class MidStore(IRNode):
    """Store statement: store(pointer, value, mask)."""

    def __init__(self, pointer, value, mask=None):
        self.pointer = pointer
        self.value = value
        self.mask = mask


class MidFor(IRNode):
    """For loop statement: for target in range(iter): body."""

    def __init__(self, target, iter_expr, body):
        self.target = target
        self.iter_expr = iter_expr
        self.body = body


class MidIf(IRNode):
    """If statement: if test: body [else: orelse]."""

    def __init__(self, test, body, orelse=None):
        self.test = test
        self.body = body
        self.orelse = orelse or []


class MidInvariant(IRNode):
    """Symbolic invariant (shape/stride/pointer computation)."""

    def __init__(self, target, value):
        self.target = target
        self.value = value


# --- Expressions ---

class MidBinOp(IRNode):
    """Binary operation: lhs op rhs."""

    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs


class MidUnaryOp(IRNode):
    """Unary operation: op operand."""

    def __init__(self, op, operand):
        self.op = op
        self.operand = operand


class MidCompare(IRNode):
    """Comparison: left op right."""

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right


class MidBoolOp(IRNode):
    """Boolean operation: op values."""

    def __init__(self, op, values):
        self.op = op
        self.values = values


class MidIfExp(IRNode):
    """Ternary expression: body if test else orelse."""

    def __init__(self, test, body, orelse):
        self.test = test
        self.body = body
        self.orelse = orelse


class MidCall(IRNode):
    """Function call: func(*args, **kwargs)."""

    def __init__(self, func, args=None, kwargs=None):
        self.func = func
        self.args = args or []
        self.kwargs = kwargs or {}


class MidName(IRNode):
    """Variable reference."""

    def __init__(self, name):
        self.name = name


class MidConstant(IRNode):
    """Literal value (int, float, bool, str)."""

    def __init__(self, value):
        self.value = value


class MidLoad(IRNode):
    """Load expression: load(pointer, mask, other)."""

    def __init__(self, pointer, mask=None, other=None, shape=None):
        self.pointer = pointer
        self.mask = mask
        self.other = other
        self.shape = shape


class MidTuple(IRNode):
    """Tuple expression: (elt0, elt1, ...)."""

    def __init__(self, elts):
        self.elts = elts


class MidSubscript(IRNode):
    """Subscript expression: value[slice]."""

    def __init__(self, value, slice):
        self.value = value
        self.slice = slice


class MidPointerExpr(IRNode):
    """Pointer expression: base + offsets."""

    def __init__(self, base, offsets):
        self.base = base
        self.offsets = offsets


class MidMaskExpr(IRNode):
    """Mask expression: condition[0] & condition[1] & ..."""

    def __init__(self, conditions):
        self.conditions = conditions


class MidProgramId(IRNode):
    """Program ID expression: program_id(axis)."""

    def __init__(self, axis=0):
        self.axis = axis


class MidArange(IRNode):
    """Arange expression: arange(start, end)."""

    def __init__(self, start, end):
        self.start = start
        self.end = end


class MidTile(IRNode):
    """Tile operation metadata."""

    def __init__(self, kind, args):
        self.kind = kind
        self.args = args


class MidTensorAccess(IRNode):
    """Tensor access: tensor[param_name]."""

    def __init__(self, param_name):
        self.param_name = param_name


class MidDataPtr(IRNode):
    """Data pointer: data_ptr(param_name)."""

    def __init__(self, param_name):
        self.param_name = param_name


class MidOffsets(IRNode):
    """Offsets computation."""

    def __init__(self, param_name):
        self.param_name = param_name


class MidStride(IRNode):
    """Stride computation."""

    def __init__(self, param_name, dim):
        self.param_name = param_name
        self.dim = dim


class MidDtypeAttr(IRNode):
    """Dtype attribute."""

    def __init__(self, param_name):
        self.param_name = param_name
