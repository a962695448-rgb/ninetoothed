"""AST to Mid IR transformation pass.

This module implements :class:`ASTToMidIRPass`, which transforms a Python
function (annotated with Tensor objects from ``arrangement``) into a
:class:`MidFunction` tree.  The pass reuses the inlining machinery from
``ninetoothed.generation._Inliner`` and adapts the pointer / mask /
offset computation logic from ``CodeGenerator`` so that the output is
pure Mid IR rather than raw Python AST.
"""

import ast
import copy
import functools
import inspect
import math
import textwrap


import ninetoothed.naming as naming
from ninetoothed.generation import _Inliner
from ninetoothed.ir.mid_ir import (
    MidArange,
    MidAssign,
    MidBinOp,
    MidBoolOp,
    MidCall,
    MidCompare,
    MidConstant,
    MidExprStmt,
    MidFor,
    MidFunction,
    MidIf,
    MidIfExp,
    MidInvariant,
    MidLoad,
    MidMaskExpr,
    MidName,
    MidParam,
    MidPointerExpr,
    MidProgramId,
    MidReturn,
    MidStore,
    MidSubscript,
    MidTile,
    MidTuple,
    MidUnaryOp,
    TileOp,
)
from ninetoothed.language import LANGUAGE
from ninetoothed.symbol import Symbol
from ninetoothed.tensor import Tensor


# ---------------------------------------------------------------------------
# AST operator -> Mid IR operator string
# ---------------------------------------------------------------------------

_AST_OP_MAP = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.MatMult: "@",
}

_AST_UNARYOP_MAP = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "not",
    ast.Invert: "~",
}

_AST_CMPOP_MAP = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.In: "in",
    ast.NotIn: "not in",
}


# ---------------------------------------------------------------------------
# AST slice conversion helper
# ---------------------------------------------------------------------------

def _ast_slice_to_mid(node):
    """Convert an ``ast`` slice node to a Mid IR representation."""
    if isinstance(node, ast.Slice):
        return MidTuple([
            MidConstant(None) if node.lower is None else _ast_expr_to_mid(node.lower),
            MidConstant(None) if node.upper is None else _ast_expr_to_mid(node.upper),
            MidConstant(None) if node.step is None else _ast_expr_to_mid(node.step),
        ])
    return _ast_expr_to_mid(node)


# ---------------------------------------------------------------------------
# Generic AST expression -> Mid IR (used for non-tensor sub-expressions)
# ---------------------------------------------------------------------------

def _resolve_binop_str(op):
    """Map an ``ast`` binary operator to its Mid IR string."""
    op_str = _AST_OP_MAP.get(type(op))
    if op_str is None:
        raise NotImplementedError(f"BinOp {type(op).__name__}")
    return op_str


def _resolve_unaryop_str(op):
    """Map an ``ast`` unary operator to its Mid IR string."""
    op_str = _AST_UNARYOP_MAP.get(type(op))
    if op_str is None:
        raise NotImplementedError(f"UnaryOp {type(op).__name__}")
    return op_str


def _resolve_cmpop_str(op):
    """Map an ``ast`` comparison operator to its Mid IR string."""
    op_str = _AST_CMPOP_MAP.get(type(op))
    if op_str is None:
        raise NotImplementedError(f"Compare op {type(op).__name__}")
    return op_str


def _build_compare_chain(left, ops, comparators, visit_fn):
    """Build a chain of MidCompare / MidBoolOp for multi-comparators.

    Parameters
    ----------
    left : Mid IR node
        The already-converted left-hand side of the first comparison.
    ops : list[ast.cmpop]
        The comparison operators.
    comparators : list[ast.expr]
        The right-hand side operands (as AST nodes).
    visit_fn : callable
        Function to convert AST nodes to Mid IR (either
        ``_ast_expr_to_mid`` or ``ASTToMidIRPass._visit_expr``).
    """
    parts = []
    for op, comparator in zip(ops, comparators):
        op_str = _resolve_cmpop_str(op)
        right = visit_fn(comparator)
        parts.append(MidCompare(op_str, left, right))
        left = right
    if len(parts) == 1:
        return parts[0]
    return MidBoolOp("and", parts)

def _ast_expr_to_mid(node):
    """Recursively convert an ``ast`` expression node into a Mid IR node.

    This is the generic converter used for arithmetic expressions,
    function calls, literals, etc.  It does **not** handle tensor
    loads/stores -- those are dealt with by the visitor methods on
    :class:`ASTToMidIRPass`.
    """
    if isinstance(node, ast.Constant):
        return MidConstant(node.value)

    if isinstance(node, ast.Name):
        return MidName(node.id)

    # --- UnaryOp ---
    if isinstance(node, ast.UnaryOp):
        return MidUnaryOp(_resolve_unaryop_str(node.op), _ast_expr_to_mid(node.operand))

    # --- BinOp ---
    if isinstance(node, ast.BinOp):
        return MidBinOp(_resolve_binop_str(node.op), _ast_expr_to_mid(node.left), _ast_expr_to_mid(node.right))

    # --- Compare ---
    if isinstance(node, ast.Compare):
        return _build_compare_chain(_ast_expr_to_mid(node.left), node.ops, node.comparators, _ast_expr_to_mid)

    # --- BoolOp ---
    if isinstance(node, ast.BoolOp):
        op_str = "and" if isinstance(node.op, ast.And) else "or"
        return MidBoolOp(op_str, [_ast_expr_to_mid(v) for v in node.values])

    # --- IfExp ---
    if isinstance(node, ast.IfExp):
        return MidIfExp(
            _ast_expr_to_mid(node.test),
            _ast_expr_to_mid(node.body),
            _ast_expr_to_mid(node.orelse),
        )

    # --- Call ---
    if isinstance(node, ast.Call):
        return _ast_call_to_mid(node)

    # --- Subscript ---
    if isinstance(node, ast.Subscript):
        return MidSubscript(
            _ast_expr_to_mid(node.value),
            _ast_slice_to_mid(node.slice),
        )

    # --- Tuple ---
    if isinstance(node, ast.Tuple):
        return MidTuple([_ast_expr_to_mid(e) for e in node.elts])

    # --- Attribute ---
    if isinstance(node, ast.Attribute):
        value = _ast_expr_to_mid(node.value)
        # Represent as a dotted-name string for now.
        value_str = _mid_to_str(value)
        return MidName(f"{value_str}.{node.attr}")

    # --- Starred ---
    if isinstance(node, ast.Starred):
        return _ast_expr_to_mid(node.value)

    # --- Slice (when encountered directly, e.g. inside Symbol AST) ---
    if isinstance(node, ast.Slice):
        return _ast_slice_to_mid(node)

    # Fallback
    raise NotImplementedError(f"AST node type {type(node).__name__} cannot be converted to Mid IR yet")


def _mid_to_str(node):
    """Best-effort conversion of a Mid IR node back to a short string (for
    attribute representation)."""
    if isinstance(node, MidName):
        return node.name
    if isinstance(node, MidConstant):
        return str(node.value)
    return "?"


def _ast_call_to_mid(node):
    """Convert an ``ast.Call`` to :class:`MidCall`.

    Special cases for ``ninetoothed.language.*`` calls are handled so that
    the function name preserves the dialect prefix rather than being
    converted to ``triton.language.*``.
    """
    func_name = _resolve_call_name(node.func)

    # Detect program_id call -> MidProgramId
    base = func_name.rsplit(".", 1)[-1] if "." in func_name else func_name
    if base == "program_id":
        axis = 0
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
            axis = node.args[0].value
        return MidProgramId(axis=axis)

    args = [_ast_expr_to_mid(a) for a in node.args]
    kwargs = {}
    for kw in node.keywords:
        kwargs[kw.arg] = _ast_expr_to_mid(kw.value)

    return MidCall(func_name, args, kwargs)


def _resolve_call_name(func_node):
    """Resolve an ``ast`` function node to a dotted-name string."""
    if isinstance(func_node, ast.Name):
        return func_node.id

    if isinstance(func_node, ast.Attribute):
        value_str = _resolve_call_name(func_node.value)
        return f"{value_str}.{func_node.attr}"

    return "<unknown>"


# ---------------------------------------------------------------------------
# Symbol (backed by AST) -> Mid IR
# ---------------------------------------------------------------------------

def _symbol_to_mid(symbol):
    """Convert a :class:`Symbol` (whose ``._node`` is an ``ast`` node) into
    the equivalent Mid IR tree."""
    return _ast_expr_to_mid(symbol.node)


# ---------------------------------------------------------------------------
# The main pass
# ---------------------------------------------------------------------------

class ASTToMidIRPass:
    """Transforms a Python function annotated with :class:`Tensor` objects
    into a :class:`MidFunction` tree.

    The public entry point is :meth:`transform`.
    """

    _NAME_FOR_PID = Symbol("ninetoothed_pid")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, func):
        """Return a :class:`MidFunction` representing *func*.

        Parameters
        ----------
        func : callable
            A function whose ``__annotations__`` have been populated by
            ``arrangement`` (i.e. each value is a :class:`Tensor`).
        """
        # 1. Parse and inline ------------------------------------------------
        func_def = ast.parse(textwrap.dedent(inspect.getsource(func)))
        inliner = _Inliner(func.__globals__)
        inliner.visit(func_def)

        # 2. Collect tensor context from annotations --------------------------
        self._context = inspect.get_annotations(func)
        self._args = list(self._context.values())

        # 3. Walk the function body -------------------------------------------
        self._invariants = {}
        self._func_name = func.__name__

        func_def_node = func_def.body[0]  # the FunctionDef

        body_stmts = []
        for stmt in func_def_node.body:
            converted = self._visit_stmt(stmt)
            if isinstance(converted, list):
                body_stmts.extend(converted)
            else:
                body_stmts.append(converted)

        # 4. Build MidParam list (flat kernel parameters) ---------------------
        params = self._build_params()

        # 5. Build MidParam list for tensor_params ----------------------------
        tensor_params = self._build_tensor_params()

        # 6. Collect invariant Mid IR nodes -----------------------------------
        invariant_nodes = self._build_invariants()

        # 7. Grid expression --------------------------------------------------
        grid_expr = self._build_grid_expr()

        return MidFunction(
            name=self._func_name,
            params=params,
            tensor_params=tensor_params,
            invariants=invariant_nodes,
            body=body_stmts,
            grid_expr=grid_expr,
        )

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _in_context(self, node):
        """Return *True* if *node* is an ``ast.Name`` whose ``id`` maps to a
        :class:`Tensor` in ``_context``."""
        return isinstance(node, ast.Name) and node.id in self._context

    # ------------------------------------------------------------------
    # Parameter building
    # ------------------------------------------------------------------

    def _build_params(self):
        """Build the flat parameter list (pointers, sizes, strides, constexpr
        symbols) that the kernel function accepts."""
        symbols = {}
        for arg in self._args:
            for name in arg.names():
                if isinstance(name, Symbol) and isinstance(name.node, ast.Name):
                    nid = name.node.id
                    if nid != "ninetoothed":
                        symbols[nid] = name

        names = list(symbols.keys())
        meta_names = sorted(n for n in names if naming.is_meta(n))
        non_meta_names = sorted(n for n in names if n not in meta_names)

        # Collect next_power_of_2 names from innermost shapes.
        # The shapes contain original constexpr names (e.g. BLOCK_SIZE);
        # _generate_innermost_indices wraps them with make_next_power_of_2
        # at code-gen time, so we must derive the np2 variants here.
        np2_names = set()
        for arg in self._args:
            if arg.ndim == 0:
                continue
            for size in arg.innermost().shape:
                if isinstance(size, Symbol):
                    for n in size.names():
                        nid = n.node.id
                        if naming.is_next_power_of_2(nid):
                            np2_names.add(nid)
                        elif naming.is_constexpr(nid) and not naming.is_meta(nid):
                            np2_names.add(naming.make_next_power_of_2(nid))
        non_meta_names = sorted(set(non_meta_names) | np2_names)

        params = []
        for name in non_meta_names:
            is_const = naming.is_constexpr(name)
            sym = symbols.get(name)
            dtype = None
            if sym is not None and hasattr(sym, "lower_bound"):
                dtype = "i32"
            params.append(MidParam(name=name, is_constexpr=is_const, dtype=dtype))

        for name in meta_names:
            sym = symbols[name]
            dtype = "i32"
            params.append(MidParam(name=name, is_constexpr=True, dtype=dtype))

        return params

    def _build_tensor_params(self):
        """Build the :class:`MidParam` list that carries tensor metadata
        (shape, dtype, tile history)."""
        tensor_params = []
        for param_name, tensor in self._context.items():
            tile_history = []
            for func, args, kwargs in tensor._history:
                kind = func.__name__
                tile_history.append(TileOp(kind, list(args), kwargs))

            shape = tuple(str(s) for s in tensor.shape)
            dtype_str = str(tensor.dtype) if tensor.dtype is not None else None
            ndim = tensor.ndim if tensor.ndim > 0 else None

            tensor_params.append(
                MidParam(
                    name=param_name,
                    dtype=dtype_str,
                    ndim=ndim,
                    shape=shape,
                    is_constexpr=bool(getattr(tensor, "constexpr", False)),
                    tensor=tensor,
                    tile_history=tile_history,
                )
            )
        return tensor_params

    def _build_invariants(self):
        """Convert ``_invariants`` dict (Symbol -> Symbol) to a list of
        :class:`MidInvariant` nodes."""
        nodes = []
        for target_sym, value_sym in self._invariants.items():
            target_str = str(target_sym)
            value_mid = _symbol_to_mid(value_sym)
            nodes.append(MidInvariant(target=target_str, value=value_mid))
        return nodes

    def _build_grid_expr(self):
        """Build a Mid IR grid expression."""
        if not self._args:
            return None
        num_elements = _ast_expr_to_mid(
            functools.reduce(
                lambda x, y: ast.BinOp(left=x, op=ast.Mult(), right=y),
                [Symbol(s).node for s in self._args[0].shape],
            )
        )
        return num_elements


    # ------------------------------------------------------------------
    # Statement visitors
    # ------------------------------------------------------------------

    def _visit_stmt(self, node):
        """Dispatch a single AST statement node to the appropriate visitor."""
        if isinstance(node, ast.Assign):
            return self._visit_Assign(node)
        if isinstance(node, ast.Return):
            return self._visit_Return(node)
        if isinstance(node, ast.For):
            return self._visit_For(node)
        if isinstance(node, ast.If):
            return self._visit_If(node)
        if isinstance(node, ast.Expr):
            return self._visit_Expr(node)
        if isinstance(node, ast.AugAssign):
            return self._visit_AugAssign(node)
        raise NotImplementedError(
            f"Statement type {type(node).__name__} is not yet supported"
        )

    def _visit_Assign(self, node):
        if len(node.targets) == 1:
            target = node.targets[0]

            # Assignment to a tensor name directly (e.g., `output[...] = ...`)
            if self._in_context(target):
                tensor = self._context[target.id]
                value_mid = self._visit_expr(node.value)
                store = self._generate_store(tensor, value_mid)
                return store

            # Assignment to a subscript of a tensor (e.g., `result[i] = ...`)
            if isinstance(target, ast.Subscript) and isinstance(target.ctx, ast.Store):
                if self._in_context(target.value):
                    tensor = self._context[target.value.id]
                    value_mid = self._visit_expr(node.value)
                    indices = (
                        target.slice.elts
                        if isinstance(target.slice, ast.Tuple)
                        else (target.slice,)
                    )
                    store = self._generate_store(tensor, value_mid, indices=indices)
                    return store

            # Regular assignment
            target_str = target.id if isinstance(target, ast.Name) else _mid_to_str(_ast_expr_to_mid(target))
            value_mid = self._visit_expr(node.value)
            return MidAssign(target=target_str, value=value_mid)

        # Multiple targets (a = b = expr) -- fall back
        target_str = node.targets[0].id if isinstance(node.targets[0], ast.Name) else _mid_to_str(_ast_expr_to_mid(node.targets[0]))
        value_mid = self._visit_expr(node.value)
        return MidAssign(target=target_str, value=value_mid)

    def _visit_Return(self, node):
        if node.value is None:
            return MidReturn()
        value_mid = self._visit_expr(node.value)
        return MidReturn(value=value_mid)

    def _visit_For(self, node):
        target_mid = _ast_expr_to_mid(node.target)
        iter_mid = self._visit_expr(node.iter)  # node.iter is the AST For attribute
        body = []
        for stmt in node.body:
            converted = self._visit_stmt(stmt)
            if isinstance(converted, list):
                body.extend(converted)
            else:
                body.append(converted)
        return MidFor(target=target_mid, iter_expr=iter_mid, body=body)

    def _visit_If(self, node):
        test_mid = self._visit_expr(node.test)
        body = []
        for stmt in node.body:
            converted = self._visit_stmt(stmt)
            if isinstance(converted, list):
                body.extend(converted)
            else:
                body.append(converted)
        orelse = []
        for stmt in node.orelse:
            converted = self._visit_stmt(stmt)
            if isinstance(converted, list):
                orelse.extend(converted)
            else:
                orelse.append(converted)
        return MidIf(test=test_mid, body=body, orelse=orelse)

    def _visit_Expr(self, node):
        value_mid = self._visit_expr(node.value)
        return MidExprStmt(value=value_mid)

    def _visit_AugAssign(self, node):
        """Convert ``x += y`` to ``MidAssign("x", MidBinOp("+", MidName("x"), y))``."""
        op_str = _AST_OP_MAP.get(type(node.op))
        if op_str is None:
            raise NotImplementedError(f"AugAssign op {type(node.op).__name__}")
        # Use string target for dict key lookup in _values
        target_str = node.target.id if isinstance(node.target, ast.Name) else _mid_to_str(_ast_expr_to_mid(node.target))
        # Reference the current value of target
        lhs_mid = MidName(target_str)
        value_mid = self._visit_expr(node.value)
        return MidAssign(target=target_str, value=MidBinOp(op_str, lhs_mid, value_mid))

    # ------------------------------------------------------------------
    # Expression visitors
    # ------------------------------------------------------------------

    def _visit_expr(self, node):
        """Dispatch an AST expression to the appropriate handler.  Tensor-aware
        conversions (loads, stores, data_ptr, offsets, stride) are handled
        here; all other compound expressions are recursively visited so that
        tensor names nested inside arithmetic / comparison / call expressions
        are correctly resolved to tensor loads.
        """

        # --- Name (leaf) ---
        if isinstance(node, ast.Name):
            if self._in_context(node) and isinstance(node.ctx, ast.Load):
                tensor = self._context[node.id]
                return self._generate_load(tensor)
            return MidName(node.id)

        # --- Subscript of a tensor -> tensor load ---
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            if self._in_context(node.value):
                tensor = self._context[node.value.id]
                indices = (
                    node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else (node.slice,),
                )
                return self._generate_load(tensor, indices=indices)

        # --- Call: data_ptr / offsets / stride on a tensor ---
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in ("data_ptr", "offsets", "stride"):
                value = node.func.value
                tensor = None
                if self._in_context(value):
                    tensor = self._context[value.id]
                elif isinstance(value, ast.Subscript) and self._in_context(value.value):
                    tensor = self._context[value.value.id]

                if tensor is not None:
                    if attr == "data_ptr":
                        ptr_name = tensor.source.pointer_string()
                        return MidName(ptr_name)
                    if attr == "offsets":
                        dim = ast.literal_eval(node.args[0]) if node.args else None
                        return self._generate_offsets(tensor, dim=dim)
                    if attr == "stride":
                        dim = ast.literal_eval(node.args[0])
                        return MidName(tensor.source.stride_string(dim))

        # --- Attribute access on a tensor (.dtype, .shape, etc.) ---
        if isinstance(node, ast.Attribute):
            value = node.value
            if self._in_context(value):
                tensor = self._context[value.id]
                dtype_tensor = tensor.dtype
                if isinstance(dtype_tensor, Tensor):
                    if node.attr == "dtype":
                        return MidName(f"{tensor.source.pointer_string()}.type.element_ty")
                    attr_val = getattr(dtype_tensor, node.attr, None)
                    if isinstance(attr_val, Tensor):
                        # Return the tensor itself for further resolution
                        return _symbol_to_mid(Symbol(attr_val))
                    if attr_val is not None:
                        return _symbol_to_mid(Symbol(attr_val))

        # --- Compound expressions: recursively visit children so that
        #     tensor-aware Name resolution works at every nesting level. ---
        if isinstance(node, ast.BinOp):
            return MidBinOp(_resolve_binop_str(node.op), self._visit_expr(node.left), self._visit_expr(node.right))

        if isinstance(node, ast.UnaryOp):
            return MidUnaryOp(_resolve_unaryop_str(node.op), self._visit_expr(node.operand))

        if isinstance(node, ast.Compare):
            return _build_compare_chain(self._visit_expr(node.left), node.ops, node.comparators, self._visit_expr)

        if isinstance(node, ast.BoolOp):
            op_str = "and" if isinstance(node.op, ast.And) else "or"
            return MidBoolOp(op_str, [self._visit_expr(v) for v in node.values])

        if isinstance(node, ast.IfExp):
            return MidIfExp(
                self._visit_expr(node.test),
                self._visit_expr(node.body),
                self._visit_expr(node.orelse),
            )

        if isinstance(node, ast.Call):
            func_name = _resolve_call_name(node.func)
            # program_id -> MidProgramId
            base = func_name.rsplit(".", 1)[-1] if "." in func_name else func_name
            if base == "program_id":
                axis = 0
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
                    axis = node.args[0].value
                return MidProgramId(axis=axis)
            args = [self._visit_expr(a) for a in node.args]
            kwargs = {}
            for kw in node.keywords:
                kwargs[kw.arg] = self._visit_expr(kw.value)
            return MidCall(func_name, args, kwargs)

        if isinstance(node, ast.Subscript):
            return MidSubscript(
                self._visit_expr(node.value),
                _ast_slice_to_mid(node.slice),
            )

        if isinstance(node, ast.Tuple):
            return MidTuple([self._visit_expr(e) for e in node.elts])

        if isinstance(node, ast.Attribute):
            value_mid = self._visit_expr(node.value)
            value_str = _mid_to_str(value_mid)
            return MidName(f"{value_str}.{node.attr}")

        # --- Remaining leaf nodes ---
        if isinstance(node, ast.Constant):
            return MidConstant(node.value)

        if isinstance(node, ast.Starred):
            return self._visit_expr(node.value)

        if isinstance(node, ast.Slice):
            return _ast_slice_to_mid(node)

        # --- Fallback for unhandled node types ---
        return _ast_expr_to_mid(node)

    # ------------------------------------------------------------------
    # Tensor load / store / pointer generation
    # ------------------------------------------------------------------

    def _generate_load(self, tensor, indices=()):
        """Generate a :class:`MidLoad` for reading from *tensor*."""
        if tensor.ndim == 0:
            return MidName(tensor.source.name)

        pointer_mid, mask_mid, other_mid = self._generate_pointers_and_mask(tensor, indices)
        innermost_shape = tensor.innermost().shape
        if all(isinstance(s, int) for s in innermost_shape):
            innermost_shape = tuple(innermost_shape)
        else:
            innermost_shape = None
        return MidLoad(pointer=pointer_mid, mask=mask_mid, other=other_mid,
                       shape=innermost_shape)

    def _generate_store(self, tensor, value_mid, indices=()):
        """Generate a :class:`MidStore` for writing to *tensor*."""
        pointer_mid, mask_mid, _ = self._generate_pointers_and_mask(tensor, indices)
        return MidStore(pointer=pointer_mid, value=value_mid, mask=mask_mid)

    def _generate_offsets(self, tensor, dim=None):
        """Generate the Mid IR for ``tensor.offsets(dim)``."""
        # offsets() without a dim returns the overall offsets as a name.
        if dim is None:
            name = f"{tensor.source.name}_last_generated_overall_offsets"
            return MidName(name)

        # offsets(dim) returns the per-dimension offsets
        name = f"{tensor.source.name}_offsets_{dim}"
        return MidName(name)

    def _generate_pointers_and_mask(self, tensor, indices):
        """Compute pointer expression and mask for a tensor access.

        Returns ``(pointer_mid, mask_mid, other_mid)``.
        """
        # Complete indices if this is a non-source tensor
        if tensor is not tensor.source:
            indices = self._complete_indices(tensor, indices)

        # Flatten: indices from the subscript handler may contain nested tuples
        flat = []
        for index in indices:
            if isinstance(index, tuple):
                flat.extend(index)
            else:
                flat.append(index)
        indices = tuple(Symbol(index) for index in flat)

        # Invariant: base pointer name
        name_for_pointers = Symbol(f"{tensor.source.name}_pointers")
        self._invariants[name_for_pointers] = Symbol(tensor.source.pointer_string())

        # Compute offsets and mask via the tile hierarchy
        overall_offsets_sym, mask_sym = self._generate_overall_offsets_and_mask(
            tensor, indices
        )

        # Pointer = base + overall_offsets
        pointer_mid = _symbol_to_mid(name_for_pointers + overall_offsets_sym)
        mask_mid = _symbol_to_mid(mask_sym)

        # Other value for out-of-bounds
        other_mid = self._generate_other(tensor)

        return pointer_mid, mask_mid, other_mid

    def _complete_indices(self, tensor, indices):
        """Prepend program-id indices and append innermost arange indices."""
        pid_indices = self._generate_pid_indices(tensor)
        innermost_indices = self._generate_innermost_indices(tensor)
        return (
            tuple(pid_indices)
            + tuple(indices)
            + tuple(innermost_indices)
        )

    def _generate_pid_indices(self, tensor):
        """Generate program-id based indices for the outermost level of
        *tensor*.  Returns a list of :class:`Symbol` objects (names)."""
        self._invariants[type(self)._NAME_FOR_PID] = Symbol(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=LANGUAGE, ctx=ast.Load()),
                    attr="program_id",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=0)],
                keywords=[],
            )
        )

        indices = list(Tensor._unravel_index(type(self)._NAME_FOR_PID, tensor.shape))

        for dim, index in enumerate(indices):
            name = Symbol(f"{tensor.source.name}_index_{dim}")
            self._invariants[name] = index
            indices[dim] = name

        # Jagged tensor handling
        if tensor.source.jagged_dim is not None:
            seq_len_name = Symbol(tensor.source.seq_len_string())
            max_seq_len_name = Symbol(tensor.source.max_seq_len_string())

            for size in tensor.shape:
                size.find_and_replace(seq_len_name, max_seq_len_name)

            offsets_name = Symbol(tensor.source.offsets_string())
            batch_dim_index_name = Symbol(f"{tensor.source.name}_index_0")
            seq_start_name = Symbol(f"{tensor.source.name}_seq_start")
            seq_end_name = Symbol(f"{tensor.source.name}_seq_end")

            self._invariants[seq_start_name] = Symbol(
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=LANGUAGE, ctx=ast.Load()),
                        attr="load",
                        ctx=ast.Load(),
                    ),
                    args=[(offsets_name + batch_dim_index_name).node],
                    keywords=[],
                )
            )
            self._invariants[seq_end_name] = Symbol(
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=LANGUAGE, ctx=ast.Load()),
                        attr="load",
                        ctx=ast.Load(),
                    ),
                    args=[(offsets_name + batch_dim_index_name + 1).node],
                    keywords=[],
                )
            )
            self._invariants[seq_len_name] = seq_end_name - seq_start_name

        return tuple(indices)

    @staticmethod
    def _generate_other(tensor):
        """Return the ``other`` (fill) value for out-of-bounds accesses."""
        other = tensor.source.other
        if other is None:
            return None
        if isinstance(other, float) and not math.isfinite(other):
            return MidConstant(float(f"{other}"))
        return MidConstant(other)

    @staticmethod
    def _generate_slices(tensor, dim):
        """Generate slice tuple for arange indexing at *dim*."""
        return tuple(
            slice(None) if target_dim == dim else None
            for target_dim in tensor.innermost().target_dims
        )

    @staticmethod
    def _generate_innermost_indices(tensor, use_power_of_2_sizes=True):
        """Generate arange-based indices for the innermost level."""
        class _NextPowerOfTwoMaker(ast.NodeTransformer):
            def visit_Name(self, node):
                name = node.id
                if not naming.is_meta(name):
                    next_power_of_2_name = naming.make_next_power_of_2(name)
                    return Symbol(next_power_of_2_name).node
                return node

        indices = []
        for size, target_dim in zip(
            tensor.innermost().shape, tensor.innermost().target_dims
        ):
            if use_power_of_2_sizes:
                size = _NextPowerOfTwoMaker().visit(Symbol(copy.deepcopy(size)).node)
                size = Symbol(size)
            else:
                size = Symbol(size)

            slices = ASTToMidIRPass._generate_slices(tensor, target_dim)
            index_sym = size[Symbol(slices)]
            indices.append(index_sym)

        return tuple(indices)

    @staticmethod
    def _generate_overall_offsets_and_mask(tensor, indices):
        """Walk the tile hierarchy of *tensor* to compute overall offsets
        and the accumulated boundary mask.

        Returns ``(overall_offsets: Symbol, mask: Symbol)``.
        """
        indices = list(indices)

        offsets, mask = ASTToMidIRPass._generate_offsets_and_mask(tensor, indices)

        # Store per-dimension offsets on the tensor for later retrieval
        tensor._last_generated_offsets = offsets

        # overall_offsets = sum(offsets[dim] * stride[dim] for dim in range(ndim))
        overall_offsets = sum(
            offsets[source_dim] * Symbol(tensor.source.stride_string(source_dim))
            for source_dim in range(tensor.source.ndim)
        )

        # Jagged offset
        if tensor.source.jagged_dim is not None:
            overall_offsets += Symbol(
                f"{tensor.source.name}_seq_start"
            ) * Symbol(tensor.source.stride_string(tensor.source.jagged_dim))

        tensor._last_generated_overall_offsets = overall_offsets

        return overall_offsets, mask

    @staticmethod
    def _generate_offsets_and_mask(tensor, indices):
        """Propagate *indices* through the tile hierarchy, calling
        ``offsets()`` on each level to accumulate per-source-dimension
        offsets and build the boundary mask.

        Returns ``(offsets: list[Symbol], mask: Symbol)``.
        """
        offsets = [Symbol(0) for _ in range(tensor.source.ndim)]

        tensor.source._mask = Symbol(True)

        curr = tensor
        start = 0

        while isinstance(curr, type(tensor)):
            stop = start + curr.ndim
            curr_indices = indices[start:stop]
            curr._inputs = [curr_indices]
            start = stop
            curr = curr.dtype

        for level in reversed(tensor._levels):
            for tensor_ in level:
                tensor_.offsets()

        for dim, offset in enumerate(tensor.source._outputs[0]):
            offsets[dim] += offset

        # Clear temporary inputs
        curr = tensor
        while isinstance(curr, type(tensor)):
            curr._inputs.clear()
            curr = curr.dtype

        return offsets, tensor.source._mask
