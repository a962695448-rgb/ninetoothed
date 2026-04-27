"""Base classes and NativeFormatter for Mid IR."""

import math


class IRNode:
    """Base class for all Mid IR nodes."""

    def __repr__(self):
        attrs = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items()
        )
        return f"{type(self).__name__}({attrs})"


class NativeFormatter:
    """Formats Mid IR nodes into human-readable text with tile semantics.

    Uses generic operation names (not MLIR dialect prefixes), preserves
    tile metadata, and shows symbolic variable names (no SSA numbering).
    """

    def format(self, node):
        if isinstance(node, type(None)):
            return "None"
        method = getattr(self, f"_format_{type(node).__name__}", None)
        if method is not None:
            return method(node)
        return str(node)

    def _format_MidFunction(self, node):
        lines = []
        lines.append(self._format_func_header(node))
        lines.append("{")
        if node.invariants:
            lines.append("  // --- invariants ---")
            for inv in node.invariants:
                lines.append(f"  {self.format(inv)}")
        if node.body:
            lines.append("")
            lines.append("  // --- body ---")
            for stmt in node.body:
                lines.append(f"  {self.format(stmt)}")
        lines.append("}")
        return "\n".join(lines)

    def _format_func_header(self, node):
        params = []
        for param in node.params:
            params.append(self._format_func_param(param))
        return f"func {node.name}({', '.join(params)})"

    def _format_func_param(self, param):
        parts = [param.name]

        if param.is_constexpr:
            parts.append(f": constexpr {param.dtype or 'i32'}")
        elif param.ndim is not None:
            shape_strs = []
            for s in param.shape:
                shape_strs.append(str(s))
            shape = f"({', '.join(shape_strs)})"

            tile_strs = []
            if param.tile_history:
                for tile_op in param.tile_history:
                    tile_strs.append(str(tile_op))
                tile = f"({', '.join(tile_strs)})"
            else:
                tile = ""

            dtype_str = f", dtype={param.dtype}" if param.dtype else ""
            parts.append(f": tensor<shape={shape}, tile={tile}{dtype_str}>")

        return "".join(parts)

    def _format_MidParam(self, node):
        return self._format_func_param(node)

    def _format_MidInvariant(self, node):
        value = self.format(node.value)
        return f"{node.target} = {value}"

    def _format_MidAssign(self, node):
        value = self.format(node.value)
        return f"{node.target} = {value}"

    def _format_MidExprStmt(self, node):
        return self.format(node.value)

    def _format_MidReturn(self, node):
        if node.value is not None:
            return f"return {self.format(node.value)}"
        return "return"

    def _format_MidStore(self, node):
        pointer = self.format(node.pointer)
        value = self.format(node.value)
        mask = self.format(node.mask) if node.mask is not None else None
        parts = [f"store({pointer}, {value}"]
        if mask is not None:
            parts.append(f", mask={mask}")
        parts.append(")")
        return "".join(parts)

    def _format_MidFor(self, node):
        header = f"for {node.target} in range({self.format(node.iter_expr)}):"
        body_lines = [f"    {self.format(stmt)}" for stmt in node.body]
        return header + "\n" + "\n".join(body_lines)

    def _format_MidIf(self, node):
        header = f"if {self.format(node.test)}:"
        body_lines = [f"    {self.format(stmt)}" for stmt in node.body]
        result = header + "\n" + "\n".join(body_lines)
        if node.orelse:
            result += "\nelse:"
            orelse_lines = [f"    {self.format(stmt)}" for stmt in node.orelse]
            result += "\n" + "\n".join(orelse_lines)
        return result

    def _format_MidBinOp(self, node):
        left = self.format(node.lhs)
        right = self.format(node.rhs)
        op = node.op
        return f"({left} {op} {right})"

    def _format_MidUnaryOp(self, node):
        operand = self.format(node.operand)
        return f"({node.op}{operand})"

    def _format_MidCompare(self, node):
        left = self.format(node.left)
        right = self.format(node.right)
        return f"({left} {node.op} {right})"

    def _format_MidBoolOp(self, node):
        values = " & ".join(self.format(v) for v in node.values) if node.op == "and" \
            else " | ".join(self.format(v) for v in node.values)
        return f"({values})"

    def _format_MidIfExp(self, node):
        return f"({self.format(node.body)} if {self.format(node.test)} else {self.format(node.orelse)})"

    def _format_MidCall(self, node):
        args = ", ".join(self.format(a) for a in node.args)
        kwargs_parts = []
        for k, v in node.kwargs.items():
            kwargs_parts.append(f"{k}={self.format(v)}")
        all_args = args
        if kwargs_parts:
            if all_args:
                all_args += ", "
            all_args += ", ".join(kwargs_parts)
        return f"{node.func}({all_args})"

    def _format_MidName(self, node):
        return node.name

    def _format_MidConstant(self, node):
        if isinstance(node.value, float):
            if math.isnan(node.value):
                return "float('nan')"
            if node.value == float("inf"):
                return "float('inf')"
            if node.value == float("-inf"):
                return "float('-inf')"
        return str(node.value)

    def _format_MidLoad(self, node):
        pointer = self.format(node.pointer)
        mask = self.format(node.mask) if node.mask is not None else None
        other = self.format(node.other) if node.other is not None else None
        parts = [f"load({pointer}"]
        if mask is not None:
            parts.append(f", mask={mask}")
        if other is not None:
            parts.append(f", other={other}")
        parts.append(")")
        return "".join(parts)

    def _format_MidTuple(self, node):
        elts = ", ".join(self.format(e) for e in node.elts)
        return f"({elts})"

    def _format_MidSubscript(self, node):
        value = self.format(node.value)
        slice_ = self.format(node.slice)
        return f"{value}[{slice_}]"

    def _format_MidPointerExpr(self, node):
        base = self.format(node.base)
        offsets = self.format(node.offsets)
        return f"({base} + {offsets})"

    def _format_MidMaskExpr(self, node):
        conditions = " & ".join(self.format(c) for c in node.conditions)
        return f"({conditions})"

    def _format_MidProgramId(self, node):
        return f"program_id({node.axis})"

    def _format_MidArange(self, node):
        return f"arange({self.format(node.start)}, {self.format(node.end)})"

    def _format_MidTile(self, node):
        args_str = ", ".join(str(a) for a in node.args)
        return f"{node.kind}({args_str})"

    def _format_MidTensorAccess(self, node):
        return f"{node.param_name}"

    def _format_MidDataPtr(self, node):
        return f"{node.param_name}.data_ptr()"

    def _format_MidOffsets(self, node):
        return f"{node.param_name}.offsets()"

    def _format_MidStride(self, node):
        return f"{node.param_name}.stride({node.dim})"

    def _format_MidDtypeAttr(self, node):
        return f"{node.param_name}.dtype"
