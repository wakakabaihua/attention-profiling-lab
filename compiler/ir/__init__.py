"""compiler.ir — 内部中间表示模块。"""

from compiler.ir.ops import OpType, OpSpec, OP_REGISTRY, get_spec
from compiler.ir.graph import IRShape, IRNode, IRGraph
from compiler.ir.printer import print_ir, format_ir, diff_ir

__all__ = [
    "OpType", "OpSpec", "OP_REGISTRY", "get_spec",
    "IRShape", "IRNode", "IRGraph",
    "print_ir", "format_ir", "diff_ir",
]
