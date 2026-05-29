"""
Frontend — FX Graph Importer
================================
将 PyTorch FX 计算图（torch.export 或 torch.fx.symbolic_trace）
转换为内部 IRGraph。

支持的 FX 节点类型:
    placeholder     → OpType.INPUT
    output          → OpType.OUTPUT
    call_function:
      aten.mm / aten.bmm / aten.matmul  → OpType.MATMUL
      aten.mul.Scalar / aten.mul.Tensor → OpType.SCALE（若推测为 scale）
      aten.masked_fill / aten.where     → OpType.MASK
      aten._softmax / aten.softmax      → OpType.SOFTMAX

设计说明:
    - 仅识别 attention 子图涉及的核心算子，其他 op 统一以 OpType.INPUT 占位
      （shape 仍然传递），以保证图结构完整性。
    - scale_factor 通过追踪 FX 图中的常量参数提取（aten.mul 的 scalar 参数）。
    - 所有节点名称沿用 FX node.name，保证与原始图可对应。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.fx as fx

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from compiler.ir.ops import OpType
from compiler.ir.graph import IRGraph, IRNode, IRShape


# ─────────────────────────────────────────────────────────────────────
# 算子名称 → OpType 映射
# ─────────────────────────────────────────────────────────────────────

# 完全限定名（torch.ops.aten.xxx.xxx）及常见别名
_MATMUL_OPS = {
    "aten.mm.default",
    "aten.bmm.default",
    "aten.matmul.default",
}
_SCALE_OPS = {
    "aten.mul.Scalar",
    "aten.mul.Tensor",   # 有时 scale 以 Tensor 形式出现
}
_MASK_OPS = {
    "aten.masked_fill.Scalar",
    "aten.masked_fill_.Scalar",
    "aten.where.ScalarSelf",
    "aten.where.self",
}
_SOFTMAX_OPS = {
    "aten._softmax.default",
    "aten.softmax.int",
}


def _fx_op_name(node: fx.Node) -> Optional[str]:
    """从 FX node 中提取规范化的 op 名称字符串。"""
    if node.op != "call_function":
        return None
    target = node.target
    if hasattr(target, "overloadpacket"):
        # torch.ops.aten.xxx 类型
        return str(target)
    if callable(target):
        return getattr(target, "__name__", str(target))
    return str(target)


def _classify_fx_node(node: fx.Node) -> OpType:
    """将 FX node 分类为内部 OpType。"""
    if node.op == "placeholder":
        return OpType.INPUT
    if node.op == "output":
        return OpType.OUTPUT

    op_name = _fx_op_name(node)
    if op_name is None:
        return OpType.INPUT  # get_attr 等占位

    # 规范化：去掉前缀 "torch.ops."
    short = op_name.replace("torch.ops.", "")

    if short in _MATMUL_OPS:
        return OpType.MATMUL
    if short in _SCALE_OPS:
        return OpType.SCALE
    if short in _MASK_OPS:
        return OpType.MASK
    if short in _SOFTMAX_OPS:
        return OpType.SOFTMAX

    # 未知 op：作为不透明输入占位（不阻断图结构）
    return OpType.INPUT


def _extract_scale_factor(node: fx.Node) -> Optional[float]:
    """
    尝试从 aten.mul.Scalar node 中提取标量 scale_factor。

    FX 中 aten.mul.Scalar(input, scalar) 的 scalar 是 args[1]。
    """
    if len(node.args) < 2:
        return None
    scalar = node.args[1]
    if isinstance(scalar, (int, float)):
        return float(scalar)
    return None


def _extract_softmax_dim(node: fx.Node) -> int:
    """
    从 aten._softmax(input, dim, half_to_float) 提取 dim 参数。
    默认返回 -1。
    """
    if len(node.args) >= 2:
        dim = node.args[1]
        if isinstance(dim, int):
            return dim
    return -1


def _extract_shape(node: fx.Node) -> Optional[IRShape]:
    """尝试从 FX node 的 meta 中提取输出形状。"""
    meta = node.meta
    val = meta.get("val") or meta.get("example_value")
    if val is not None and hasattr(val, "shape"):
        return IRShape(list(val.shape))
    return None


def _safe_node_name(name: str) -> str:
    """确保节点名称在 IRGraph 中唯一可用（FX 已保证同图内唯一）。"""
    return name.replace(".", "_")


# ─────────────────────────────────────────────────────────────────────
# 主导入函数
# ─────────────────────────────────────────────────────────────────────

def import_fx_graph(
    fx_graph: fx.Graph,
    graph_name: str = "imported",
) -> IRGraph:
    """
    将 torch.fx.Graph 转换为内部 IRGraph。

    Args:
        fx_graph:   来自 symbolic_trace 或 torch.export 的 FX 计算图
        graph_name: 输出 IRGraph 的名称

    Returns:
        构建好的 IRGraph
    """
    ir_graph = IRGraph(name=graph_name)
    # FX node.name → IR node.name 映射
    name_map: Dict[str, str] = {}

    for fx_node in fx_graph.nodes:
        ir_name = _safe_node_name(fx_node.name)
        name_map[fx_node.name] = ir_name

        op_type = _classify_fx_node(fx_node)
        shape = _extract_shape(fx_node)

        # 收集输入（仅统计 FX node 类型的 args，过滤常量）
        inputs: List[str] = []
        for arg in fx_node.args:
            if isinstance(arg, fx.Node):
                inputs.append(name_map.get(arg.name, _safe_node_name(arg.name)))
            elif isinstance(arg, (list, tuple)):
                # output node 的 args 是一个 list/tuple 的节点
                for sub in arg:
                    if isinstance(sub, fx.Node):
                        inputs.append(name_map.get(sub.name, _safe_node_name(sub.name)))

        # 提取属性
        attrs: Dict[str, Any] = {}
        if op_type == OpType.SCALE:
            sf = _extract_scale_factor(fx_node)
            if sf is not None:
                attrs["scale_factor"] = sf
        elif op_type == OpType.MASK:
            attrs["is_causal"] = True
            attrs["mask_value"] = float("-inf")
        elif op_type == OpType.SOFTMAX:
            attrs["dim"] = _extract_softmax_dim(fx_node)

        ir_node = IRNode(
            name=ir_name,
            op_type=op_type,
            inputs=inputs,
            output_shape=shape,
            attrs=attrs,
            meta={"fx_node_name": fx_node.name, "fx_op": fx_node.op},
        )

        # 处理名称冲突（极少情况，加后缀解决）
        if ir_graph.contains(ir_name):
            suffix = 1
            while ir_graph.contains(f"{ir_name}_{suffix}"):
                suffix += 1
            ir_node.name = f"{ir_name}_{suffix}"
            name_map[fx_node.name] = ir_node.name

        ir_graph.add_node(ir_node)

    return ir_graph


def import_module(
    module: torch.nn.Module,
    example_inputs: Any,
    graph_name: str = "imported",
) -> IRGraph:
    """
    对 PyTorch Module 执行 symbolic_trace 并导入为 IRGraph。

    Args:
        module:         待分析的 nn.Module
        example_inputs: 用于 tracing 的示例输入（单个 tensor 或 tuple）
        graph_name:     输出图名称

    Returns:
        IRGraph
    """
    traced = fx.symbolic_trace(module)
    return import_fx_graph(traced.graph, graph_name=graph_name)
