"""
Passes — Canonicalize Pass
============================
对 IRGraph 执行规范化变换，包括:

1. 属性规范化
   - scale_factor 统一转换为 float
   - softmax_dim 统一转换为 int
   - is_causal 统一转换为 bool
   - mask_value 统一转换为 float

2. 冗余节点清理
   - 删除没有任何 user 且非 OUTPUT 类型的叶节点（死节点）
   - 删除未被任何节点引用的 INPUT 占位节点（孤立输入）

3. 节点顺序规范化
   - 按拓扑顺序重建节点存储顺序，确保 printer 输出稳定

Canonicalize pass 应在 Pattern Match / Fusion pass 之前运行，
以统一属性格式，避免 match 因类型不一致而失败。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compiler.ir.graph import IRGraph

from compiler.ir.ops import OpType


class CanonicalizationPass:
    """
    对图执行规范化变换。

    Usage:
        new_graph = CanonicalizationPass().run(graph)
    """

    def run(self, graph: "IRGraph") -> "IRGraph":
        """
        在图的副本上执行规范化，返回新图（原图不被修改）。
        """
        new_graph = graph.copy()
        self._normalize_attrs(new_graph)
        self._remove_dead_nodes(new_graph)
        self._reorder_topological(new_graph)
        return new_graph

    # ──────────────────────────────────────────────────
    # 步骤 1: 属性规范化
    # ──────────────────────────────────────────────────

    def _normalize_attrs(self, graph: "IRGraph") -> None:
        for node in graph.nodes:
            if node.op_type == OpType.SCALE:
                if "scale_factor" in node.attrs:
                    node.attrs["scale_factor"] = float(node.attrs["scale_factor"])

            elif node.op_type == OpType.MASK:
                if "is_causal" in node.attrs:
                    node.attrs["is_causal"] = bool(node.attrs["is_causal"])
                if "mask_value" in node.attrs:
                    node.attrs["mask_value"] = float(node.attrs["mask_value"])
                else:
                    node.attrs["mask_value"] = float("-inf")

            elif node.op_type == OpType.SOFTMAX:
                if "dim" in node.attrs:
                    node.attrs["dim"] = int(node.attrs["dim"])
                else:
                    node.attrs["dim"] = -1

            elif node.op_type == OpType.FUSED_SCALE_MASK_SOFTMAX:
                for key in ("scale_factor",):
                    if key in node.attrs:
                        node.attrs[key] = float(node.attrs[key])
                for key in ("softmax_dim",):
                    if key in node.attrs:
                        node.attrs[key] = int(node.attrs[key])
                for key in ("is_causal",):
                    if key in node.attrs:
                        node.attrs[key] = bool(node.attrs[key])

    # ──────────────────────────────────────────────────
    # 步骤 2: 删除死节点
    # ──────────────────────────────────────────────────

    def _remove_dead_nodes(self, graph: "IRGraph") -> None:
        """
        反复删除无 user 的非输出节点，直到图稳定。
        （固定点迭代，处理链式死节点）
        """
        changed = True
        while changed:
            changed = False
            for node in list(graph.nodes):
                if node.op_type == OpType.OUTPUT:
                    continue
                if not graph.get_users(node.name):
                    # 检查是否是 OUTPUT 的输入（最后一个非 OUTPUT 节点不算死节点）
                    # 实际上 OUTPUT 节点的输入已通过 get_users 包含，此处不会误删
                    graph._nodes.pop(node.name)
                    changed = True

    # ──────────────────────────────────────────────────
    # 步骤 3: 拓扑顺序重建
    # ──────────────────────────────────────────────────

    def _reorder_topological(self, graph: "IRGraph") -> None:
        """按拓扑顺序重建 _nodes 字典（Python 3.7+ dict 保持插入顺序）。"""
        try:
            ordered = graph.topological_sort()
        except ValueError:
            return  # 有环时跳过排序

        new_nodes = {node.name: node for node in ordered}
        graph._nodes = new_nodes
