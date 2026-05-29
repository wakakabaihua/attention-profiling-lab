"""
Frontend — Graph Utilities
============================
提供 IR 图遍历、节点搜索和子图提取等工具函数。
"""

from __future__ import annotations

from typing import Callable, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from compiler.ir.graph import IRGraph, IRNode
from compiler.ir.ops import OpType


def find_nodes_by_op(graph: "IRGraph", op_type: OpType) -> List["IRNode"]:
    """返回图中所有指定 op_type 的节点（拓扑顺序）。"""
    try:
        ordered = graph.topological_sort()
    except ValueError:
        ordered = graph.nodes
    return [n for n in ordered if n.op_type == op_type]


def get_single_user(graph: "IRGraph", name: str) -> Optional["IRNode"]:
    """
    若节点 name 恰好有一个使用者，返回该使用者；否则返回 None。
    用于验证线性链结构（不存在分叉的 def-use）。
    """
    users = graph.get_users(name)
    return users[0] if len(users) == 1 else None


def extract_linear_chain(
    graph: "IRGraph",
    start_name: str,
    op_sequence: List[OpType],
) -> Optional[List["IRNode"]]:
    """
    从 start_name 节点出发，沿唯一 def-use 链尝试匹配 op_sequence。

    Args:
        graph:        待搜索的 IRGraph
        start_name:   链的起点节点名
        op_sequence:  期望的 OpType 序列（包含起点的 op_type）

    Returns:
        若成功匹配，返回匹配的节点列表（与 op_sequence 等长）；否则返回 None。

    Example:
        chain = extract_linear_chain(g, "scale_0", [OpType.SCALE, OpType.MASK, OpType.SOFTMAX])
        # 若匹配成功, chain == [scale_node, mask_node, softmax_node]
    """
    if not op_sequence:
        return []

    if not graph.contains(start_name):
        return None

    node = graph.get_node(start_name)
    if node.op_type != op_sequence[0]:
        return None

    chain = [node]
    for expected_op in op_sequence[1:]:
        next_node = get_single_user(graph, chain[-1].name)
        if next_node is None or next_node.op_type != expected_op:
            return None
        chain.append(next_node)

    return chain


def reachable_nodes(graph: "IRGraph", root_name: str) -> Set[str]:
    """
    从 root_name 出发，沿 def-use 方向（即 root 的使用者方向）收集所有可达节点名称。
    """
    visited: Set[str] = set()
    stack = [root_name]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for user in graph.get_users(current):
            stack.append(user.name)
    return visited


def walk_nodes(
    graph: "IRGraph",
    visitor: Callable[["IRNode"], None],
    reverse: bool = False,
) -> None:
    """
    以拓扑顺序遍历图中所有节点，对每个节点调用 visitor。

    Args:
        reverse: 若为 True，按逆拓扑顺序遍历（后序）
    """
    try:
        ordered = graph.topological_sort()
    except ValueError:
        ordered = graph.nodes

    if reverse:
        ordered = list(reversed(ordered))

    for node in ordered:
        visitor(node)
