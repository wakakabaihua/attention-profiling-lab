"""
Internal IR — Graph Structure
================================
定义 Mini AI Compiler Pipeline 的内部中间表示（IR）。

核心结构:
    IRShape  — 形状描述符（支持动态维度 -1）
    IRNode   — 图中的单个操作节点
    IRGraph  — 有向无环图（DAG），节点通过名称引用

设计原则:
    - 节点通过名称（str）互相引用，避免循环引用
    - 所有图变换（pass）生成新节点或新图，不原地修改
    - shape 信息是可选的，但 fusion pass 后建议填充
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from compiler.ir.ops import OpType


# ─────────────────────────────────────────────────────────────────────
# IRShape
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IRShape:
    """
    张量形状描述符。

    dims 中 -1 表示动态维度（运行时确定）。

    示例:
        IRShape([1, 12, 128, 128])  # B=1, H=12, T=128, T=128
        IRShape([-1, 12, -1, -1])   # 动态 batch / seq_len
    """

    dims: List[int]

    @property
    def rank(self) -> int:
        return len(self.dims)

    def __repr__(self) -> str:
        return f"IRShape({self.dims})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IRShape):
            return NotImplemented
        return self.dims == other.dims


# ─────────────────────────────────────────────────────────────────────
# IRNode
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IRNode:
    """
    IR 图中的单个操作节点。

    Attributes:
        name:       节点唯一标识符（在图内唯一）
        op_type:    算子类型（来自 OpType 枚举）
        inputs:     输入节点名称列表（按位置排列）
        output_shape: 输出张量形状（可选）
        attrs:      属性字典，存储 scale_factor / is_causal / dim 等超参数
        meta:       元数据字典，存储来源 FX 节点名等调试信息
    """

    name: str
    op_type: OpType
    inputs: List[str] = field(default_factory=list)
    output_shape: Optional[IRShape] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        inputs_str = ", ".join(self.inputs) if self.inputs else ""
        shape_str = f" -> {self.output_shape}" if self.output_shape else ""
        attrs_str = f" {self.attrs}" if self.attrs else ""
        return f"IRNode({self.name}: {self.op_type.name}({inputs_str}){shape_str}{attrs_str})"


# ─────────────────────────────────────────────────────────────────────
# IRGraph
# ─────────────────────────────────────────────────────────────────────

class IRGraph:
    """
    内部 IR 计算图（有向无环图）。

    节点通过名称唯一标识，依赖关系通过 IRNode.inputs 记录。
    提供拓扑排序、用户查询（反向 def-use 链）等工具。

    Invariants:
        - 每个 node.name 在图内唯一
        - node.inputs 中所有名称必须是已存在节点
        - 无环（由 topological_sort 验证）
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self._nodes: Dict[str, IRNode] = {}   # name -> IRNode（保持插入顺序）

    # ──────────────────────────────────────────────────
    # 基础操作
    # ──────────────────────────────────────────────────

    def add_node(self, node: IRNode) -> IRNode:
        """添加节点；若名称重复则抛出 ValueError。"""
        if node.name in self._nodes:
            raise ValueError(f"Node '{node.name}' already exists in graph '{self.name}'")
        self._nodes[node.name] = node
        return node

    def get_node(self, name: str) -> IRNode:
        """获取节点；若不存在则抛出 KeyError。"""
        if name not in self._nodes:
            raise KeyError(f"Node '{name}' not found in graph '{self.name}'")
        return self._nodes[name]

    def remove_node(self, name: str) -> IRNode:
        """移除节点；若有其他节点依赖它则抛出 ValueError。"""
        users = self.get_users(name)
        if users:
            user_names = [u.name for u in users]
            raise ValueError(
                f"Cannot remove '{name}': still used by {user_names}. "
                "Remove or reroute users first."
            )
        return self._nodes.pop(name)

    def contains(self, name: str) -> bool:
        return name in self._nodes

    @property
    def nodes(self) -> List[IRNode]:
        """按插入顺序返回所有节点（不含拓扑保证）。"""
        return list(self._nodes.values())

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    # ──────────────────────────────────────────────────
    # 图分析
    # ──────────────────────────────────────────────────

    def get_users(self, name: str) -> List[IRNode]:
        """返回所有将 name 节点用作输入的节点（反向 def-use 链）。"""
        return [n for n in self._nodes.values() if name in n.inputs]

    def get_input_nodes(self) -> List[IRNode]:
        """返回所有 OpType.INPUT 节点（图的外部输入）。"""
        return [n for n in self._nodes.values() if n.op_type == OpType.INPUT]

    def get_output_nodes(self) -> List[IRNode]:
        """返回所有 OpType.OUTPUT 节点（图的最终输出）。"""
        return [n for n in self._nodes.values() if n.op_type == OpType.OUTPUT]

    def topological_sort(self) -> List[IRNode]:
        """
        Kahn 算法拓扑排序。

        Returns:
            节点列表（生产者先于消费者）

        Raises:
            ValueError: 若图中存在环
        """
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes.values():
            for inp in node.inputs:
                if inp in in_degree:
                    in_degree[node.name] += 1  # 每条入边计一次

        # 重新统计（以实际边为准）
        in_degree = {n: 0 for n in self._nodes}
        for node in self._nodes.values():
            for inp in node.inputs:
                if inp in self._nodes:
                    in_degree[node.name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        queue.sort()  # 保证确定性顺序
        result: List[IRNode] = []

        while queue:
            name = queue.pop(0)
            result.append(self._nodes[name])
            for user in self.get_users(name):
                in_degree[user.name] -= 1
                if in_degree[user.name] == 0:
                    queue.append(user.name)
                    queue.sort()

        if len(result) != len(self._nodes):
            visited = {n.name for n in result}
            cycle_nodes = [n for n in self._nodes if n not in visited]
            raise ValueError(
                f"Graph '{self.name}' contains a cycle. "
                f"Nodes involved: {cycle_nodes}"
            )
        return result

    # ──────────────────────────────────────────────────
    # 工厂方法
    # ──────────────────────────────────────────────────

    def copy(self) -> "IRGraph":
        """返回图的深拷贝（节点和属性均为新对象）。"""
        import copy
        new_graph = IRGraph(name=self.name + "_copy")
        for node in self._nodes.values():
            new_graph.add_node(copy.deepcopy(node))
        return new_graph

    def __repr__(self) -> str:
        return f"IRGraph(name={self.name!r}, nodes={self.num_nodes})"
