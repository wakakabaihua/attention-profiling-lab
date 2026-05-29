"""
Passes — Fusion Pass
======================
实现 ScaleMaskSoftmaxFusionPass：将匹配到的 SCALE -> MASK -> SOFTMAX 子图
替换为单一 FUSED_SCALE_MASK_SOFTMAX 节点。

Pass 执行步骤:
    1. 调用 pattern_match.find_all_patterns 找到所有候选
    2. 对每个候选：
       a. 插入 FUSED_SCALE_MASK_SOFTMAX 节点（复用 graph_input、继承属性）
       b. 将原 softmax 节点的所有用户重定向到新融合节点
       c. 移除被融合的三个原始节点（SCALE、MASK、SOFTMAX）
    3. 返回新图和 FusionResult（包含统计信息）

结果合法性通过 validation pass 验证（由 pipeline 调用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from compiler.ir.graph import IRGraph

from compiler.ir.ops import OpType
from compiler.ir.graph import IRNode
from compiler.passes.pattern_match import MatchResult, find_all_patterns


# ─────────────────────────────────────────────────────────────────────
# 融合结果统计
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    """ScaleMaskSoftmaxFusionPass 的执行结果。"""

    fused_count: int                        # 成功融合的子图数量
    fused_node_names: List[str] = field(default_factory=list)   # 新融合节点名列表
    eliminated_node_names: List[str] = field(default_factory=list)  # 被删除的节点名

    def __repr__(self) -> str:
        return (
            f"FusionResult(fused={self.fused_count}, "
            f"new_nodes={self.fused_node_names}, "
            f"eliminated={self.eliminated_node_names})"
        )


# ─────────────────────────────────────────────────────────────────────
# Fusion Pass
# ─────────────────────────────────────────────────────────────────────

class ScaleMaskSoftmaxFusionPass:
    """
    将图中所有 SCALE -> MASK -> SOFTMAX 链融合为 FUSED_SCALE_MASK_SOFTMAX 节点。

    This pass:
        - 在图的副本上操作，原图不被修改
        - 对每个匹配按拓扑顺序（先出现的先处理）逐一融合
        - 融合后节点名为 fused_sms_{i}（i 从 0 起）

    Usage:
        pass_ = ScaleMaskSoftmaxFusionPass()
        new_graph, result = pass_.run(graph)
    """

    def run(self, graph: "IRGraph") -> tuple["IRGraph", FusionResult]:
        """
        执行 fusion pass。

        Args:
            graph: 输入 IRGraph（不被修改）

        Returns:
            (new_graph, FusionResult)
        """
        new_graph = graph.copy()
        candidates = find_all_patterns(new_graph)

        fused_node_names: List[str] = []
        eliminated_node_names: List[str] = []

        for idx, candidate in enumerate(candidates):
            fused_name = f"fused_sms_{idx}"
            self._apply_fusion(new_graph, candidate, fused_name)
            fused_node_names.append(fused_name)
            eliminated_node_names.extend(candidate.node_names)

        return new_graph, FusionResult(
            fused_count=len(candidates),
            fused_node_names=fused_node_names,
            eliminated_node_names=eliminated_node_names,
        )

    # ──────────────────────────────────────────────────
    # 内部实现
    # ──────────────────────────────────────────────────

    def _apply_fusion(
        self,
        graph: "IRGraph",
        candidate: MatchResult,
        fused_name: str,
    ) -> None:
        """
        将单个候选替换为融合节点。

        执行顺序:
            1. 插入新融合节点（继承来自 candidate.attrs 的属性）
            2. 将所有依赖原 softmax 输出的节点重定向到融合节点
            3. 按逆拓扑顺序（后序）删除原始节点（先删 softmax，再 mask，再 scale）
        """
        # 1. 获取原 softmax 节点的输出形状
        softmax_node = graph.get_node(candidate.graph_output)
        output_shape = softmax_node.output_shape

        # 2. 创建融合节点
        fused_node = IRNode(
            name=fused_name,
            op_type=OpType.FUSED_SCALE_MASK_SOFTMAX,
            inputs=[candidate.graph_input],
            output_shape=output_shape,
            attrs=dict(candidate.attrs),
            meta={"fused_from": candidate.node_names, "pattern": candidate.pattern_name},
        )
        graph.add_node(fused_node)

        # 3. 将所有使用旧 softmax 输出的节点重定向到融合节点
        users = graph.get_users(candidate.graph_output)
        for user in users:
            user.inputs = [
                fused_name if inp == candidate.graph_output else inp
                for inp in user.inputs
            ]

        # 4. 逆序删除原有节点（先删末端，避免被依赖检查阻断）
        for node in reversed(candidate.matched_nodes):
            # 在删除前，断开该节点的 users（使其 use-count 为零）
            for user in graph.get_users(node.name):
                user.inputs = [
                    inp for inp in user.inputs if inp != node.name
                ]
            if graph.contains(node.name):
                # 强制删除（已手动清理上游引用）
                graph._nodes.pop(node.name)
