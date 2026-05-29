"""
Passes — Pattern Matching
===========================
在内部 IRGraph 上识别可融合的 attention 子图模式。

支持的模式:
    Pattern A (ScaleMaskSoftmax):
        SCALE -> MASK -> SOFTMAX

    Pattern B (QKScaleMaskSoftmax, 可选扩展):
        MATMUL -> SCALE -> MASK -> SOFTMAX

匹配结果以 MatchResult 数据类返回，包含:
    - 匹配到的节点列表
    - 子图的原始输入节点名
    - 子图的最终输出节点名
    - 提取到的属性（scale_factor, is_causal, softmax_dim）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from compiler.ir.graph import IRGraph, IRNode

from compiler.ir.ops import OpType
from compiler.frontend.graph_utils import extract_linear_chain, find_nodes_by_op


# ─────────────────────────────────────────────────────────────────────
# 匹配结果
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """
    一次成功的子图模式匹配结果。

    Attributes:
        pattern_name:     匹配到的模式名称（如 "ScaleMaskSoftmax"）
        matched_nodes:    参与匹配的节点列表（按模式顺序）
        graph_input:      子图的原始输入节点名（pattern 链的第一个节点的 input）
        graph_output:     子图的最终输出节点名（pattern 链的最后一个节点的 name）
        attrs:            提取到的融合属性（scale_factor, is_causal, softmax_dim）
    """

    pattern_name: str
    matched_nodes: List["IRNode"]
    graph_input: str    # 喂入 scale 节点的上游节点名
    graph_output: str   # softmax 节点的名称（其他节点依赖此名查找输出）
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def node_names(self) -> List[str]:
        return [n.name for n in self.matched_nodes]

    def __repr__(self) -> str:
        return (
            f"MatchResult(pattern={self.pattern_name!r}, "
            f"nodes={self.node_names}, "
            f"input={self.graph_input!r}, output={self.graph_output!r}, "
            f"attrs={self.attrs})"
        )


# ─────────────────────────────────────────────────────────────────────
# Pattern A: SCALE -> MASK -> SOFTMAX
# ─────────────────────────────────────────────────────────────────────

_SCALE_MASK_SOFTMAX_SEQ = [OpType.SCALE, OpType.MASK, OpType.SOFTMAX]


def match_scale_mask_softmax(graph: "IRGraph") -> List[MatchResult]:
    """
    在图中搜索所有 SCALE -> MASK -> SOFTMAX 链。

    每个 SCALE 节点至多参与一个匹配（避免重复报告）。

    Returns:
        所有找到的 MatchResult 列表（可能为空）
    """
    results: List[MatchResult] = []
    already_matched: set[str] = set()

    scale_nodes = find_nodes_by_op(graph, OpType.SCALE)
    for scale_node in scale_nodes:
        if scale_node.name in already_matched:
            continue

        chain = extract_linear_chain(graph, scale_node.name, _SCALE_MASK_SOFTMAX_SEQ)
        if chain is None:
            continue

        scale_n, mask_n, softmax_n = chain

        # 提取来自 scale 节点的上游输入（scale 节点的第一个输入）
        if not scale_n.inputs:
            continue
        graph_input = scale_n.inputs[0]

        # 聚合属性
        attrs: Dict[str, Any] = {}
        attrs["scale_factor"] = scale_n.attrs.get("scale_factor", 1.0)
        attrs["is_causal"] = mask_n.attrs.get("is_causal", True)
        attrs["mask_value"] = mask_n.attrs.get("mask_value", float("-inf"))
        attrs["softmax_dim"] = softmax_n.attrs.get("dim", -1)

        result = MatchResult(
            pattern_name="ScaleMaskSoftmax",
            matched_nodes=chain,
            graph_input=graph_input,
            graph_output=softmax_n.name,
            attrs=attrs,
        )
        results.append(result)
        already_matched.update(n.name for n in chain)

    return results


# ─────────────────────────────────────────────────────────────────────
# Pattern B: MATMUL -> SCALE -> MASK -> SOFTMAX（可选扩展）
# ─────────────────────────────────────────────────────────────────────

_QK_SCALE_MASK_SOFTMAX_SEQ = [OpType.MATMUL, OpType.SCALE, OpType.MASK, OpType.SOFTMAX]


def match_qk_scale_mask_softmax(graph: "IRGraph") -> List[MatchResult]:
    """
    在图中搜索 MATMUL -> SCALE -> MASK -> SOFTMAX 链（完整 attention score）。
    """
    results: List[MatchResult] = []
    already_matched: set[str] = set()

    matmul_nodes = find_nodes_by_op(graph, OpType.MATMUL)
    for mm_node in matmul_nodes:
        if mm_node.name in already_matched:
            continue

        chain = extract_linear_chain(graph, mm_node.name, _QK_SCALE_MASK_SOFTMAX_SEQ)
        if chain is None:
            continue

        mm_n, scale_n, mask_n, softmax_n = chain

        # 完整 attention score 的输入是 matmul 的两个操作数
        if not mm_n.inputs:
            continue
        graph_input = mm_n.inputs[0]  # Q 节点

        attrs: Dict[str, Any] = {}
        attrs["scale_factor"] = scale_n.attrs.get("scale_factor", 1.0)
        attrs["is_causal"] = mask_n.attrs.get("is_causal", True)
        attrs["mask_value"] = mask_n.attrs.get("mask_value", float("-inf"))
        attrs["softmax_dim"] = softmax_n.attrs.get("dim", -1)

        result = MatchResult(
            pattern_name="QKScaleMaskSoftmax",
            matched_nodes=chain,
            graph_input=graph_input,
            graph_output=softmax_n.name,
            attrs=attrs,
        )
        results.append(result)
        already_matched.update(n.name for n in chain)

    return results


# ─────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────

def find_all_patterns(graph: "IRGraph") -> List[MatchResult]:
    """
    在图中搜索所有已知 attention 融合模式。

    优先匹配更大的模式（QKScaleMaskSoftmax > ScaleMaskSoftmax），
    避免小模式与大模式的节点重叠。
    """
    all_results: List[MatchResult] = []
    matched_names: set[str] = set()

    # 先搜索更大的模式
    for result in match_qk_scale_mask_softmax(graph):
        if not any(n in matched_names for n in result.node_names):
            all_results.append(result)
            matched_names.update(result.node_names)

    # 再搜索 ScaleMaskSoftmax（避免与已匹配节点重叠）
    for result in match_scale_mask_softmax(graph):
        if not any(n in matched_names for n in result.node_names):
            all_results.append(result)
            matched_names.update(result.node_names)

    return all_results
