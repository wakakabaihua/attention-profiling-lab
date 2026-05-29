"""
Tests — Pattern Match Tests
===============================
验证 compiler/passes/pattern_match.py 的核心功能:
    - 能稳定识别 SCALE -> MASK -> SOFTMAX 模式
    - 错误 pattern 不误匹配（负样本）
    - 多个 pattern 不重叠匹配
    - 属性提取正确
"""

import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from compiler.ir.ops import OpType
from compiler.ir.graph import IRShape, IRNode, IRGraph
from compiler.passes.pattern_match import (
    match_scale_mask_softmax,
    match_qk_scale_mask_softmax,
    find_all_patterns,
)


# ─────────────────────────────────────────────────────────────────────
# 图构建工具
# ─────────────────────────────────────────────────────────────────────

def build_sms_graph(scale: float = 0.125) -> IRGraph:
    """构建标准 INPUT -> SCALE -> MASK -> SOFTMAX -> OUTPUT 图。"""
    g = IRGraph(name="sms")
    g.add_node(IRNode("scores", OpType.INPUT, output_shape=IRShape([1, 12, 128, 128])))
    g.add_node(IRNode("scale_0", OpType.SCALE, inputs=["scores"],
                      attrs={"scale_factor": scale}))
    g.add_node(IRNode("mask_0", OpType.MASK, inputs=["scale_0"],
                      attrs={"is_causal": True, "mask_value": float("-inf")}))
    g.add_node(IRNode("softmax_0", OpType.SOFTMAX, inputs=["mask_0"],
                      attrs={"dim": -1}))
    g.add_node(IRNode("output", OpType.OUTPUT, inputs=["softmax_0"]))
    return g


def build_qk_sms_graph() -> IRGraph:
    """构建 Q,K INPUT -> MATMUL -> SCALE -> MASK -> SOFTMAX -> OUTPUT 图。"""
    g = IRGraph(name="qk_sms")
    shape = IRShape([1, 12, 128, 64])
    g.add_node(IRNode("q", OpType.INPUT, output_shape=shape))
    g.add_node(IRNode("k", OpType.INPUT, output_shape=shape))
    score_shape = IRShape([1, 12, 128, 128])
    g.add_node(IRNode("matmul_0", OpType.MATMUL, inputs=["q", "k"],
                      output_shape=score_shape))
    g.add_node(IRNode("scale_0", OpType.SCALE, inputs=["matmul_0"],
                      attrs={"scale_factor": 0.125}))
    g.add_node(IRNode("mask_0", OpType.MASK, inputs=["scale_0"],
                      attrs={"is_causal": True, "mask_value": float("-inf")}))
    g.add_node(IRNode("softmax_0", OpType.SOFTMAX, inputs=["mask_0"],
                      attrs={"dim": -1}))
    g.add_node(IRNode("output", OpType.OUTPUT, inputs=["softmax_0"]))
    return g


def build_no_mask_graph() -> IRGraph:
    """构建 INPUT -> SCALE -> SOFTMAX（没有 MASK）图，不应匹配 ScaleMaskSoftmax。"""
    g = IRGraph(name="no_mask")
    g.add_node(IRNode("scores", OpType.INPUT))
    g.add_node(IRNode("scale_0", OpType.SCALE, inputs=["scores"],
                      attrs={"scale_factor": 0.125}))
    g.add_node(IRNode("softmax_0", OpType.SOFTMAX, inputs=["scale_0"],
                      attrs={"dim": -1}))
    g.add_node(IRNode("output", OpType.OUTPUT, inputs=["softmax_0"]))
    return g


def build_double_sms_graph() -> IRGraph:
    """构建两条独立的 SMS 链（测试多 pattern 不重叠）。"""
    g = IRGraph(name="double_sms")
    g.add_node(IRNode("s1", OpType.INPUT))
    g.add_node(IRNode("scale1", OpType.SCALE, inputs=["s1"], attrs={"scale_factor": 0.125}))
    g.add_node(IRNode("mask1", OpType.MASK, inputs=["scale1"], attrs={"is_causal": True}))
    g.add_node(IRNode("soft1", OpType.SOFTMAX, inputs=["mask1"], attrs={"dim": -1}))

    g.add_node(IRNode("s2", OpType.INPUT))
    g.add_node(IRNode("scale2", OpType.SCALE, inputs=["s2"], attrs={"scale_factor": 0.0625}))
    g.add_node(IRNode("mask2", OpType.MASK, inputs=["scale2"], attrs={"is_causal": True}))
    g.add_node(IRNode("soft2", OpType.SOFTMAX, inputs=["mask2"], attrs={"dim": -1}))

    g.add_node(IRNode("output", OpType.OUTPUT, inputs=["soft2"]))
    return g


# ─────────────────────────────────────────────────────────────────────
# 测试：正样本
# ─────────────────────────────────────────────────────────────────────

class TestMatchScaleMaskSoftmax:
    def test_matches_standard_graph(self):
        g = build_sms_graph()
        results = match_scale_mask_softmax(g)
        assert len(results) == 1
        result = results[0]
        assert result.pattern_name == "ScaleMaskSoftmax"
        assert len(result.matched_nodes) == 3
        assert result.matched_nodes[0].op_type == OpType.SCALE
        assert result.matched_nodes[1].op_type == OpType.MASK
        assert result.matched_nodes[2].op_type == OpType.SOFTMAX

    def test_extracts_correct_attrs(self):
        g = build_sms_graph(scale=0.25)
        results = match_scale_mask_softmax(g)
        assert len(results) == 1
        attrs = results[0].attrs
        assert attrs["scale_factor"] == pytest.approx(0.25)
        assert attrs["is_causal"] is True
        assert attrs["softmax_dim"] == -1

    def test_extracts_graph_input_and_output(self):
        g = build_sms_graph()
        results = match_scale_mask_softmax(g)
        result = results[0]
        # graph_input 应是 scale 节点的上游输入（scores）
        assert result.graph_input == "scores"
        # graph_output 应是 softmax 节点名
        assert result.graph_output == "softmax_0"

    def test_matches_double_patterns(self):
        g = build_double_sms_graph()
        results = match_scale_mask_softmax(g)
        assert len(results) == 2
        # 确认两个 pattern 的 scale_factor 不同
        scales = {r.attrs["scale_factor"] for r in results}
        assert 0.125 in scales
        assert 0.0625 in scales

    def test_no_duplicate_matching(self):
        g = build_double_sms_graph()
        results = match_scale_mask_softmax(g)
        # 确认节点不重叠
        all_names = [name for r in results for name in r.node_names]
        assert len(all_names) == len(set(all_names))


# ─────────────────────────────────────────────────────────────────────
# 测试：负样本（不应误匹配）
# ─────────────────────────────────────────────────────────────────────

class TestNegativePatternMatch:
    def test_no_match_without_mask(self):
        g = build_no_mask_graph()
        results = match_scale_mask_softmax(g)
        assert len(results) == 0

    def test_no_match_empty_graph(self):
        g = IRGraph(name="empty")
        results = match_scale_mask_softmax(g)
        assert len(results) == 0

    def test_no_match_input_only(self):
        g = IRGraph(name="only_input")
        g.add_node(IRNode("inp", OpType.INPUT))
        results = match_scale_mask_softmax(g)
        assert len(results) == 0

    def test_no_match_branching_scale(self):
        """若 scale 节点有两个 user（分支），不应匹配为 SMS 链。"""
        g = IRGraph(name="branching")
        g.add_node(IRNode("scores", OpType.INPUT))
        g.add_node(IRNode("scale_0", OpType.SCALE, inputs=["scores"],
                          attrs={"scale_factor": 0.125}))
        # scale_0 有两个 user：mask_0 和 direct_out
        g.add_node(IRNode("mask_0", OpType.MASK, inputs=["scale_0"],
                          attrs={"is_causal": True}))
        g.add_node(IRNode("softmax_0", OpType.SOFTMAX, inputs=["mask_0"],
                          attrs={"dim": -1}))
        g.add_node(IRNode("direct_out", OpType.OUTPUT, inputs=["scale_0"]))
        # 此时 scale_0 有两个 user（mask_0 和 direct_out），不满足单用户链要求
        results = match_scale_mask_softmax(g)
        assert len(results) == 0


# ─────────────────────────────────────────────────────────────────────
# 测试：QKScaleMaskSoftmax
# ─────────────────────────────────────────────────────────────────────

class TestMatchQKScaleMaskSoftmax:
    def test_matches_qk_graph(self):
        g = build_qk_sms_graph()
        results = match_qk_scale_mask_softmax(g)
        assert len(results) == 1
        result = results[0]
        assert result.pattern_name == "QKScaleMaskSoftmax"
        assert len(result.matched_nodes) == 4
        assert result.matched_nodes[0].op_type == OpType.MATMUL


# ─────────────────────────────────────────────────────────────────────
# 测试：find_all_patterns 优先级
# ─────────────────────────────────────────────────────────────────────

class TestFindAllPatterns:
    def test_qk_takes_priority_over_sms(self):
        """在 QKScaleMaskSoftmax 图中，大模式应优先匹配，不产生 SMS 子集匹配。"""
        g = build_qk_sms_graph()
        results = find_all_patterns(g)
        pattern_names = [r.pattern_name for r in results]
        assert "QKScaleMaskSoftmax" in pattern_names
        # 相同节点不被 SMS 重复匹配
        qk_nodes = {n for r in results if r.pattern_name == "QKScaleMaskSoftmax"
                    for n in r.node_names}
        sms_results = [r for r in results if r.pattern_name == "ScaleMaskSoftmax"]
        for r in sms_results:
            assert not any(n in qk_nodes for n in r.node_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
