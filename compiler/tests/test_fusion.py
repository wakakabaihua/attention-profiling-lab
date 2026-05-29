"""
Tests — Fusion Pass Tests
============================
验证 ScaleMaskSoftmaxFusionPass 的核心功能:
    - 正确替换 SCALE -> MASK -> SOFTMAX 为 FUSED_SCALE_MASK_SOFTMAX
    - 原图不被修改
    - 融合节点继承正确属性（scale_factor, is_causal, softmax_dim）
    - 融合后图结构合法（无悬空引用、无环）
    - 原始三节点被删除
    - 融合后图可通过 ValidationPass
    - 数值正确性（ReferenceBackend 执行 fused 图结果与 unfused 一致）
"""

import sys
from pathlib import Path
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from compiler.ir.ops import OpType
from compiler.ir.graph import IRShape, IRNode, IRGraph
from compiler.passes.fusion import ScaleMaskSoftmaxFusionPass
from compiler.passes.validation import ValidationPass
from compiler.passes.canonicalize import CanonicalizationPass
from compiler.backends.reference_backend import ReferenceBackend


# ─────────────────────────────────────────────────────────────────────
# 辅助图构建
# ─────────────────────────────────────────────────────────────────────

def build_sms_graph(scale: float = 0.125) -> IRGraph:
    g = IRGraph(name="sms")
    g.add_node(IRNode("scores", OpType.INPUT,
                      output_shape=IRShape([1, 12, 128, 128])))
    g.add_node(IRNode("scale_0", OpType.SCALE, inputs=["scores"],
                      attrs={"scale_factor": scale},
                      output_shape=IRShape([1, 12, 128, 128])))
    g.add_node(IRNode("mask_0", OpType.MASK, inputs=["scale_0"],
                      attrs={"is_causal": True, "mask_value": float("-inf")},
                      output_shape=IRShape([1, 12, 128, 128])))
    g.add_node(IRNode("softmax_0", OpType.SOFTMAX, inputs=["mask_0"],
                      attrs={"dim": -1},
                      output_shape=IRShape([1, 12, 128, 128])))
    g.add_node(IRNode("output", OpType.OUTPUT, inputs=["softmax_0"]))
    return g


def build_double_sms_graph() -> IRGraph:
    g = IRGraph(name="double_sms")
    for i, scale in enumerate([0.125, 0.0625]):
        sfx = str(i)
        g.add_node(IRNode(f"inp{sfx}", OpType.INPUT,
                          output_shape=IRShape([1, 12, 128, 128])))
        g.add_node(IRNode(f"scale{sfx}", OpType.SCALE, inputs=[f"inp{sfx}"],
                          attrs={"scale_factor": scale}))
        g.add_node(IRNode(f"mask{sfx}", OpType.MASK, inputs=[f"scale{sfx}"],
                          attrs={"is_causal": True, "mask_value": float("-inf")}))
        g.add_node(IRNode(f"soft{sfx}", OpType.SOFTMAX, inputs=[f"mask{sfx}"],
                          attrs={"dim": -1}))
    g.add_node(IRNode("output", OpType.OUTPUT, inputs=["soft1"]))
    return g


# ─────────────────────────────────────────────────────────────────────
# 融合结构测试
# ─────────────────────────────────────────────────────────────────────

class TestScaleMaskSoftmaxFusionPass:
    def test_fused_node_created(self):
        g = build_sms_graph()
        pass_ = ScaleMaskSoftmaxFusionPass()
        new_g, result = pass_.run(g)
        assert result.fused_count == 1
        assert len(result.fused_node_names) == 1
        fused_name = result.fused_node_names[0]
        assert new_g.contains(fused_name)
        assert new_g.get_node(fused_name).op_type == OpType.FUSED_SCALE_MASK_SOFTMAX

    def test_original_nodes_removed(self):
        g = build_sms_graph()
        pass_ = ScaleMaskSoftmaxFusionPass()
        new_g, result = pass_.run(g)
        for eliminated in result.eliminated_node_names:
            assert not new_g.contains(eliminated), \
                f"Eliminated node '{eliminated}' should not exist in fused graph"

    def test_original_graph_unchanged(self):
        g = build_sms_graph()
        original_node_count = g.num_nodes
        pass_ = ScaleMaskSoftmaxFusionPass()
        _, _ = pass_.run(g)
        assert g.num_nodes == original_node_count
        assert g.contains("scale_0")
        assert g.contains("mask_0")
        assert g.contains("softmax_0")

    def test_fused_attrs_correct(self):
        g = build_sms_graph(scale=0.25)
        pass_ = ScaleMaskSoftmaxFusionPass()
        new_g, result = pass_.run(g)
        fused_node = new_g.get_node(result.fused_node_names[0])
        assert fused_node.attrs["scale_factor"] == pytest.approx(0.25)
        assert fused_node.attrs["is_causal"] is True
        assert fused_node.attrs["softmax_dim"] == -1

    def test_fused_node_inherits_input(self):
        g = build_sms_graph()
        pass_ = ScaleMaskSoftmaxFusionPass()
        new_g, result = pass_.run(g)
        fused_node = new_g.get_node(result.fused_node_names[0])
        # fused 节点应直接接受 "scores" 作为输入
        assert "scores" in fused_node.inputs

    def test_output_node_rewired(self):
        g = build_sms_graph()
        pass_ = ScaleMaskSoftmaxFusionPass()
        new_g, result = pass_.run(g)
        fused_name = result.fused_node_names[0]
        # output 节点应指向 fused 节点
        output_node = new_g.get_output_nodes()[0]
        assert fused_name in output_node.inputs

    def test_fused_graph_has_no_cycles(self):
        g = build_sms_graph()
        pass_ = ScaleMaskSoftmaxFusionPass()
        new_g, _ = pass_.run(g)
        # 拓扑排序不抛出 ValueError
        ordered = new_g.topological_sort()
        assert len(ordered) > 0

    def test_double_fusion(self):
        g = build_double_sms_graph()
        pass_ = ScaleMaskSoftmaxFusionPass()
        new_g, result = pass_.run(g)
        assert result.fused_count == 2
        assert len(result.fused_node_names) == 2


# ─────────────────────────────────────────────────────────────────────
# 融合后 ValidationPass
# ─────────────────────────────────────────────────────────────────────

class TestFusionValidation:
    def test_fused_graph_passes_validation(self):
        g = build_sms_graph()
        canon_g = CanonicalizationPass().run(g)
        new_g, _ = ScaleMaskSoftmaxFusionPass().run(canon_g)
        val = ValidationPass().run(new_g)
        assert val.ok, f"Validation errors: {val.errors}"

    def test_fused_graph_no_dangling_refs(self):
        g = build_sms_graph()
        new_g, _ = ScaleMaskSoftmaxFusionPass().run(g)
        val = ValidationPass().run(new_g)
        ref_errors = [e for e in val.errors if "unknown input" in e]
        assert len(ref_errors) == 0


# ─────────────────────────────────────────────────────────────────────
# 数值正确性（ReferenceBackend）
# ─────────────────────────────────────────────────────────────────────

class TestFusionCorrectness:
    """验证 fusion pass 前后执行结果一致（numerical equivalence）。"""

    @pytest.fixture
    def scores(self):
        torch.manual_seed(42)
        return torch.randn(1, 12, 128, 128, dtype=torch.float32)

    def test_reference_backend_unfused_vs_fused(self, scores):
        # 未融合图
        g_unfused = build_sms_graph()
        ref = ReferenceBackend()
        out_unfused = ref.run_graph(g_unfused, scores)

        # 融合图
        g_fused, _ = ScaleMaskSoftmaxFusionPass().run(g_unfused)
        out_fused = ref.run_graph(g_fused, scores)

        assert torch.allclose(out_unfused, out_fused, atol=1e-5), \
            f"Max diff: {(out_unfused - out_fused).abs().max():.6f}"

    def test_output_shape_preserved(self, scores):
        g = build_sms_graph()
        g_fused, _ = ScaleMaskSoftmaxFusionPass().run(g)
        out = ReferenceBackend().run_graph(g_fused, scores)
        assert out.shape == scores.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
