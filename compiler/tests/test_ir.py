"""
Tests — IR Construction Tests
================================
验证 compiler/ir 模块的核心功能:
    - IRShape 创建和等值比较
    - IRNode 创建和属性访问
    - IRGraph 节点增删和拓扑排序
    - IRGraph 环检测
    - IR Printer 输出格式
"""

import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from compiler.ir.ops import OpType, get_spec
from compiler.ir.graph import IRShape, IRNode, IRGraph
from compiler.ir.printer import format_ir, diff_ir


# ─────────────────────────────────────────────────────────────────────
# IRShape
# ─────────────────────────────────────────────────────────────────────

class TestIRShape:
    def test_rank(self):
        s = IRShape([1, 12, 128, 128])
        assert s.rank == 4

    def test_dynamic_dims(self):
        s = IRShape([-1, 12, -1, -1])
        assert s.dims[0] == -1
        assert s.dims[1] == 12

    def test_equality(self):
        a = IRShape([1, 12, 128, 128])
        b = IRShape([1, 12, 128, 128])
        c = IRShape([1, 12, 64, 64])
        assert a == b
        assert a != c

    def test_repr(self):
        s = IRShape([4, 8])
        assert "4" in repr(s)
        assert "8" in repr(s)


# ─────────────────────────────────────────────────────────────────────
# IRNode
# ─────────────────────────────────────────────────────────────────────

class TestIRNode:
    def test_basic_creation(self):
        node = IRNode(name="scale_0", op_type=OpType.SCALE)
        assert node.name == "scale_0"
        assert node.op_type == OpType.SCALE
        assert node.inputs == []
        assert node.attrs == {}

    def test_with_inputs_and_attrs(self):
        node = IRNode(
            name="scale_0",
            op_type=OpType.SCALE,
            inputs=["scores"],
            attrs={"scale_factor": 0.125},
            output_shape=IRShape([1, 12, 128, 128]),
        )
        assert node.inputs == ["scores"]
        assert node.attrs["scale_factor"] == 0.125
        assert node.output_shape.dims == [1, 12, 128, 128]

    def test_repr_contains_name(self):
        node = IRNode(name="my_node", op_type=OpType.SOFTMAX)
        assert "my_node" in repr(node)
        assert "SOFTMAX" in repr(node)


# ─────────────────────────────────────────────────────────────────────
# IRGraph
# ─────────────────────────────────────────────────────────────────────

def _build_scale_mask_softmax_graph(name: str = "test") -> IRGraph:
    """构建一个 INPUT -> SCALE -> MASK -> SOFTMAX -> OUTPUT 图。"""
    g = IRGraph(name=name)
    g.add_node(IRNode("scores", OpType.INPUT))
    g.add_node(IRNode("scale_0", OpType.SCALE, inputs=["scores"],
                      attrs={"scale_factor": 0.125}))
    g.add_node(IRNode("mask_0", OpType.MASK, inputs=["scale_0"],
                      attrs={"is_causal": True, "mask_value": float("-inf")}))
    g.add_node(IRNode("softmax_0", OpType.SOFTMAX, inputs=["mask_0"],
                      attrs={"dim": -1}))
    g.add_node(IRNode("output", OpType.OUTPUT, inputs=["softmax_0"]))
    return g


class TestIRGraph:
    def test_add_and_get_node(self):
        g = IRGraph(name="test")
        node = IRNode("n1", OpType.SCALE)
        g.add_node(node)
        assert g.get_node("n1") is node
        assert g.num_nodes == 1

    def test_duplicate_name_raises(self):
        g = IRGraph(name="test")
        g.add_node(IRNode("n1", OpType.SCALE))
        with pytest.raises(ValueError, match="already exists"):
            g.add_node(IRNode("n1", OpType.MASK))

    def test_contains(self):
        g = IRGraph(name="test")
        g.add_node(IRNode("n1", OpType.SCALE))
        assert g.contains("n1")
        assert not g.contains("n2")

    def test_topological_sort(self):
        g = _build_scale_mask_softmax_graph()
        ordered = g.topological_sort()
        names = [n.name for n in ordered]
        assert names.index("scores") < names.index("scale_0")
        assert names.index("scale_0") < names.index("mask_0")
        assert names.index("mask_0") < names.index("softmax_0")

    def test_get_users(self):
        g = _build_scale_mask_softmax_graph()
        users = g.get_users("scale_0")
        assert len(users) == 1
        assert users[0].name == "mask_0"

    def test_get_input_output_nodes(self):
        g = _build_scale_mask_softmax_graph()
        assert len(g.get_input_nodes()) == 1
        assert g.get_input_nodes()[0].name == "scores"
        assert len(g.get_output_nodes()) == 1
        assert g.get_output_nodes()[0].name == "output"

    def test_cycle_detection(self):
        g = IRGraph(name="cyclic")
        g.add_node(IRNode("a", OpType.SCALE, inputs=["b"]))
        g.add_node(IRNode("b", OpType.MASK, inputs=["a"]))
        with pytest.raises(ValueError, match="cycle"):
            g.topological_sort()

    def test_copy_is_independent(self):
        g = _build_scale_mask_softmax_graph("original")
        c = g.copy()
        # 修改副本不影响原图
        c.get_node("scale_0").attrs["scale_factor"] = 999.0
        assert g.get_node("scale_0").attrs["scale_factor"] == 0.125

    def test_remove_node(self):
        g = IRGraph(name="test")
        g.add_node(IRNode("a", OpType.INPUT))
        g.add_node(IRNode("b", OpType.SCALE, inputs=["a"]))
        # 不能删除有 user 的节点
        with pytest.raises(ValueError):
            g.remove_node("a")
        # 可以删除无 user 的节点
        g.remove_node("b")
        assert not g.contains("b")


# ─────────────────────────────────────────────────────────────────────
# Printer
# ─────────────────────────────────────────────────────────────────────

class TestPrinter:
    def test_format_ir_contains_node_names(self):
        g = _build_scale_mask_softmax_graph()
        output = format_ir(g)
        assert "scale_0" in output
        assert "SCALE" in output
        assert "SOFTMAX" in output

    def test_format_ir_contains_attrs(self):
        g = _build_scale_mask_softmax_graph()
        output = format_ir(g)
        assert "0.125" in output

    def test_diff_ir_shows_changes(self):
        g1 = _build_scale_mask_softmax_graph("before")
        g2 = g1.copy()
        g2.name = "after"
        g2.add_node(IRNode("extra", OpType.SCALE, inputs=["softmax_0"]))
        output = diff_ir(g1, g2)
        assert "extra" in output


# ─────────────────────────────────────────────────────────────────────
# OpSpec Registry
# ─────────────────────────────────────────────────────────────────────

class TestOpRegistry:
    def test_get_known_spec(self):
        spec = get_spec(OpType.FUSED_SCALE_MASK_SOFTMAX)
        assert spec.num_inputs == 1
        assert spec.num_outputs == 1

    def test_unknown_op_raises(self):
        import enum
        fake_op = 9999  # 不在枚举中的值
        with pytest.raises((KeyError, AttributeError)):
            get_spec(fake_op)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
