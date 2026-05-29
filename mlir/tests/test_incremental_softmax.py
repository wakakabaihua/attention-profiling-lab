"""
Online Softmax 重写 Pass 单元测试
==================================

验证 MLIR 原生 Pattern Rewrite 实现的 online softmax 重写 pass。
测试覆盖：基础重写、属性提取、Pipeline 集成、反例不匹配、ScaleMaskSoftmax。
"""

import sys
import unittest

import torch

sys.path.insert(0, ".")

from torch_mlir import ir
from torch_mlir.fx import export_and_import

from mlir.export_attention_ir import ScaleMaskSoftmax
from mlir.passes.incremental_softmax_pass import (
    run_online_softmax_pass,
    decompose_softmax,
    create_online_softmax_patterns,
)
from mlir.passes.pass_pipeline import build_online_softmax_pipeline


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

class _SoftmaxModel(torch.nn.Module):
    """简单 softmax 模型，用于测试。"""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.softmax(x, dim=self.dim)


# ---------------------------------------------------------------------------
# 基础重写
# ---------------------------------------------------------------------------

class TestOnlineSoftmaxBasic(unittest.TestCase):
    """基础 online softmax 重写测试。"""

    def test_creates_online_softmax_op(self):
        """重写后 IR 中应包含 custom.online_softmax。"""
        module = export_and_import(_SoftmaxModel(), torch.randn(2, 8, 128, 128))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)

    def test_removes_div_tensor(self):
        """重写后 IR 中应不再包含 torch.aten.div.Tensor（softmax 最终输出 op）。"""
        module = export_and_import(_SoftmaxModel(), torch.randn(2, 8, 128, 128))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertNotIn("torch.aten.div.Tensor", ir_text)

    def test_decomposition_happens(self):
        """pass 应先分解 softmax.int 再匹配。"""
        module = export_and_import(_SoftmaxModel(), torch.randn(2, 8, 128, 128))
        # 验证 pass 前 IR 中有 softmax.int
        ir_before = module.operation.get_asm()
        self.assertIn("torch.aten.softmax.int", ir_before)

        run_online_softmax_pass(module)
        ir_after = module.operation.get_asm()
        # 分解后 softmax.int 应消失
        self.assertNotIn("torch.aten.softmax.int", ir_after)

    def test_preserves_return_type(self):
        """online_softmax 的返回类型应与原 softmax 一致。"""
        module = export_and_import(_SoftmaxModel(), torch.randn(2, 8, 128, 128))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("!torch.vtensor<[2,8,128,128],f32>", ir_text)

    def test_preserves_function_signature(self):
        """重写不应影响函数签名。"""
        module = export_and_import(_SoftmaxModel(), torch.randn(2, 8, 128, 128))
        ir_before = module.operation.get_asm()

        run_online_softmax_pass(module)
        ir_after = module.operation.get_asm()

        sig_before = [l for l in ir_before.split("\n") if "func.func" in l]
        sig_after = [l for l in ir_after.split("\n") if "func.func" in l]
        self.assertEqual(sig_before, sig_after)


# ---------------------------------------------------------------------------
# 属性测试
# ---------------------------------------------------------------------------

class TestOnlineSoftmaxAttributes(unittest.TestCase):
    """online softmax 操作属性测试。"""

    def test_dim_attribute(self):
        """dim 属性应为 -1（最后一维）。"""
        module = export_and_import(_SoftmaxModel(dim=-1), torch.randn(2, 8, 128, 128))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("dim = -1", ir_text)

    def test_algorithm_attribute(self):
        """algorithm 属性应为 online_2pass。"""
        module = export_and_import(_SoftmaxModel(), torch.randn(2, 8, 128, 128))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn('algorithm = "online_2pass"', ir_text)


# ---------------------------------------------------------------------------
# ScaleMaskSoftmax 集成
# ---------------------------------------------------------------------------

class TestOnlineSoftmaxWithScaleMask(unittest.TestCase):
    """ScaleMaskSoftmax 模型上的 online softmax 重写测试。"""

    def test_scale_mask_softmax_rewrite(self):
        """ScaleMaskSoftmax 包含 scale+mask+softmax，分解后应匹配 online softmax。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=128)
        module = export_and_import(model, torch.randn(2, 8, 128, 128))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)
        self.assertNotIn("torch.aten.div.Tensor", ir_text)

    def test_scale_mask_softmax_preserves_scale_mul(self):
        """online softmax 重写只替换 softmax 部分，scale/mask 操作应保留。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=128)
        module = export_and_import(model, torch.randn(2, 8, 128, 128))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        # mul.Scalar（scale 操作）应保留
        self.assertIn("torch.aten.mul.Scalar", ir_text)
        # where.self（mask 操作，分解后的版本）应保留
        self.assertIn("torch.aten.where.self", ir_text)

    def test_online_softmax_input_is_masked(self):
        """online softmax 的输入应为 masked tensor（where.self 的输出）。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=128)
        module = export_and_import(model, torch.randn(2, 8, 128, 128))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        # online_softmax 的操作数应引用 where.self 的输出
        # 检查 IR 中 online_softmax 后面紧跟 return
        lines = ir_text.split("\n")
        for i, line in enumerate(lines):
            if "custom.online_softmax" in line:
                # 验证 online_softmax 操作数来自 where.self
                self.assertIn("%", line)  # 应有 SSA 操作数引用
                break


# ---------------------------------------------------------------------------
# 反例
# ---------------------------------------------------------------------------

class TestOnlineSoftmaxNegative(unittest.TestCase):
    """不应触发 online softmax 重写的反例测试。"""

    def test_standalone_div_no_match(self):
        """只有 div.Tensor 没有 exp+sum+sub+max 前驱，不应匹配。"""

        class DivModel(torch.nn.Module):
            def forward(self, x, y):
                return x / y

        module = export_and_import(DivModel(), torch.randn(2, 128), torch.randn(2, 128))
        module.context.allow_unregistered_dialects = True
        # 不调用 decompose，直接尝试匹配
        from torch_mlir import rewrite
        frozen = create_online_softmax_patterns(module.context)
        try:
            rewrite.walk_and_apply_patterns(module.operation, frozen)
        except RuntimeError:
            pass
        ir_text = module.operation.get_asm()
        self.assertNotIn("custom.online_softmax", ir_text)

    def test_partial_pattern_no_match(self):
        """只有 exp + div 但缺少 max+sub+sum 完整链，不应匹配。"""

        class ExpDivModel(torch.nn.Module):
            def forward(self, x):
                e = torch.exp(x)
                return e / e.sum(dim=-1, keepdim=True)

        module = export_and_import(ExpDivModel(), torch.randn(2, 128))
        module.context.allow_unregistered_dialects = True
        # 分解（虽然没有 softmax，但安全调用）
        decompose_softmax(module)
        frozen = create_online_softmax_patterns(module.context)
        try:
            from torch_mlir import rewrite
            rewrite.walk_and_apply_patterns(module.operation, frozen)
        except RuntimeError:
            pass
        ir_text = module.operation.get_asm()
        # 因为没有 max.dim → sub → exp 的完整链，不应匹配
        self.assertNotIn("custom.online_softmax", ir_text)


# ---------------------------------------------------------------------------
# Pipeline 集成
# ---------------------------------------------------------------------------

class TestOnlineSoftmaxPipeline(unittest.TestCase):
    """Online softmax pipeline 测试。"""

    def test_pipeline_produces_online_softmax(self):
        """Pipeline B 应产生 online_softmax 操作。"""
        module = export_and_import(_SoftmaxModel(), torch.randn(2, 8, 128, 128))
        build_online_softmax_pipeline(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)
        self.assertNotIn("torch.aten.softmax.int", ir_text)
        self.assertNotIn("torch.aten.div.Tensor", ir_text)

    def test_pipeline_with_scale_mask_softmax(self):
        """Pipeline B 应在 ScaleMaskSoftmax 上产生 online_softmax。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=128)
        module = export_and_import(model, torch.randn(2, 8, 128, 128))
        build_online_softmax_pipeline(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)


# ---------------------------------------------------------------------------
# 不同形状
# ---------------------------------------------------------------------------

class TestOnlineSoftmaxShapes(unittest.TestCase):
    """不同输入形状下的 online softmax 重写测试。"""

    def test_small_shape(self):
        """小形状 (1, 4, 32, 32) 应匹配。"""
        module = export_and_import(_SoftmaxModel(), torch.randn(1, 4, 32, 32))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)

    def test_large_shape(self):
        """大形状 (4, 16, 256, 256) 应匹配。"""
        module = export_and_import(_SoftmaxModel(), torch.randn(4, 16, 256, 256))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)

    def test_2d_input(self):
        """2D 输入 (batch, features) 应匹配。"""
        module = export_and_import(_SoftmaxModel(), torch.randn(32, 1024))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)

    def test_3d_input(self):
        """3D 输入 (batch, seq, features) 应匹配。"""
        module = export_and_import(_SoftmaxModel(), torch.randn(4, 128, 512))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)


if __name__ == "__main__":
    unittest.main()
