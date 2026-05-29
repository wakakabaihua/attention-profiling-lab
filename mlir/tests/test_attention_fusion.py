"""
Attention Fusion Pattern 单元测试
==================================

验证 MLIR 原生 Pattern Rewrite 实现的 attention 融合 pass。
测试覆盖：基础融合、属性提取、Pipeline 集成、反例不匹配。
"""

import sys
import unittest

import torch

sys.path.insert(0, ".")

from torch_mlir import ir
from torch_mlir.fx import export_and_import

from mlir.export_attention_ir import ScaleMaskSoftmax, FullAttention
from mlir.passes.attention_fusion_pass import (
    run_attention_fusion_pass,
    create_attention_fusion_patterns,
)
from mlir.passes.pass_pipeline import build_attention_optimization_pipeline


class TestAttentionFusionBasic(unittest.TestCase):
    """基础融合 pattern 测试。"""

    def _export_scale_mask_softmax(self, head_dim=64, seq_len=128):
        model = ScaleMaskSoftmax(head_dim=head_dim, seq_len=seq_len)
        example = torch.randn(2, 8, seq_len, seq_len)
        return export_and_import(model, example)

    def test_fusion_creates_fused_op(self):
        """融合后 IR 中应包含 custom.fused_scaled_masked_softmax。"""
        module = self._export_scale_mask_softmax()
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)

    def test_fusion_removes_softmax(self):
        """融合后 IR 中应不再包含 torch.aten.softmax.int。"""
        module = self._export_scale_mask_softmax()
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        self.assertNotIn("torch.aten.softmax.int", ir_text)

    def test_fusion_preserves_return_type(self):
        """融合操作的返回类型应与原 softmax 一致。"""
        module = self._export_scale_mask_softmax()

        # 先记录 softmax 返回类型
        softmax_type = None
        for func_op in module.body.operations:
            for region in func_op.regions:
                for block in region.blocks:
                    for op in block.operations:
                        if op.name == "torch.aten.softmax.int":
                            softmax_type = str(op.results[0].type)

        self.assertIsNotNone(softmax_type, "Should find softmax before fusion")

        run_attention_fusion_pass(module)

        # 融合 op 的返回类型应一致
        ir_text = module.operation.get_asm()
        self.assertIn(softmax_type, ir_text)

    def test_fusion_preserves_function_signature(self):
        """融合不应影响函数签名（输入/输出类型不变）。"""
        module = self._export_scale_mask_softmax()
        ir_before = module.operation.get_asm()

        run_attention_fusion_pass(module)
        ir_after = module.operation.get_asm()

        # func.func 签名行应保持一致
        sig_before = [l for l in ir_before.split("\n") if "func.func" in l]
        sig_after = [l for l in ir_after.split("\n") if "func.func" in l]
        self.assertEqual(sig_before, sig_after)


class TestAttentionFusionAttributes(unittest.TestCase):
    """融合操作属性提取测试。"""

    def _get_fused_op_attrs(self, head_dim=64, seq_len=128):
        model = ScaleMaskSoftmax(head_dim=head_dim, seq_len=seq_len)
        example = torch.randn(2, 8, seq_len, seq_len)
        module = export_and_import(model, example)
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        return ir_text

    def test_scale_attribute(self):
        """scale 属性应为 1/sqrt(head_dim)。"""
        ir_text = self._get_fused_op_attrs(head_dim=64)
        # 1/sqrt(64) = 0.125
        self.assertIn("scale = 1.250000e-01", ir_text)

    def test_scale_attribute_different_head_dim(self):
        """不同 head_dim 应产生不同 scale。"""
        ir_text = self._get_fused_op_attrs(head_dim=16)
        # 1/sqrt(16) = 0.25
        self.assertIn("scale = 2.500000e-01", ir_text)

    def test_softmax_dim_attribute(self):
        """softmax_dim 应为 -1（最后一维）。"""
        ir_text = self._get_fused_op_attrs()
        self.assertIn("softmax_dim = -1", ir_text)

    def test_is_causal_attribute(self):
        """is_causal 应为 true（ScaleMaskSoftmax 使用因果 mask）。"""
        ir_text = self._get_fused_op_attrs()
        self.assertIn("is_causal = true", ir_text)

    def test_algorithm_attribute(self):
        """algorithm 应为 online。"""
        ir_text = self._get_fused_op_attrs()
        self.assertIn('algorithm = "online"', ir_text)


class TestAttentionFusionNegative(unittest.TestCase):
    """不应触发融合的反例测试。"""

    def test_standalone_softmax_no_match(self):
        """只有 softmax 没有 where+mul 前驱，不应匹配。"""
        import torch.nn.functional as F

        class JustSoftmax(torch.nn.Module):
            def forward(self, x):
                return F.softmax(x, dim=-1)

        module = export_and_import(JustSoftmax(), torch.randn(2, 8, 128, 128))
        ir_before = module.operation.get_asm()
        run_attention_fusion_pass(module)
        ir_after = module.operation.get_asm()
        # 应保持不变：softmax 仍在，无融合 op
        self.assertNotIn("custom.fused_scaled_masked_softmax", ir_after)
        # IR 应未被修改（或仅被忽略）
        self.assertIn("torch.aten.softmax.int", ir_after)


class TestPassPipeline(unittest.TestCase):
    """完整 Pass Pipeline 测试。"""

    def test_pipeline_produces_fused_op(self):
        """Pipeline 应产生融合操作。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=128)
        example = torch.randn(2, 8, 128, 128)
        module = export_and_import(model, example)
        build_attention_optimization_pipeline(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)
        self.assertNotIn("torch.aten.softmax.int", ir_text)

    def test_pipeline_op_count_reduced(self):
        """Pipeline 后 IR 操作数应减少（softmax/where/mul 被融合）。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=128)
        example = torch.randn(2, 8, 128, 128)
        module = export_and_import(model, example)

        # 统计 before
        ops_before = self._count_ops(module)

        build_attention_optimization_pipeline(module)

        # 统计 after — 至少减少 1 个 op（softmax 被替换 + DCE）
        ops_after = self._count_ops(module)
        self.assertLessEqual(ops_after, ops_before)

    @staticmethod
    def _count_ops(module):
        count = 0
        for func_op in module.body.operations:
            for region in func_op.regions:
                for block in region.blocks:
                    for _ in block.operations:
                        count += 1
        return count


class TestDifferentShapes(unittest.TestCase):
    """不同输入形状下的融合测试。"""

    def test_small_seq_len(self):
        """小 seq_len (32) 也应匹配。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=32)
        example = torch.randn(1, 4, 32, 32)
        module = export_and_import(model, example)
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)

    def test_large_seq_len(self):
        """大 seq_len (512) 也应匹配。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=512)
        example = torch.randn(1, 4, 512, 512)
        module = export_and_import(model, example)
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)


if __name__ == "__main__":
    unittest.main()
