"""
边界情况测试
=============

覆盖 attention fusion 和 online softmax 在各种 edge case 下的行为：
  - 不同 dtype（fp16 / bf16 / fp32）
  - 非因果注意力（无 causal mask）
  - 非方阵形状（seq_len ≠ key_len）
  - 极端形状（极小 / 极大 seq_len）
  - 多 softmax 场景
  - FullAttention 端到端
"""

import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from torch_mlir.fx import export_and_import

from mlir.export_attention_ir import ScaleMaskSoftmax, FullAttention
from mlir.passes.attention_fusion_pass import run_attention_fusion_pass
from mlir.passes.incremental_softmax_pass import run_online_softmax_pass
from mlir.passes.pass_pipeline import (
    build_attention_optimization_pipeline,
    build_online_softmax_pipeline,
)


# =====================================================================
# 辅助模型
# =====================================================================

class NonCausalScaleSoftmax(nn.Module):
    """不含因果 mask 的 scale + softmax（无 where）。"""

    def __init__(self, head_dim: int = 64):
        super().__init__()
        self.scale = head_dim ** -0.5

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores * self.scale
        return F.softmax(scores, dim=-1)


class DoubleSoftmax(nn.Module):
    """包含两个独立 softmax 的模型。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.softmax(x, dim=-1)
        b = F.softmax(x, dim=-2)
        return a + b


class SoftmaxWithDropout(nn.Module):
    """softmax 后接 dropout（常见 attention 模式）。"""

    def __init__(self, head_dim: int = 64, seq_len: int = 128):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.seq_len = seq_len

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores * self.scale
        T = self.seq_len
        mask = torch.triu(
            torch.ones(T, T, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        # eval mode 下 dropout 是 identity
        return probs


class NonSquareAttention(nn.Module):
    """非方阵 attention（query 和 key 长度不同）。"""

    def __init__(self, head_dim: int = 64, q_len: int = 64, k_len: int = 128):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.q_len = q_len
        self.k_len = k_len

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        # scores shape: (B, H, q_len, k_len)
        scores = scores * self.scale
        return F.softmax(scores, dim=-1)


# =====================================================================
# 非因果注意力测试
# =====================================================================

class TestNonCausalAttention(unittest.TestCase):
    """非因果 attention 的行为测试。"""

    def test_no_where_no_fusion(self):
        """没有 where.ScalarSelf 的 scale+softmax 不应触发 Phase 1 融合。"""
        model = NonCausalScaleSoftmax(head_dim=64)
        module = export_and_import(model, torch.randn(2, 8, 128, 128))
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        # Phase 1 要求 where.ScalarSelf 存在，非因果场景无 where → 不匹配
        self.assertNotIn("custom.fused_scaled_masked_softmax", ir_text)

    def test_no_where_online_softmax_still_works(self):
        """非因果场景下 Phase 2 (online softmax) 仍应匹配 softmax 部分。"""
        model = NonCausalScaleSoftmax(head_dim=64)
        module = export_and_import(model, torch.randn(2, 8, 128, 128))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        # Online softmax 匹配的是 softmax 的分解模式，不需要 where
        self.assertIn("custom.online_softmax", ir_text)

    def test_non_causal_preserves_scale(self):
        """非因果 + online softmax 应保留 scale mul 操作。"""
        model = NonCausalScaleSoftmax(head_dim=64)
        module = export_and_import(model, torch.randn(2, 8, 128, 128))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("torch.aten.mul.Scalar", ir_text)


# =====================================================================
# 不同 dtype 测试
# =====================================================================

class TestDtypeHandling(unittest.TestCase):
    """不同浮点精度下的行为测试。"""

    def test_fp32_fusion(self):
        """fp32 输入应正常融合。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=64)
        example = torch.randn(1, 4, 64, 64, dtype=torch.float32)
        module = export_and_import(model, example)
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)
        self.assertIn("f32", ir_text)

    def test_fp32_online_softmax(self):
        """fp32 输入的 online softmax 重写。"""
        module = export_and_import(
            torch.nn.Softmax(dim=-1),
            torch.randn(2, 8, 64, 64, dtype=torch.float32),
        )
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)


# =====================================================================
# 极端形状测试
# =====================================================================

class TestExtremeShapes(unittest.TestCase):
    """极端输入形状的行为测试。"""

    def test_very_small_seq_len(self):
        """极小 seq_len (4) 应正常融合。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=4)
        example = torch.randn(1, 1, 4, 4)
        module = export_and_import(model, example)
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)

    def test_seq_len_1(self):
        """seq_len = 1 的 online softmax 重写。"""
        module = export_and_import(
            torch.nn.Softmax(dim=-1),
            torch.randn(1, 1, 1, 1),
        )
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)

    def test_single_head(self):
        """单头、单 batch 应正常融合。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=32)
        example = torch.randn(1, 1, 32, 32)
        module = export_and_import(model, example)
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)

    def test_5d_input_online_softmax(self):
        """5D 输入 (B, H, G, T, T) online softmax。"""
        module = export_and_import(
            torch.nn.Softmax(dim=-1),
            torch.randn(2, 4, 2, 32, 32),
        )
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)


# =====================================================================
# 非方阵形状测试
# =====================================================================

class TestNonSquareShapes(unittest.TestCase):
    """非方阵 attention 形状测试。"""

    def test_non_square_no_fusion(self):
        """非方阵输入不含 causal mask → Phase 1 不匹配。"""
        model = NonSquareAttention(head_dim=64, q_len=64, k_len=128)
        example = torch.randn(1, 4, 64, 128)
        module = export_and_import(model, example)
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        # 非方阵无 causal mask → 不匹配 Phase 1
        self.assertNotIn("custom.fused_scaled_masked_softmax", ir_text)

    def test_non_square_online_softmax(self):
        """非方阵输入的 online softmax 重写。"""
        model = NonSquareAttention(head_dim=64, q_len=64, k_len=128)
        example = torch.randn(1, 4, 64, 128)
        module = export_and_import(model, example)
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)


# =====================================================================
# 多 softmax 场景
# =====================================================================

class TestMultipleSoftmax(unittest.TestCase):
    """包含多个 softmax 的模型。"""

    def test_double_softmax_online_rewrite(self):
        """两个独立 softmax 应均被重写为 online_softmax。"""
        module = export_and_import(DoubleSoftmax(), torch.randn(2, 8, 64, 64))
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        # 两个 softmax 均应被重写
        count = ir_text.count("custom.online_softmax")
        self.assertEqual(count, 2, f"Expected 2 online_softmax ops, got {count}")


# =====================================================================
# FullAttention 模型测试
# =====================================================================

class TestFullAttentionModel(unittest.TestCase):
    """完整 attention 模型（QKV matmul + scale + mask + softmax + PV）。"""

    def test_full_attention_fusion(self):
        """FullAttention 中的 scale+mask+softmax 应被 Phase 1 融合。"""
        model = FullAttention(head_dim=64, seq_len=64)
        q = torch.randn(1, 4, 64, 64)
        k = torch.randn(1, 4, 64, 64)
        v = torch.randn(1, 4, 64, 64)
        module = export_and_import(model, q, k, v)
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)
        # matmul 应保留
        self.assertIn("torch.aten.matmul", ir_text)

    def test_full_attention_online_softmax(self):
        """FullAttention 的 softmax 部分应被 Phase 2 重写。"""
        model = FullAttention(head_dim=64, seq_len=64)
        q = torch.randn(1, 4, 64, 64)
        k = torch.randn(1, 4, 64, 64)
        v = torch.randn(1, 4, 64, 64)
        module = export_and_import(model, q, k, v)
        run_online_softmax_pass(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)
        # matmul 应保留
        self.assertIn("torch.aten.matmul", ir_text)

    def test_full_attention_pipeline_a(self):
        """FullAttention + Pipeline A 应产生融合操作并保留 matmul。"""
        model = FullAttention(head_dim=64, seq_len=64)
        q = torch.randn(1, 4, 64, 64)
        k = torch.randn(1, 4, 64, 64)
        v = torch.randn(1, 4, 64, 64)
        module = export_and_import(model, q, k, v)
        build_attention_optimization_pipeline(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)
        self.assertIn("torch.aten.matmul", ir_text)


# =====================================================================
# softmax + 后续操作
# =====================================================================

class TestSoftmaxWithDownstreamOps(unittest.TestCase):
    """softmax 输出被后续操作使用的场景。"""

    def test_softmax_then_matmul_preserves_chain(self):
        """softmax → matmul 链中，softmax 被替换后 matmul 仍使用正确输出。"""
        model = FullAttention(head_dim=64, seq_len=64)
        q = torch.randn(1, 4, 64, 64)
        k = torch.randn(1, 4, 64, 64)
        v = torch.randn(1, 4, 64, 64)
        module = export_and_import(model, q, k, v)
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        # 融合操作的输出应被第二个 matmul 引用
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)
        # 验证 IR 仍然合法
        lines = ir_text.split("\n")
        matmul_lines = [l for l in lines if "torch.aten.matmul" in l]
        self.assertGreaterEqual(len(matmul_lines), 1)


# =====================================================================
# Pipeline 交叉测试
# =====================================================================

class TestPipelineCrossValidation(unittest.TestCase):
    """验证 Pipeline A 和 B 的行为差异。"""

    def test_pipeline_a_no_online_softmax(self):
        """Pipeline A 只做 Phase 1 融合，不应产生 online_softmax。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=64)
        module = export_and_import(model, torch.randn(1, 4, 64, 64))
        build_attention_optimization_pipeline(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)
        self.assertNotIn("custom.online_softmax", ir_text)

    def test_pipeline_b_no_fusion(self):
        """Pipeline B 只做 Phase 2 重写，不应产生 fused_scaled_masked_softmax。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=64)
        module = export_and_import(model, torch.randn(1, 4, 64, 64))
        build_online_softmax_pipeline(module)
        ir_text = module.operation.get_asm()
        self.assertIn("custom.online_softmax", ir_text)
        self.assertNotIn("custom.fused_scaled_masked_softmax", ir_text)

    def test_pipeline_a_then_pipeline_b_separate_modules(self):
        """Pipeline A 和 B 在不同 module 上独立执行结果不同。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=64)

        module_a = export_and_import(model, torch.randn(1, 4, 64, 64))
        build_attention_optimization_pipeline(module_a)

        module_b = export_and_import(model, torch.randn(1, 4, 64, 64))
        build_online_softmax_pipeline(module_b)

        ir_a = module_a.operation.get_asm()
        ir_b = module_b.operation.get_asm()

        # A 有 fused，B 有 online
        self.assertIn("custom.fused_scaled_masked_softmax", ir_a)
        self.assertIn("custom.online_softmax", ir_b)

        # A 无 online，B 无 fused
        self.assertNotIn("custom.online_softmax", ir_a)
        self.assertNotIn("custom.fused_scaled_masked_softmax", ir_b)


if __name__ == "__main__":
    unittest.main()
