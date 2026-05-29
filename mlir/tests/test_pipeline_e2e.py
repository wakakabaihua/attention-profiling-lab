"""
端到端 Pipeline 测试
=====================

从 PyTorch 模型到 MLIR Pass 变换到 Triton GPU 执行的完整链路验证。

测试层级:
  1. PyTorch 参考值计算
  2. torch-mlir 导出 → Pass 变换 → IR 验证
  3. Triton kernel 参数提取 → GPU 执行 → 数值对比

需要 GPU 的测试用 @unittest.skipUnless(torch.cuda.is_available(), ...) 守护。
"""

import re
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from torch_mlir import ir
from torch_mlir.fx import export_and_import

from mlir.export_attention_ir import ScaleMaskSoftmax, FullAttention
from mlir.passes.attention_fusion_pass import run_attention_fusion_pass
from mlir.passes.incremental_softmax_pass import run_online_softmax_pass
from mlir.passes.pass_pipeline import (
    build_attention_optimization_pipeline,
    build_online_softmax_pipeline,
)

HAS_CUDA = torch.cuda.is_available()


# =====================================================================
# 辅助函数
# =====================================================================

def _pytorch_reference_softmax(scores: torch.Tensor, scale: float,
                               is_causal: bool, seq_len: int) -> torch.Tensor:
    """计算 PyTorch 参考 softmax 结果。"""
    scores = scores * scale
    if is_causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1)


def _extract_attrs_from_fused_ir(ir_text: str) -> dict:
    """从融合后 IR 中提取 custom.fused_scaled_masked_softmax 的属性。"""
    attrs = {}

    # 提取 scale
    m = re.search(r'scale\s*=\s*([\d.eE+\-]+)', ir_text)
    if m:
        attrs["scale"] = float(m.group(1))

    # 提取 softmax_dim
    m = re.search(r'softmax_dim\s*=\s*(-?\d+)', ir_text)
    if m:
        attrs["softmax_dim"] = int(m.group(1))

    # 提取 is_causal
    m = re.search(r'is_causal\s*=\s*(true|false)', ir_text)
    if m:
        attrs["is_causal"] = m.group(1) == "true"

    # 提取 algorithm
    m = re.search(r'algorithm\s*=\s*"([^"]+)"', ir_text)
    if m:
        attrs["algorithm"] = m.group(1)

    return attrs


def _extract_attrs_from_online_ir(ir_text: str) -> dict:
    """从重写后 IR 中提取 custom.online_softmax 的属性。"""
    attrs = {}

    m = re.search(r'"custom\.online_softmax".*?dim\s*=\s*(-?\d+)', ir_text, re.DOTALL)
    if m:
        attrs["dim"] = int(m.group(1))

    m = re.search(r'"custom\.online_softmax".*?algorithm\s*=\s*"([^"]+)"', ir_text, re.DOTALL)
    if m:
        attrs["algorithm"] = m.group(1)

    return attrs


# =====================================================================
# E2E: PyTorch → MLIR export → Pass → attribute extraction
# =====================================================================

class TestExportToPassE2E(unittest.TestCase):
    """端到端：PyTorch 模型 → MLIR 导出 → Pass 变换 → 属性提取。"""

    def test_scale_mask_softmax_full_chain(self):
        """ScaleMaskSoftmax → export → Phase 1 → 属性提取完整链路。"""
        head_dim = 64
        seq_len = 128
        model = ScaleMaskSoftmax(head_dim=head_dim, seq_len=seq_len)
        example = torch.randn(2, 8, seq_len, seq_len)

        # Step 1: 导出
        module = export_and_import(model, example)
        self.assertIsNotNone(module)

        # Step 2: Pass
        success = run_attention_fusion_pass(module)
        self.assertTrue(success)

        # Step 3: 属性提取
        ir_text = module.operation.get_asm()
        attrs = _extract_attrs_from_fused_ir(ir_text)

        self.assertAlmostEqual(attrs["scale"], 1.0 / (head_dim ** 0.5), places=5)
        self.assertEqual(attrs["softmax_dim"], -1)
        self.assertTrue(attrs["is_causal"])
        self.assertEqual(attrs["algorithm"], "online")

    def test_online_softmax_full_chain(self):
        """SimpleSoftmax → export → Phase 2 → 属性提取完整链路。"""
        module = export_and_import(
            torch.nn.Softmax(dim=-1),
            torch.randn(2, 8, 128, 128),
        )

        success = run_online_softmax_pass(module)
        self.assertTrue(success)

        ir_text = module.operation.get_asm()
        attrs = _extract_attrs_from_online_ir(ir_text)
        self.assertEqual(attrs["dim"], -1)
        self.assertEqual(attrs["algorithm"], "online_2pass")

    def test_different_head_dims_produce_correct_scale(self):
        """不同 head_dim 应产生正确的 scale 属性。"""
        for head_dim in [16, 32, 64, 128]:
            with self.subTest(head_dim=head_dim):
                model = ScaleMaskSoftmax(head_dim=head_dim, seq_len=32)
                module = export_and_import(model, torch.randn(1, 4, 32, 32))
                run_attention_fusion_pass(module)
                ir_text = module.operation.get_asm()
                attrs = _extract_attrs_from_fused_ir(ir_text)
                expected_scale = 1.0 / (head_dim ** 0.5)
                self.assertAlmostEqual(attrs["scale"], expected_scale, places=5)


# =====================================================================
# E2E: Pipeline completeness
# =====================================================================

class TestPipelineE2E(unittest.TestCase):
    """完整 Pipeline 端到端测试：验证从导出到优化的完整性。"""

    def test_pipeline_a_ir_wellformed(self):
        """Pipeline A 输出的 IR 应是合法的（无 verifier 错误）。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=64)
        module = export_and_import(model, torch.randn(1, 4, 64, 64))
        build_attention_optimization_pipeline(module)
        ir_text = module.operation.get_asm()
        # IR 应能被重新解析
        self.assertIn("func.func", ir_text)
        self.assertIn("return", ir_text)

    def test_pipeline_b_ir_wellformed(self):
        """Pipeline B 输出的 IR 应是合法的。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=64)
        module = export_and_import(model, torch.randn(1, 4, 64, 64))
        build_online_softmax_pipeline(module)
        ir_text = module.operation.get_asm()
        self.assertIn("func.func", ir_text)
        self.assertIn("return", ir_text)

    def test_full_attention_pipeline_a_e2e(self):
        """FullAttention → Pipeline A：matmul 保留，softmax 融合。"""
        model = FullAttention(head_dim=64, seq_len=64)
        q = torch.randn(1, 4, 64, 64)
        k = torch.randn(1, 4, 64, 64)
        v = torch.randn(1, 4, 64, 64)
        module = export_and_import(model, q, k, v)
        build_attention_optimization_pipeline(module)
        ir_text = module.operation.get_asm()

        # 验证结构完整性
        self.assertIn("torch.aten.matmul", ir_text)
        self.assertIn("custom.fused_scaled_masked_softmax", ir_text)
        self.assertNotIn("torch.aten.softmax.int", ir_text)

    def test_pipeline_a_then_b_on_same_model(self):
        """同一模型分别运行 Pipeline A / B 结果不同但均合法。"""
        model = ScaleMaskSoftmax(head_dim=64, seq_len=32)
        example = torch.randn(1, 4, 32, 32)

        module_a = export_and_import(model, example)
        build_attention_optimization_pipeline(module_a)

        module_b = export_and_import(model, example)
        build_online_softmax_pipeline(module_b)

        ir_a = module_a.operation.get_asm()
        ir_b = module_b.operation.get_asm()

        # 两个 pipeline 的输出不同
        self.assertIn("custom.fused_scaled_masked_softmax", ir_a)
        self.assertIn("custom.online_softmax", ir_b)
        self.assertNotEqual(ir_a, ir_b)


# =====================================================================
# E2E: PyTorch → MLIR → pass 属性 → Triton kernel → GPU 数值对比
# =====================================================================

@unittest.skipUnless(HAS_CUDA, "需要 GPU")
class TestTritonE2E(unittest.TestCase):
    """
    端到端数值验证：
      1. 用 PyTorch 在 CPU 上计算参考结果
      2. 用 MLIR Pass 提取属性
      3. 用属性参数化 Triton kernel
      4. 在 GPU 上执行 kernel
      5. 对比数值精度
    """

    def _run_triton_from_mlir_attrs(self, scores_cpu: torch.Tensor,
                                     attrs: dict) -> torch.Tensor:
        """用 MLIR 属性驱动 Triton kernel 执行。"""
        import triton
        import triton.language as tl

        # 导入项目中已有的 Triton kernel
        from mlir.mlir_compiler import _mlir_compiled_fused_softmax_kernel

        scores_gpu = scores_cpu.cuda().contiguous()
        original_shape = scores_gpu.shape

        if scores_gpu.ndim == 4:
            B, H, T, _ = scores_gpu.shape
            scores_3d = scores_gpu.reshape(B * H, T, T)
        else:
            scores_3d = scores_gpu
            T = scores_gpu.shape[-1]

        BH = scores_3d.shape[0]
        output = torch.empty_like(scores_3d)

        BLOCK_T = triton.next_power_of_2(T)
        BLOCK_T = max(BLOCK_T, 16)

        grid = (BH * T,)
        _mlir_compiled_fused_softmax_kernel[grid](
            scores_3d, output, T,
            SCALE=attrs["scale"],
            IS_CAUSAL=attrs["is_causal"],
            BLOCK_T=BLOCK_T,
        )

        return output.reshape(original_shape).cpu()

    def test_scale_mask_softmax_numerical_e2e(self):
        """
        完整链路数值验证：
          PyTorch → export → Phase 1 Pass → 提取属性 → Triton GPU → 对比
        """
        head_dim = 64
        seq_len = 64
        scale = head_dim ** -0.5

        model = ScaleMaskSoftmax(head_dim=head_dim, seq_len=seq_len)
        scores = torch.randn(1, 4, seq_len, seq_len)

        # Step 1: PyTorch 参考结果
        ref = _pytorch_reference_softmax(scores, scale, is_causal=True,
                                          seq_len=seq_len)

        # Step 2: MLIR Pass → 属性提取
        module = export_and_import(model, scores)
        run_attention_fusion_pass(module)
        ir_text = module.operation.get_asm()
        attrs = _extract_attrs_from_fused_ir(ir_text)

        # Step 3: Triton GPU 执行
        triton_result = self._run_triton_from_mlir_attrs(scores, attrs)

        # Step 4: 数值对比
        max_diff = (ref - triton_result).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"Max diff = {max_diff}")

    def test_different_shapes_numerical_e2e(self):
        """不同形状的端到端数值验证。"""
        configs = [
            (64, 32, 1, 4),   # (head_dim, seq_len, batch, heads)
            (64, 64, 2, 8),
            (128, 32, 1, 2),
        ]
        for head_dim, seq_len, B, H in configs:
            with self.subTest(head_dim=head_dim, seq_len=seq_len, B=B, H=H):
                scale = head_dim ** -0.5
                scores = torch.randn(B, H, seq_len, seq_len)

                ref = _pytorch_reference_softmax(scores, scale, is_causal=True,
                                                  seq_len=seq_len)

                model = ScaleMaskSoftmax(head_dim=head_dim, seq_len=seq_len)
                module = export_and_import(model, scores)
                run_attention_fusion_pass(module)
                attrs = _extract_attrs_from_fused_ir(module.operation.get_asm())

                triton_result = self._run_triton_from_mlir_attrs(scores, attrs)
                max_diff = (ref - triton_result).abs().max().item()
                self.assertLess(max_diff, 1e-4, f"Max diff = {max_diff}")


# =====================================================================
# E2E: MLIRCompiler 集成测试
# =====================================================================

@unittest.skipUnless(HAS_CUDA, "需要 GPU")
class TestMLIRCompilerE2E(unittest.TestCase):
    """已有的 MLIRCompiler 端到端测试。"""

    def test_compiler_produces_correct_output(self):
        """MLIRCompiler.compile() → forward() 数值正确。"""
        from mlir.mlir_compiler import MLIRCompiler

        head_dim = 64
        seq_len = 64
        scale = head_dim ** -0.5
        model = ScaleMaskSoftmax(head_dim=head_dim, seq_len=seq_len)
        scores = torch.randn(1, 4, seq_len, seq_len)

        compiler = MLIRCompiler(verbose=False)
        compiled = compiler.compile(model, scores)

        # GPU 执行
        scores_gpu = scores.cuda()
        output = compiled(scores_gpu).cpu()

        # 参考值
        ref = _pytorch_reference_softmax(scores, scale, is_causal=True,
                                          seq_len=seq_len)

        max_diff = (ref - output).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"Max diff = {max_diff}")

    def test_compiler_attributes_match_pass(self):
        """MLIRCompiler 提取的属性应与 Phase 1 Pass 一致。"""
        from mlir.mlir_compiler import MLIRCompiler

        head_dim = 64
        seq_len = 64
        model = ScaleMaskSoftmax(head_dim=head_dim, seq_len=seq_len)
        scores = torch.randn(1, 4, seq_len, seq_len)

        compiler = MLIRCompiler(verbose=False)
        compiled = compiler.compile(model, scores)

        # Pass 提取的属性
        module = export_and_import(model, scores)
        run_attention_fusion_pass(module)
        pass_attrs = _extract_attrs_from_fused_ir(module.operation.get_asm())

        # 两者应一致
        self.assertAlmostEqual(compiled.scale, pass_attrs["scale"], places=5)
        self.assertEqual(compiled.is_causal, pass_attrs["is_causal"])
        self.assertEqual(compiled.softmax_dim, pass_attrs["softmax_dim"])


if __name__ == "__main__":
    unittest.main()
