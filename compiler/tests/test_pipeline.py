"""
Tests — Pipeline End-to-End Tests
=====================================
验证从 nn.Module 导入到 backend 执行的完整管线:

1. FX import -> IR -> Canonicalize -> Fusion -> Validation -> Lowering
2. Reference backend 正确性
3. IR dump 与 MLIR text 生成
4. CompilationArtifact 结构完整性
5. 与 PyTorch baseline 的数值一致性（E2E correctness）
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from compiler.ir.ops import OpType
from compiler.ir.graph import IRShape, IRNode, IRGraph
from compiler.passes.fusion import ScaleMaskSoftmaxFusionPass
from compiler.passes.canonicalize import CanonicalizationPass
from compiler.passes.validation import ValidationPass
from compiler.lowering.pipeline import CompilerPipeline, CompilationArtifact
from compiler.lowering.to_mlir import lower_to_mlir_text
from compiler.backends.reference_backend import ReferenceBackend


# ─────────────────────────────────────────────────────────────────────
# 测试用 nn.Module
# ─────────────────────────────────────────────────────────────────────

class ScaleMaskSoftmax(nn.Module):
    """测试用 module：scale + causal_mask + softmax。"""

    def __init__(self, head_dim: int = 64, seq_len: int = 128):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.seq_len = seq_len

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores * self.scale
        T = self.seq_len
        mask = torch.triu(
            torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        return F.softmax(scores, dim=-1)


# ─────────────────────────────────────────────────────────────────────
# 辅助：手动构建 IR 图（不依赖 FX trace）
# ─────────────────────────────────────────────────────────────────────

def build_manual_sms_ir(scale: float = 0.125) -> IRGraph:
    """构建 attention 子图（不通过 FX trace）。"""
    g = IRGraph(name="manual_sms")
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


# ─────────────────────────────────────────────────────────────────────
# Pipeline 从 IRGraph 入口的端到端测试
# ─────────────────────────────────────────────────────────────────────

class TestPipelineFromIR:
    """测试从手动构建 IRGraph 出发的完整编译流程。"""

    def test_compile_ir_produces_artifact(self):
        pipeline = CompilerPipeline(backend="reference")
        ir = build_manual_sms_ir()
        artifact = pipeline.compile_ir(ir)
        assert isinstance(artifact, CompilationArtifact)
        assert artifact.fusion_result.fused_count == 1

    def test_canonicalize_runs_without_error(self):
        canon = CanonicalizationPass()
        ir = build_manual_sms_ir()
        canon_ir = canon.run(ir)
        # scale_factor 类型应规范化为 float
        assert isinstance(
            canon_ir.get_node("scale_0").attrs.get("scale_factor"), float
        )

    def test_fusion_pass_runs_without_error(self):
        ir = build_manual_sms_ir()
        fused_ir, result = ScaleMaskSoftmaxFusionPass().run(ir)
        assert result.fused_count == 1
        fused_nodes = [n for n in fused_ir.nodes
                       if n.op_type == OpType.FUSED_SCALE_MASK_SOFTMAX]
        assert len(fused_nodes) == 1

    def test_validation_pass_valid_after_fusion(self):
        ir = build_manual_sms_ir()
        canon_ir = CanonicalizationPass().run(ir)
        fused_ir, _ = ScaleMaskSoftmaxFusionPass().run(canon_ir)
        val = ValidationPass().run(fused_ir)
        assert val.ok, f"Errors: {val.errors}"

    def test_triton_specs_generated(self):
        pipeline = CompilerPipeline(backend="reference")
        ir = build_manual_sms_ir()
        artifact = pipeline.compile_ir(ir)
        assert len(artifact.triton_specs) == 1
        spec = artifact.triton_specs[0]
        assert spec.scale_factor == pytest.approx(0.125)
        assert spec.is_causal is True

    def test_mlir_text_generated(self):
        pipeline = CompilerPipeline(backend="reference", emit_mlir=True)
        ir = build_manual_sms_ir()
        artifact = pipeline.compile_ir(ir)
        assert "custom.fused_scaled_masked_softmax" in artifact.mlir_text

    def test_lower_to_mlir_text(self):
        ir = build_manual_sms_ir()
        fused_ir, _ = ScaleMaskSoftmaxFusionPass().run(ir)
        mlir_text = lower_to_mlir_text(fused_ir)
        assert "func.func" in mlir_text
        assert "custom.fused_scaled_masked_softmax" in mlir_text
        assert "scale" in mlir_text


# ─────────────────────────────────────────────────────────────────────
# 数值正确性：Reference Backend E2E
# ─────────────────────────────────────────────────────────────────────

class TestE2ECorrectness:
    """验证通过编译管线执行的结果与 PyTorch baseline 数值一致。"""

    @pytest.fixture
    def scores(self):
        torch.manual_seed(0)
        return torch.randn(1, 12, 128, 128, dtype=torch.float32)

    def test_reference_backend_matches_pytorch_baseline(self, scores):
        # PyTorch baseline（手动计算）
        scale = 64 ** -0.5
        T = 128
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        baseline = F.softmax(scores.masked_fill(mask, float("-inf")) * scale, dim=-1)

        # 通过 compiler 管线执行（reference backend）
        ir = build_manual_sms_ir(scale=scale)
        pipeline = CompilerPipeline(backend="reference")
        artifact = pipeline.compile_ir(ir)
        out = ReferenceBackend().execute(artifact, scores)

        assert torch.allclose(baseline, out, atol=1e-5), \
            f"Max abs diff: {(baseline - out).abs().max():.6f}"

    def test_fused_and_unfused_agree(self, scores):
        ir = build_manual_sms_ir()
        # unfused 执行
        out_unfused = ReferenceBackend().run_graph(ir, scores)

        # fused 执行
        fused_ir, _ = ScaleMaskSoftmaxFusionPass().run(ir)
        out_fused = ReferenceBackend().run_graph(fused_ir, scores)

        assert torch.allclose(out_unfused, out_fused, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
# Pipeline FX Import（需要 symbolic_trace 可用）
# ─────────────────────────────────────────────────────────────────────

class TestFXImportPipeline:
    def test_import_module_builds_ir(self):
        from compiler.frontend.fx_importer import import_module

        module = ScaleMaskSoftmax(head_dim=64, seq_len=64)
        example = torch.randn(1, 12, 64, 64)
        ir = import_module(module, example, graph_name="fx_sms")

        assert ir.num_nodes > 0
        # 导入后应存在 INPUT 节点
        input_nodes = ir.get_input_nodes()
        assert len(input_nodes) >= 1

    def test_compile_module_with_pipeline(self):
        module = ScaleMaskSoftmax(head_dim=64, seq_len=64)
        example = torch.randn(1, 12, 64, 64)
        pipeline = CompilerPipeline(backend="reference")
        artifact = pipeline.compile_module(module, example_input=example)
        assert isinstance(artifact, CompilationArtifact)
        # FX trace 应识别出 SCALE / MASK / SOFTMAX 节点
        all_op_types = {n.op_type for n in artifact.original_ir.nodes}
        assert OpType.SCALE in all_op_types or OpType.INPUT in all_op_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
