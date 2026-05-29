"""
Tests — TVM Backend
=====================
验证 TVM Relax 后端的完整流程：
1. relax_importer: IRGraph → tvm.IRModule
2. TVMBackend.execute() → 数值正确性（与 PyTorch baseline 比对）
3. 无 GPU 时的 graceful fallback
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
from compiler.passes.canonicalize import CanonicalizationPass
from compiler.lowering.pipeline import CompilerPipeline, CompilationArtifact
from compiler.backends.reference_backend import ReferenceBackend

# ─────────────────────────────────────────────────────────────────────
# TVM 可用性检查
# ─────────────────────────────────────────────────────────────────────

def _tvm_available() -> bool:
    try:
        import tvm
        import tvm.relax
        return True
    except ImportError:
        return False


TVM_AVAILABLE = _tvm_available()
CUDA_AVAILABLE = torch.cuda.is_available()

requires_tvm = pytest.mark.skipif(
    not TVM_AVAILABLE, reason="TVM not installed"
)
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA not available"
)


# ─────────────────────────────────────────────────────────────────────
# 辅助：构建融合后的 IRGraph
# ─────────────────────────────────────────────────────────────────────

def _build_manual_sms_ir(scale: float = 0.125, B=1, H=12, T=128) -> IRGraph:
    """手动构建融合前的 IRGraph（不依赖 FX trace）。"""
    g = IRGraph(name="manual_sms")
    g.add_node(IRNode("scores", OpType.INPUT,
                      output_shape=IRShape([B, H, T, T])))
    g.add_node(IRNode("scale_0", OpType.SCALE, inputs=["scores"],
                      attrs={"scale_factor": scale},
                      output_shape=IRShape([B, H, T, T])))
    g.add_node(IRNode("mask_0", OpType.MASK, inputs=["scale_0"],
                      attrs={"is_causal": True, "mask_value": float("-inf")},
                      output_shape=IRShape([B, H, T, T])))
    g.add_node(IRNode("softmax_0", OpType.SOFTMAX, inputs=["mask_0"],
                      attrs={"dim": -1},
                      output_shape=IRShape([B, H, T, T])))
    g.add_node(IRNode("output", OpType.OUTPUT, inputs=["softmax_0"]))
    return g


def _compile_to_artifact(B=1, H=12, T=128, D=64) -> CompilationArtifact:
    """手动构建 IRGraph 并编译为 CompilationArtifact。"""
    ir = _build_manual_sms_ir(scale=D ** -0.5, B=B, H=H, T=T)
    pipeline = CompilerPipeline(backend="reference")
    return pipeline.compile_ir(ir)


# ─────────────────────────────────────────────────────────────────────
# Tests: relax_importer
# ─────────────────────────────────────────────────────────────────────

@requires_tvm
class TestRelaxImporter:
    def test_lower_to_relax_returns_irmodule(self):
        from tvm_integration.relax_importer import lower_to_relax
        import tvm
        artifact = _compile_to_artifact()
        mod = lower_to_relax(artifact.fused_ir, input_shape=(1, 12, 128, 128))
        assert isinstance(mod, tvm.IRModule)

    def test_relax_module_has_main_function(self):
        from tvm_integration.relax_importer import lower_to_relax
        artifact = _compile_to_artifact()
        mod = lower_to_relax(artifact.fused_ir, input_shape=(1, 12, 128, 128))
        assert "main" in mod

    def test_relax_ir_text_contains_softmax(self):
        from tvm_integration.relax_importer import print_relax_ir
        artifact = _compile_to_artifact()
        ir_text = print_relax_ir(artifact.fused_ir, input_shape=(1, 12, 128, 128))
        assert "softmax" in ir_text.lower()

    def test_relax_ir_text_contains_multiply(self):
        from tvm_integration.relax_importer import print_relax_ir
        artifact = _compile_to_artifact()
        ir_text = print_relax_ir(artifact.fused_ir, input_shape=(1, 12, 128, 128))
        assert "multiply" in ir_text.lower()

    def test_lower_to_relax_raises_without_fused_node(self):
        from tvm_integration.relax_importer import lower_to_relax
        # 构建一个没有 FUSED 节点的空图
        empty_graph = IRGraph(name="empty")
        with pytest.raises(ValueError, match="FUSED_SCALE_MASK_SOFTMAX"):
            lower_to_relax(empty_graph)

    def test_build_relax_module_noncausal(self):
        from tvm_integration.relax_importer import _build_relax_module
        import tvm
        mod = _build_relax_module(
            scale=0.125, is_causal=False, softmax_dim=-1,
            shape=(1, 12, 128, 128)
        )
        assert isinstance(mod, tvm.IRModule)
        ir_text = mod.script()
        # non-causal 不应包含 tril
        assert "tril" not in ir_text.lower()

    def test_build_relax_module_causal(self):
        from tvm_integration.relax_importer import _build_relax_module
        import tvm
        mod = _build_relax_module(
            scale=0.125, is_causal=True, softmax_dim=-1,
            shape=(1, 12, 128, 128)
        )
        ir_text = mod.script()
        assert "tril" in ir_text.lower()


# ─────────────────────────────────────────────────────────────────────
# Tests: TVMBackend — requires CUDA
# ─────────────────────────────────────────────────────────────────────

@requires_tvm
@requires_cuda
class TestTVMBackendCUDA:
    def test_tvm_backend_execute_returns_tensor(self):
        from compiler.backends.tvm_backend import TVMBackend
        artifact = _compile_to_artifact()
        backend = TVMBackend()
        scores = torch.randn(1, 12, 128, 128, dtype=torch.float16, device="cuda")
        out = backend.execute(artifact, scores)
        assert isinstance(out, torch.Tensor)

    def test_tvm_backend_output_shape(self):
        from compiler.backends.tvm_backend import TVMBackend
        artifact = _compile_to_artifact()
        backend = TVMBackend()
        scores = torch.randn(1, 12, 128, 128, dtype=torch.float16, device="cuda")
        out = backend.execute(artifact, scores)
        assert out.shape == scores.shape

    def test_tvm_backend_softmax_rows_sum_to_one(self):
        from compiler.backends.tvm_backend import TVMBackend
        artifact = _compile_to_artifact()
        backend = TVMBackend()
        scores = torch.randn(1, 12, 128, 128, dtype=torch.float16, device="cuda")
        out = backend.execute(artifact, scores)
        row_sums = out.float().sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2), \
            f"Softmax rows not summing to 1: {row_sums[0,0,:3]}"

    def test_tvm_backend_matches_reference(self):
        """TVM 输出应与 ReferenceBackend 数值接近（fp16 精度）。"""
        from compiler.backends.tvm_backend import TVMBackend
        artifact = _compile_to_artifact()
        tvm_backend = TVMBackend()
        ref_backend = ReferenceBackend()

        scores_cpu = torch.randn(1, 12, 128, 128)
        scores_cuda = scores_cpu.half().cuda()

        ref_out = ref_backend.execute(artifact, scores_cpu).half()
        tvm_out = tvm_backend.execute(artifact, scores_cuda).cpu()

        assert torch.allclose(ref_out, tvm_out, atol=1e-2, rtol=1e-2), \
            f"TVM and reference outputs differ: max_diff={( ref_out - tvm_out).abs().max():.4f}"

    def test_tvm_backend_caching(self):
        """相同形状第二次调用应使用缓存（不重新编译）。"""
        from compiler.backends.tvm_backend import TVMBackend
        artifact = _compile_to_artifact()
        backend = TVMBackend(cache_compiled=True)
        scores = torch.randn(1, 12, 128, 128, dtype=torch.float16, device="cuda")

        out1 = backend.execute(artifact, scores)
        out2 = backend.execute(artifact, scores)  # should hit cache
        assert torch.allclose(out1, out2, atol=1e-4)

    def test_tvm_backend_rejects_cpu_tensor(self):
        from compiler.backends.tvm_backend import TVMBackend
        artifact = _compile_to_artifact()
        backend = TVMBackend()
        scores_cpu = torch.randn(1, 12, 128, 128, dtype=torch.float16)
        with pytest.raises(RuntimeError, match="CUDA"):
            backend.execute(artifact, scores_cpu)


# ─────────────────────────────────────────────────────────────────────
# Tests: TVMBackend — CPU-only (no CUDA needed)
# ─────────────────────────────────────────────────────────────────────

@requires_tvm
class TestTVMBackendNoCUDA:
    def test_tvm_backend_available_flag(self):
        from compiler.backends.tvm_backend import TVMBackend
        backend = TVMBackend()
        assert backend._tvm_available is True

    def test_tvm_backend_finds_fused_node(self):
        from compiler.backends.tvm_backend import TVMBackend
        artifact = _compile_to_artifact()
        backend = TVMBackend()
        fused = backend._find_fused_node(artifact.fused_ir)
        assert fused is not None
        assert fused.op_type == OpType.FUSED_SCALE_MASK_SOFTMAX

    def test_tvm_backend_find_fused_node_none_on_empty(self):
        from compiler.backends.tvm_backend import TVMBackend
        backend = TVMBackend()
        empty = IRGraph(name="empty")
        assert backend._find_fused_node(empty) is None


class TestTVMBackendFallback:
    """当 TVM 不可用时应 graceful fallback 到 ReferenceBackend。"""

    def test_fallback_when_tvm_unavailable(self, monkeypatch):
        from compiler.backends import tvm_backend as tvm_mod
        # Build a backend instance and patch _tvm_available on the instance
        backend = tvm_mod.TVMBackend.__new__(tvm_mod.TVMBackend)
        backend._tvm_available = False
        backend._target_str = "cuda"
        backend._cache = {}

        artifact = _compile_to_artifact()
        scores = torch.randn(1, 12, 128, 128)
        # Should not raise; falls back to ReferenceBackend (CPU tensor OK)
        out = backend.execute(artifact, scores)
        assert isinstance(out, torch.Tensor)
