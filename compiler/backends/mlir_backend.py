"""
Backends — MLIR Backend
=========================
通过现有 mlir/ 子系统执行编译后的 attention 图。

职责:
    - 将 TritonKernelSpec 映射到 mlir/mlir_compiler.py 的 MLIRCompiler
    - 复用 Stage 3 的 MLIR 编译链路（PyTorch → torch-mlir → MLIR pass → Triton codegen）
    - 保证内部 IR 与 MLIR 子系统的边界清晰

依赖:
    - mlir/mlir_compiler.py（MLIRCompiler, ScaleMaskSoftmax）
    - torch-mlir（可选，若未安装则回退到 TritonBackend）

若 torch-mlir 未安装，MLIRBackend 会自动回退到 TritonBackend 并打印警告。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List, TYPE_CHECKING

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if TYPE_CHECKING:
    from compiler.lowering.pipeline import CompilationArtifact


class MLIRBackend:
    """
    MLIR 编译器 backend。

    使用 mlir/mlir_compiler.py 中的 MLIRCompiler 编译并执行
    scale + mask + softmax 子图。

    若 torch-mlir 不可用，自动回退到 TritonBackend。
    """

    def __init__(self):
        self._mlir_available = self._check_mlir()

    def execute(
        self,
        artifact: "CompilationArtifact",
        *inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行 MLIR 编译路径（工业级: IRGraph → ir.Module → Triton codegen）。

        若 torch-mlir 不可用，回退到 TritonBackend。
        """
        if not self._mlir_available:
            warnings.warn(
                "torch-mlir not available; MLIRBackend falling back to TritonBackend.",
                RuntimeWarning,
                stacklevel=2,
            )
            from compiler.backends.triton_backend import TritonBackend
            return TritonBackend().execute(artifact, *inputs)

        if not artifact.triton_specs:
            # 图中没有可融合节点，直接用 reference 执行
            from compiler.backends.reference_backend import ReferenceBackend
            return ReferenceBackend().execute(artifact, *inputs)

        scores = inputs[0]
        return self._run_via_mlir_compiler(artifact, scores)

    def _run_via_mlir_compiler(
        self, artifact: "CompilationArtifact", scores: torch.Tensor
    ) -> torch.Tensor:
        """
        工业级路径: IRGraph → ir.Module → 直接对接 MLIRCompiler 属性提取 → Triton codegen。

        流程:
            lower_to_mlir_module(artifact.optimized_graph)
              → ir.Module (custom.fused_scaled_masked_softmax 已在 IR 中，带真实属性对象)
              → MLIRCompiler._find_fused_op()      ← 无需重新运行 FusionPass
              → ir.FloatAttr / BoolAttr / IntegerAttr 属性提取
              → MLIRCompiledModule(scale, is_causal, softmax_dim) → forward()

        与原路径（text-gen 版本）的对比:
            原路径 (lossy round-trip):
                spec.scale_factor → round(1/scale²) → 重建 ScaleMaskSoftmax module
                → export_and_import()（torch-mlir 重新 trace）
                → run_attention_fusion_pass()（再次融合）
                → 提取属性 → Triton codegen

            工业级路径（本函数）:
                artifact.optimized_graph（已融合的 IRGraph）
                → lower_to_mlir_module()  直接生成 ir.Module（跳过 torch-mlir export）
                → _find_fused_op()        直接读属性（跳过 FusionPass）
                → Triton codegen
        """
        from torch_mlir import ir
        from mlir.mlir_compiler import MLIRCompiler, MLIRCompiledModule
        from compiler.lowering.to_mlir import lower_to_mlir_module

        # ── Step 1: IRGraph → ir.Module ──────────────────────────────────────────
        # lower_to_mlir_module 用 ir.Operation.create() 将 FUSED_SCALE_MASK_SOFTMAX
        # 节点转为真实 custom.fused_scaled_masked_softmax op，属性为 ir.*Attr 对象。
        mlir_module = lower_to_mlir_module(artifact.optimized_graph)

        # ── Step 2: 直接读取属性（无需 FusionPass）────────────────────────────────
        fused_op = MLIRCompiler._find_fused_op(mlir_module)
        if fused_op is None:
            # 图中没有融合节点（理论上不会走到这里，前置检查已过滤）
            from compiler.backends.reference_backend import ReferenceBackend
            return ReferenceBackend().execute(artifact, scores)

        attrs = dict(fused_op.attributes)
        scale     = float(ir.FloatAttr(attrs["scale"]).value)
        is_causal = bool(ir.BoolAttr(attrs["is_causal"]).value)
        softmax_dim = int(ir.IntegerAttr(attrs["softmax_dim"]))

        # ── Step 3: Triton codegen ────────────────────────────────────────────────
        compiled = MLIRCompiledModule(
            scale=scale,
            is_causal=is_causal,
            softmax_dim=softmax_dim,
            input_shape=tuple(scores.shape),
            compilation_log=["lower_to_mlir_module → _find_fused_op → Triton codegen"],
        )
        return compiled(scores)

    @staticmethod
    def _check_mlir() -> bool:
        """检查 torch-mlir 是否可用。"""
        try:
            import torch_mlir  # noqa: F401
            return True
        except ImportError:
            return False
