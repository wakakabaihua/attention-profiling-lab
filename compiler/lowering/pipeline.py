"""
Lowering — Compiler Pipeline
==============================
将前端导入、Pass 序列和 Lowering 串联为单一编译管线入口。

编译流程:
    PyTorch Module / FX Graph
      -> Frontend Import   (compiler/frontend/fx_importer.py)
      -> Canonicalize Pass (compiler/passes/canonicalize.py)
      -> Fusion Pass       (compiler/passes/fusion.py)
      -> Validation Pass   (compiler/passes/validation.py)
      -> Lowering          (compiler/lowering/to_triton.py or to_mlir.py)
      -> Backend Execution (compiler/backends/)

CompilerPipeline 是项目的主入口，支持:
    - 从 nn.Module 或已有 IRGraph 编译
    - 选择目标 backend（reference / triton / mlir）
    - 输出各阶段 IR dump（用于调试）
    - 直接 compile_and_run 执行
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from compiler.ir.graph import IRGraph
from compiler.ir.printer import print_ir, diff_ir, format_ir
from compiler.frontend.fx_importer import import_module, import_fx_graph
from compiler.passes.canonicalize import CanonicalizationPass
from compiler.passes.fusion import ScaleMaskSoftmaxFusionPass, FusionResult
from compiler.passes.validation import ValidationPass
from compiler.lowering.to_triton import lower_to_triton_specs, TritonKernelSpec
from compiler.lowering.to_mlir import lower_to_mlir_text

if TYPE_CHECKING:
    import torch.fx as fx


# ─────────────────────────────────────────────────────────────────────
# 编译结果
# ─────────────────────────────────────────────────────────────────────

class CompilationArtifact:
    """
    编译管线的完整输出产物。

    Attributes:
        original_ir:     前端导入后的原始 IRGraph
        canonicalized_ir: Canonicalize pass 后的 IRGraph
        fused_ir:        Fusion pass 后的 IRGraph
        fusion_result:   Fusion pass 统计（fused_count 等）
        triton_specs:    Triton lowering 产生的 kernel 规格列表
        mlir_text:       MLIR lowering 产生的文本（可选）
        backend:         执行 backend 名称
    """

    def __init__(
        self,
        original_ir: IRGraph,
        canonicalized_ir: IRGraph,
        fused_ir: IRGraph,
        fusion_result: FusionResult,
        triton_specs: list[TritonKernelSpec],
        mlir_text: str,
        backend: str,
    ):
        self.original_ir = original_ir
        self.canonicalized_ir = canonicalized_ir
        self.fused_ir = fused_ir
        self.fusion_result = fusion_result
        self.triton_specs = triton_specs
        self.mlir_text = mlir_text
        self.backend = backend

    def dump(self, verbose: bool = False) -> None:
        """打印所有阶段的 IR dump 和 fusion 统计。"""
        print("\n=== Compilation Artifact ===")
        print(f"Backend: {self.backend}")
        print(f"Fusion result: {self.fusion_result}")

        if verbose:
            print_ir(self.original_ir, title="Original")
            print_ir(self.canonicalized_ir, title="After Canonicalize")

        print_ir(self.fused_ir, title="After Fusion")
        diff_ir(self.original_ir, self.fused_ir)

        if self.triton_specs:
            print("\nTriton Kernel Specs:")
            for spec in self.triton_specs:
                print(f"  {spec}")

        if self.mlir_text:
            print("\nMLIR Text:")
            print(self.mlir_text)


# ─────────────────────────────────────────────────────────────────────
# CompilerPipeline
# ─────────────────────────────────────────────────────────────────────

class CompilerPipeline:
    """
    Mini AI Compiler Pipeline 主入口。

    Usage:
        # 从 nn.Module 编译并执行
        pipeline = CompilerPipeline(backend="triton")
        result = pipeline.compile_and_run(module, scores)

        # 只编译（不执行），获取产物
        artifact = pipeline.compile_module(module, example_input=scores)
        artifact.dump()
    """

    def __init__(
        self,
        backend: str = "triton",
        verbose: bool = False,
        emit_mlir: bool = False,
    ):
        """
        Args:
            backend:    执行 backend，可选 "reference" / "triton" / "mlir"
            verbose:    是否打印所有阶段 IR dump
            emit_mlir:  是否生成 MLIR 文本 dump
        """
        if backend not in ("reference", "triton", "mlir"):
            raise ValueError(f"Unknown backend {backend!r}. Choose: reference / triton / mlir")
        self.backend = backend
        self.verbose = verbose
        self.emit_mlir = emit_mlir

        # Pass 实例
        self._canonicalize = CanonicalizationPass()
        self._fusion = ScaleMaskSoftmaxFusionPass()
        self._validation = ValidationPass()

    # ──────────────────────────────────────────────────
    # 编译入口
    # ──────────────────────────────────────────────────

    def compile_module(
        self,
        module: nn.Module,
        example_input: Optional[torch.Tensor] = None,
        graph_name: str = "attention",
    ) -> CompilationArtifact:
        """
        对 nn.Module 执行完整编译流程，返回 CompilationArtifact。

        Args:
            module:        目标 nn.Module
            example_input: 用于 tracing 的示例输入（可选，仅用于 shape 传播）
            graph_name:    输出图名称

        Returns:
            CompilationArtifact
        """
        ir_graph = import_module(module, example_input, graph_name=graph_name)
        return self._run_passes(ir_graph)

    def compile_graph(
        self,
        fx_graph: "fx.Graph",
        graph_name: str = "attention",
    ) -> CompilationArtifact:
        """
        对已有 FX Graph 执行完整编译流程。
        """
        ir_graph = import_fx_graph(fx_graph, graph_name=graph_name)
        return self._run_passes(ir_graph)

    def compile_ir(self, ir_graph: IRGraph) -> CompilationArtifact:
        """
        从已有 IRGraph 执行编译流程（跳过 frontend）。
        """
        return self._run_passes(ir_graph)

    def compile_and_run(
        self,
        module: nn.Module,
        *inputs: torch.Tensor,
        graph_name: str = "attention",
    ) -> torch.Tensor:
        """
        编译 module 并立即执行，返回输出 tensor。

        本方法是端到端快速路径:
            compile_module -> lower -> select backend -> execute
        """
        artifact = self.compile_module(module, example_input=inputs[0] if inputs else None,
                                       graph_name=graph_name)
        if self.verbose:
            artifact.dump()
        return self._execute(artifact, *inputs)

    # ──────────────────────────────────────────────────
    # 内部实现
    # ──────────────────────────────────────────────────

    def _run_passes(self, ir_graph: IRGraph) -> CompilationArtifact:
        original_ir = ir_graph

        if self.verbose:
            print_ir(original_ir, title="[Pipeline] Original IR")

        # Pass 1: Canonicalize
        canonicalized_ir = self._canonicalize.run(original_ir)
        if self.verbose:
            print_ir(canonicalized_ir, title="[Pipeline] After Canonicalize")

        # Pass 2: Fusion
        fused_ir, fusion_result = self._fusion.run(canonicalized_ir)
        if self.verbose:
            diff_ir(canonicalized_ir, fused_ir)

        # Pass 3: Validation
        val_result = self._validation.run(fused_ir)
        if not val_result.ok:
            import warnings
            warnings.warn(
                f"IR validation warnings after fusion: {val_result.errors}",
                RuntimeWarning,
                stacklevel=2,
            )

        # Lowering
        triton_specs = lower_to_triton_specs(fused_ir)
        mlir_text = lower_to_mlir_text(fused_ir) if self.emit_mlir else ""

        return CompilationArtifact(
            original_ir=original_ir,
            canonicalized_ir=canonicalized_ir,
            fused_ir=fused_ir,
            fusion_result=fusion_result,
            triton_specs=triton_specs,
            mlir_text=mlir_text,
            backend=self.backend,
        )

    def _execute(
        self, artifact: CompilationArtifact, *inputs: torch.Tensor
    ) -> torch.Tensor:
        """根据 self.backend 选择 backend 并执行。"""
        # 延迟导入避免循环依赖
        if self.backend == "reference":
            from compiler.backends.reference_backend import ReferenceBackend
            return ReferenceBackend().execute(artifact, *inputs)
        elif self.backend == "triton":
            from compiler.backends.triton_backend import TritonBackend
            return TritonBackend().execute(artifact, *inputs)
        elif self.backend == "mlir":
            from compiler.backends.mlir_backend import MLIRBackend
            return MLIRBackend().execute(artifact, *inputs)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")
