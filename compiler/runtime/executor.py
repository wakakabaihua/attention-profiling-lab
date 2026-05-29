"""
Runtime — Executor
====================
根据 CompilationArtifact 和 backend 选择执行编译结果。

Executor 是 pipeline.py 中 CompilerPipeline._execute() 的更完整版本，
提供:
    - 执行前 correctness 检查（reference 对比）
    - 多次 warm-up 执行
    - 可选的逐 backend 执行结果收集

与 BenchmarkRunner 的分工:
    - Executor:          执行单次 forward pass，返回 tensor
    - BenchmarkRunner:   多次计时，返回 latency / speedup 统计
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if TYPE_CHECKING:
    from compiler.lowering.pipeline import CompilationArtifact


@dataclass
class ExecutionResult:
    """单次执行结果。"""

    output: torch.Tensor
    backend: str
    correct: Optional[bool] = None      # 与 reference 对比的正确性结论
    max_abs_err: Optional[float] = None  # 最大绝对误差
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        correct_str = f", correct={self.correct}" if self.correct is not None else ""
        err_str = f", max_err={self.max_abs_err:.2e}" if self.max_abs_err is not None else ""
        return f"ExecutionResult(backend={self.backend!r}{correct_str}{err_str})"


class Executor:
    """
    统一执行入口。

    Usage:
        executor = Executor(backend="triton", check_correctness=True)
        result = executor.run(artifact, scores)
        print(result)
    """

    _ATOL = 1e-3
    _RTOL = 1e-3

    def __init__(
        self,
        backend: str = "triton",
        check_correctness: bool = True,
        warmup_iters: int = 2,
    ):
        if backend not in ("reference", "triton", "mlir"):
            raise ValueError(f"Unknown backend: {backend!r}")
        self.backend = backend
        self.check_correctness = check_correctness
        self.warmup_iters = warmup_iters

    def run(
        self,
        artifact: "CompilationArtifact",
        *inputs: torch.Tensor,
    ) -> ExecutionResult:
        """
        执行一次 forward pass 并（可选地）验证正确性。

        Args:
            artifact: CompilationArtifact（包含 fused_ir 和 triton_specs）
            *inputs:  与图 INPUT 节点对应的 tensor 列表

        Returns:
            ExecutionResult
        """
        # Warm-up
        if self.warmup_iters > 0 and inputs and inputs[0].is_cuda:
            backend_fn = self._get_backend_fn(artifact)
            for _ in range(self.warmup_iters):
                backend_fn(*inputs)
            torch.cuda.synchronize()

        # 执行
        backend_fn = self._get_backend_fn(artifact)
        output = backend_fn(*inputs)

        result = ExecutionResult(output=output, backend=self.backend)

        # 正确性检查
        if self.check_correctness and self.backend != "reference":
            result.correct, result.max_abs_err = self._check_correctness(
                artifact, output, *inputs
            )
            if not result.correct:
                result.warnings.append(
                    f"Correctness check FAILED: max_abs_err={result.max_abs_err:.4e} "
                    f"(atol={self._ATOL})"
                )
                warnings.warn(result.warnings[-1], RuntimeWarning, stacklevel=2)

        return result

    # ──────────────────────────────────────────────────
    # 内部实现
    # ──────────────────────────────────────────────────

    def _get_backend_fn(self, artifact: "CompilationArtifact"):
        if self.backend == "reference":
            from compiler.backends.reference_backend import ReferenceBackend
            b = ReferenceBackend()
            return lambda *inputs: b.execute(artifact, *inputs)
        elif self.backend == "triton":
            from compiler.backends.triton_backend import TritonBackend
            b = TritonBackend()
            return lambda *inputs: b.execute(artifact, *inputs)
        elif self.backend == "mlir":
            from compiler.backends.mlir_backend import MLIRBackend
            b = MLIRBackend()
            return lambda *inputs: b.execute(artifact, *inputs)
        raise ValueError(f"Unknown backend: {self.backend!r}")

    def _check_correctness(
        self,
        artifact: "CompilationArtifact",
        output: torch.Tensor,
        *inputs: torch.Tensor,
    ):
        from compiler.backends.reference_backend import ReferenceBackend
        ref_output = ReferenceBackend().execute(artifact, *inputs)
        try:
            match = torch.allclose(output.float(), ref_output.float(),
                                   atol=self._ATOL, rtol=self._RTOL)
            max_err = float((output.float() - ref_output.float()).abs().max())
            return match, max_err
        except Exception as e:
            return False, float("inf")
