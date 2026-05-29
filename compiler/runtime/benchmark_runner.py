"""
Runtime — Benchmark Runner
============================
对统一输入在多个 backend 上运行 benchmark，输出:
    - 各 backend 的 latency 统计（mean / std / min）
    - baseline vs fused speedup
    - correctness 验证结果（与 reference 对比）
    - 固定格式的 Markdown 报告

与 benchmarks/compare_all_backends.py 的分工:
    - BenchmarkRunner:              负责 compiler 管线内部的 backend 对比
    - compare_all_backends.py:      负责 Stage 1-4 跨阶段的端到端对比
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if TYPE_CHECKING:
    from compiler.lowering.pipeline import CompilationArtifact


# ─────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BackendStats:
    """单个 backend 的 benchmark 统计。"""

    backend: str
    latency_ms: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    @property
    def std_ms(self) -> float:
        if len(self.latency_ms) < 2:
            return 0.0
        mean = self.mean_ms
        return (sum((x - mean) ** 2 for x in self.latency_ms) / (len(self.latency_ms) - 1)) ** 0.5

    @property
    def min_ms(self) -> float:
        return min(self.latency_ms) if self.latency_ms else 0.0


@dataclass
class BenchmarkReport:
    """完整 benchmark 报告。"""

    title: str
    timestamp: str
    stats: Dict[str, BackendStats] = field(default_factory=dict)
    correctness: Dict[str, bool] = field(default_factory=dict)
    max_abs_err: Dict[str, float] = field(default_factory=dict)
    baseline_backend: str = "reference"

    @property
    def speedups(self) -> Dict[str, float]:
        """相对 baseline_backend 的加速比。"""
        if self.baseline_backend not in self.stats:
            return {}
        base_ms = self.stats[self.baseline_backend].mean_ms
        if base_ms == 0:
            return {}
        return {
            name: base_ms / s.mean_ms
            for name, s in self.stats.items()
            if s.mean_ms > 0
        }

    def print_table(self) -> None:
        """打印 benchmark 结果表格到控制台。"""
        speedups = self.speedups
        print(f"\n{'='*60}")
        print(f"  {self.title}")
        print(f"  {self.timestamp}")
        print(f"{'='*60}")
        header = f"{'Backend':<20} {'Mean(ms)':>10} {'Std(ms)':>9} {'Min(ms)':>9} {'Speedup':>9} {'Correct':>9}"
        print(header)
        print("-" * 60)
        for name, stat in self.stats.items():
            speedup = speedups.get(name, 1.0)
            correct = self.correctness.get(name, True)
            err = self.max_abs_err.get(name)
            err_str = f"{err:.2e}" if err is not None else "N/A"
            correct_str = "✓" if correct else f"✗ ({err_str})"
            print(
                f"{name:<20} {stat.mean_ms:>10.3f} {stat.std_ms:>9.3f} "
                f"{stat.min_ms:>9.3f} {speedup:>9.2f}x {correct_str:>9}"
            )
        print("=" * 60)

    def to_markdown(self) -> str:
        """生成 Markdown 格式的 benchmark 报告。"""
        speedups = self.speedups
        lines = [
            f"# Benchmark Report: {self.title}",
            f"",
            f"**Date**: {self.timestamp}",
            f"**Baseline**: {self.baseline_backend}",
            f"",
            f"## Results",
            f"",
            f"| Backend | Mean(ms) | Std(ms) | Min(ms) | Speedup | Correct |",
            f"|---------|----------|---------|---------|---------|---------|",
        ]
        for name, stat in self.stats.items():
            speedup = speedups.get(name, 1.0)
            correct = self.correctness.get(name, True)
            err = self.max_abs_err.get(name)
            err_str = f"{err:.2e}" if err is not None else "N/A"
            correct_str = "✓" if correct else f"✗ (err={err_str})"
            lines.append(
                f"| {name} | {stat.mean_ms:.3f} | {stat.std_ms:.3f} | "
                f"{stat.min_ms:.3f} | {speedup:.2f}x | {correct_str} |"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# BenchmarkRunner
# ─────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    多 backend benchmark 运行器。

    Usage:
        runner = BenchmarkRunner(backends=["reference", "triton"], n_iter=100)
        report = runner.run(artifact, scores)
        report.print_table()
    """

    def __init__(
        self,
        backends: Optional[List[str]] = None,
        n_iter: int = 100,
        warmup_iters: int = 10,
        check_correctness: bool = True,
        title: str = "Compiler Pipeline Benchmark",
    ):
        self.backends = backends or ["reference", "triton"]
        self.n_iter = n_iter
        self.warmup_iters = warmup_iters
        self.check_correctness = check_correctness
        self.title = title

    def run(
        self,
        artifact: "CompilationArtifact",
        *inputs: torch.Tensor,
    ) -> BenchmarkReport:
        """
        在所有配置的 backend 上运行 benchmark。

        Args:
            artifact: CompilationArtifact（包含 fused_ir 和 triton_specs）
            *inputs:  输入 tensor

        Returns:
            BenchmarkReport
        """
        report = BenchmarkReport(
            title=self.title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # 先跑 reference，收集基准值
        ref_output = None
        if "reference" in self.backends:
            stats, output = self._benchmark_backend("reference", artifact, *inputs)
            report.stats["reference"] = stats
            report.correctness["reference"] = True
            report.max_abs_err["reference"] = 0.0
            ref_output = output

        # 跑其他 backend
        for backend in self.backends:
            if backend == "reference":
                continue
            stats, output = self._benchmark_backend(backend, artifact, *inputs)
            report.stats[backend] = stats

            if self.check_correctness and ref_output is not None:
                correct, err = self._check(output, ref_output)
                report.correctness[backend] = correct
                report.max_abs_err[backend] = err
            else:
                report.correctness[backend] = True

        return report

    # ──────────────────────────────────────────────────
    # 内部实现
    # ──────────────────────────────────────────────────

    def _benchmark_backend(
        self,
        backend: str,
        artifact: "CompilationArtifact",
        *inputs: torch.Tensor,
    ):
        backend_fn = self._make_fn(backend, artifact)
        is_cuda = inputs and inputs[0].is_cuda

        # Warm-up
        for _ in range(self.warmup_iters):
            out = backend_fn(*inputs)
        if is_cuda:
            torch.cuda.synchronize()

        # 计时
        latencies: List[float] = []
        for _ in range(self.n_iter):
            if is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = backend_fn(*inputs)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                out = backend_fn(*inputs)
                latencies.append((time.perf_counter() - t0) * 1000)

        stats = BackendStats(backend=backend, latency_ms=latencies)
        return stats, out

    def _make_fn(self, backend: str, artifact: "CompilationArtifact"):
        if backend == "reference":
            from compiler.backends.reference_backend import ReferenceBackend
            b = ReferenceBackend()
        elif backend == "triton":
            from compiler.backends.triton_backend import TritonBackend
            b = TritonBackend()
        elif backend == "mlir":
            from compiler.backends.mlir_backend import MLIRBackend
            b = MLIRBackend()
        else:
            raise ValueError(f"Unknown backend: {backend!r}")
        return lambda *inputs: b.execute(artifact, *inputs)

    @staticmethod
    def _check(output: torch.Tensor, ref: torch.Tensor, atol: float = 1e-3):
        try:
            match = torch.allclose(output.float(), ref.float(), atol=atol, rtol=1e-3)
            err = float((output.float() - ref.float()).abs().max())
            return match, err
        except Exception:
            return False, float("inf")
