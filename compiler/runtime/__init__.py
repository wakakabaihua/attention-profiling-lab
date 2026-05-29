"""compiler.runtime — 执行与 benchmark 模块。"""

from compiler.runtime.executor import Executor, ExecutionResult
from compiler.runtime.benchmark_runner import BenchmarkRunner, BenchmarkReport, BackendStats

__all__ = [
    "Executor", "ExecutionResult",
    "BenchmarkRunner", "BenchmarkReport", "BackendStats",
]
