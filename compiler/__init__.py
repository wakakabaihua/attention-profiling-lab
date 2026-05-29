"""
Mini AI Compiler Pipeline
==========================
面向 attention 子图的最小编译管线。

链路:
    PyTorch / FX Graph
      -> Frontend Import       (compiler/frontend/)
      -> Internal IR           (compiler/ir/)
      -> Canonicalize Pass     (compiler/passes/canonicalize.py)
      -> Fusion Pass           (compiler/passes/fusion.py)
      -> Validation Pass       (compiler/passes/validation.py)
      -> Lowering              (compiler/lowering/)
      -> Backend Execution     (compiler/backends/)
      -> Benchmark + Trace     (compiler/runtime/)

快速入口:
    from compiler.lowering.pipeline import CompilerPipeline
    pipeline = CompilerPipeline(backend="triton")
    result = pipeline.compile_and_run(scores)
"""
