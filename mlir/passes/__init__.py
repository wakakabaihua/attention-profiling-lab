"""
MLIR 原生 Pass 实现
===================

使用 torch-mlir 提供的 MLIR Python bindings（非模拟）实现 Attention 优化 Pass。

核心 API 对应关系（Python bindings ↔ C++ MLIR）:
  rewrite.RewritePatternSet           ↔ mlir::RewritePatternSet
  rewrite.PatternRewriter.replace_op  ↔ mlir::PatternRewriter::replaceOp
  rewrite.walk_and_apply_patterns     ↔ mlir::applyPatternsAndFoldGreedily（walk 版）
  ir.Operation.create                 ↔ mlir::Operation::create

模块:
  attention_fusion_pass    — Phase 1: scale+mask+softmax 融合 Pattern
  incremental_softmax_pass — Phase 2: 标准 softmax → online softmax 重写
  pass_pipeline            — Pass 管线编排
"""

from .attention_fusion_pass import (
    attention_fusion_pattern,
    create_attention_fusion_patterns,
    run_attention_fusion_pass,
)
from .incremental_softmax_pass import (
    online_softmax_rewrite,
    create_online_softmax_patterns,
    decompose_softmax,
    run_online_softmax_pass,
)
from .pass_pipeline import (
    build_attention_optimization_pipeline,
    build_online_softmax_pipeline,
)
