"""
MLIR Attention Fusion 实验模块
==============================

使用 torch-mlir 将 PyTorch attention 子操作导出为 MLIR IR，
在 Torch dialect 和 Linalg dialect 两个层级分析融合机会，
并实现 MLIR 原生 Attention Fusion Pass。

模块结构:
  export_attention_ir  — MLIR IR 导出与解析工具
  fusion_pass          — v1 Attention 融合 Pass（文本匹配，保留作参考）
  passes/              — v2 MLIR 原生 Pass 实现（Pattern Rewrite 框架）
  run_mlir_experiment  — 端到端实验驱动脚本
"""
