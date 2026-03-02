"""
MLIR Attention Fusion 实验模块
==============================

使用 torch-mlir 将 PyTorch attention 子操作导出为 MLIR IR，
在 Torch dialect 和 Linalg dialect 两个层级分析融合机会，
并实现 Python 版 Attention Fusion Pass。

模块结构:
  export_attention_ir  — MLIR IR 导出与解析工具
  fusion_pass          — Attention 融合 Pass 实现（模式匹配 + 替换）
  run_mlir_experiment  — 端到端实验驱动脚本
"""
