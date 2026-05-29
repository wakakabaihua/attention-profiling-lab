"""
Attention 优化 Pass Pipeline
=============================

编排 MLIR 原生 Pass 管线，将多个变换按序组合。

管线 A（Torch dialect 融合）:
  1. AttentionFusionPass — 融合 scale + mask + softmax
  2. 内置 canonicalize / CSE — 常量折叠、公共子表达式消除

管线 B（增量计算重写）:
  1. torch-decompose-complex-ops — 分解 softmax.int
  2. OnlineSoftmaxPass — 匹配分解后的标准 softmax，重写为 online 2-pass
  3. 内置 canonicalize / CSE
"""

from torch_mlir import ir, rewrite, passmanager

from .attention_fusion_pass import create_attention_fusion_patterns
from .incremental_softmax_pass import (
    create_online_softmax_patterns,
    decompose_softmax,
)


def build_attention_optimization_pipeline(module: ir.Module) -> None:
    """
    在 MLIR Module 上运行完整的 attention 优化管线（管线 A：Torch 层融合）。

    Pipeline 步骤:
      1. AttentionFusionPass — 匹配 mul.Scalar→where.ScalarSelf→softmax.int，
         替换为 custom.fused_scaled_masked_softmax
      2. canonicalize + CSE — MLIR 内置优化 pass

    Args:
        module: torch-mlir 导出的 ir.Module（Torch dialect）

    Side Effect:
        module 被原地变换。
    """
    module.context.allow_unregistered_dialects = True

    # Step 1: 自定义融合 pattern
    frozen = create_attention_fusion_patterns(module.context)
    rewrite.walk_and_apply_patterns(module.operation, frozen)

    # Step 2: MLIR 内置优化 passes
    pm = passmanager.PassManager.parse(
        "builtin.module(canonicalize,cse)",
        context=module.context,
    )
    pm.run(module.operation)


def build_online_softmax_pipeline(module: ir.Module) -> None:
    """
    在 MLIR Module 上运行增量计算（online softmax）优化管线（管线 B）。

    Pipeline 步骤:
      1. torch-decompose-complex-ops — 将 softmax.int 分解为
         max.dim + sub + exp + sum.dim + div
      2. OnlineSoftmaxPass — 匹配分解后的 5-op 标准 softmax 模式，
         替换为 custom.online_softmax（2-pass 增量计算）
      3. canonicalize + CSE — MLIR 内置优化 pass

    算法转换:
      标准 3-pass: max → exp(x-max) → sum(exp) → exp/sum
      Online 2-pass: (max, sum) 同步计算 → exp(x-max)/sum

    Args:
        module: torch-mlir 导出的 ir.Module（Torch dialect）

    Side Effect:
        module 被原地变换。
    """
    module.context.allow_unregistered_dialects = True

    # Step 1: 分解 softmax 为组件操作
    decompose_softmax(module)

    # Step 2: 匹配分解后的标准 softmax 模式，重写为 online softmax
    frozen = create_online_softmax_patterns(module.context)
    try:
        rewrite.walk_and_apply_patterns(module.operation, frozen)
    except RuntimeError:
        pass  # 无匹配时可能触发，安全忽略

    # Step 3: MLIR 内置优化 passes
    pm = passmanager.PassManager.parse(
        "builtin.module(canonicalize,cse)",
        context=module.context,
    )
    pm.run(module.operation)
