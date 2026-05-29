"""
Internal IR — Op Definitions
==============================
定义 Mini AI Compiler Pipeline 的最小算子集合。

仅覆盖 attention 实验涉及的核心算子:
    INPUT / OUTPUT            — 图的边界占位符
    MATMUL                    — Q @ K^T 矩阵乘法
    SCALE                     — 逐元素缩放
    MASK                      — 因果遮罩
    SOFTMAX                   — Softmax 归约
    FUSED_SCALE_MASK_SOFTMAX  — 融合 pass 产生的融合算子
    ATTENTION_SCORE           — 完整 attention score 子图

每个 OpType 关联一个 OpSpec，描述期望输入/输出数量和含义，
供 validation pass 检查图合法性。
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict


class OpType(Enum):
    """所有已知内部算子类型。"""

    # 图边界
    INPUT = auto()
    OUTPUT = auto()

    # 基础算子
    MATMUL = auto()
    SCALE = auto()
    MASK = auto()
    SOFTMAX = auto()

    # 融合算子（fusion pass 产生）
    FUSED_SCALE_MASK_SOFTMAX = auto()

    # 完整 attention score 子图（可选扩展）
    ATTENTION_SCORE = auto()


@dataclass(frozen=True)
class OpSpec:
    """算子规格说明：期望输入/输出数量与含义描述。"""

    op_type: OpType
    num_inputs: int    # 期望输入数；-1 表示可变
    num_outputs: int   # 期望输出数
    description: str


# ─────────────────────────────────────────────────────────────────────
# 算子注册表
# ─────────────────────────────────────────────────────────────────────

OP_REGISTRY: Dict[OpType, OpSpec] = {
    OpType.INPUT: OpSpec(
        OpType.INPUT, 0, 1,
        "图输入占位符，无输入，输出为外部张量"
    ),
    OpType.OUTPUT: OpSpec(
        OpType.OUTPUT, 1, 0,
        "图输出占位符，接受最终结果张量"
    ),
    OpType.MATMUL: OpSpec(
        OpType.MATMUL, 2, 1,
        "矩阵乘法 Q @ K^T，输入为 (Q, K)，输出为 attention scores"
    ),
    OpType.SCALE: OpSpec(
        OpType.SCALE, 1, 1,
        "逐元素缩放 x * scale_factor，属性: scale_factor (float)"
    ),
    OpType.MASK: OpSpec(
        OpType.MASK, 1, 1,
        "因果遮罩 masked_fill(upper_tri, -inf)，属性: is_causal (bool), mask_value (float)"
    ),
    OpType.SOFTMAX: OpSpec(
        OpType.SOFTMAX, 1, 1,
        "Softmax 归约，属性: dim (int，默认 -1)"
    ),
    OpType.FUSED_SCALE_MASK_SOFTMAX: OpSpec(
        OpType.FUSED_SCALE_MASK_SOFTMAX, 1, 1,
        "融合 scale + causal_mask + softmax，属性: scale_factor, is_causal, softmax_dim"
    ),
    OpType.ATTENTION_SCORE: OpSpec(
        OpType.ATTENTION_SCORE, 3, 1,
        "完整 attention score: Q @ K^T -> scale -> mask -> softmax，输入: (Q, K, V)"
    ),
}


def get_spec(op_type: OpType) -> OpSpec:
    """获取算子规格；遇到未知类型时抛出 KeyError。"""
    if op_type not in OP_REGISTRY:
        raise KeyError(f"Unknown op type: {op_type!r}. Register it in OP_REGISTRY first.")
    return OP_REGISTRY[op_type]
