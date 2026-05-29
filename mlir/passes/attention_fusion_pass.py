"""
Attention 融合 Pattern — MLIR 原生实现
======================================

使用 MLIR Pattern Rewrite 框架，在 Torch dialect IR 上
匹配 mul.Scalar → where.ScalarSelf → softmax.int 子图，
替换为 custom.fused_scaled_masked_softmax 融合操作。

核心 API（MLIR Python bindings ↔ C++ MLIR）:
  rewrite.RewritePatternSet.add()       ↔ mlir::RewritePatternSet::add<Pattern>
  rewrite.PatternRewriter.replace_op()  ↔ mlir::PatternRewriter::replaceOp()
  rewrite.PatternRewriter.ip            ↔ mlir::PatternRewriter::getInsertionPoint()
  rewrite.walk_and_apply_patterns()     ↔ walk + applyPatternsAndFoldGreedily
  ir.Operation.create()                 ↔ mlir::Operation::create
  operand.owner                         ↔ mlir::Value::getDefiningOp()

匹配的 Torch dialect 操作链:
  %scaled = torch.aten.mul.Scalar    %scores, %scale_const
  %masked = torch.aten.where.ScalarSelf %causal_mask, %neg_inf, %scaled
  %probs  = torch.aten.softmax.int   %masked, %dim, %none

替换为:
  %probs  = "custom.fused_scaled_masked_softmax"(%scores)
              { scale = 0.125, softmax_dim = -1, is_causal = true, algorithm = "online" }
"""

from torch_mlir import ir, rewrite


# ---------------------------------------------------------------------------
# Pattern callback
# ---------------------------------------------------------------------------

def attention_fusion_pattern(
    softmax_op: ir.Operation,
    rewriter: rewrite.PatternRewriter,
):
    """
    从 torch.aten.softmax.int 反向匹配 scale+mask+softmax 子图。

    匹配路径（反向 def-use chain 追溯）:
      softmax.operands[0].owner → torch.aten.where.ScalarSelf
      where.operands[2].owner   → torch.aten.mul.Scalar
      mul.operands[0]           → 原始 scores（BlockArgument 或上游 OpResult）

    返回 None 表示匹配成功（MLIR 约定），返回非 None 表示不匹配。
    """
    # 进入 MLIR Context —— 在 callback 内创建 Type / Attr 时需要
    with softmax_op.context, ir.Location.unknown(context=softmax_op.context):

        # ---- Step 1: softmax 输入应来自 where.ScalarSelf ----
        masked_value = softmax_op.operands[0]
        if not isinstance(masked_value, ir.OpResult):
            return "softmax input is not an OpResult"
        where_op = masked_value.owner
        if where_op.name != "torch.aten.where.ScalarSelf":
            return "softmax input not from where.ScalarSelf"

        # ---- Step 2: where 的第三个操作数应来自 mul.Scalar ----
        scaled_value = where_op.operands[2]
        if not isinstance(scaled_value, ir.OpResult):
            return "where other-operand is not an OpResult"
        scale_op = scaled_value.owner
        if scale_op.name != "torch.aten.mul.Scalar":
            return "where other-operand not from mul.Scalar"

        # ---- Step 3: 提取参数 ----
        scores_value = scale_op.operands[0]  # 原始 %scores

        # 提取 scale 常量值
        scale_float = _extract_float_const(scale_op.operands[1])

        # 提取 softmax dim
        dim_int = _extract_int_const(softmax_op.operands[1], default=-1)

        # ---- Step 4: 创建融合操作 ----
        ip = rewriter.ip
        fused_op = ir.Operation.create(
            "custom.fused_scaled_masked_softmax",
            results=[softmax_op.results[0].type],
            operands=[scores_value],
            attributes={
                "scale": ir.FloatAttr.get(ir.F64Type.get(), scale_float),
                "softmax_dim": ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(64), dim_int
                ),
                "is_causal": ir.BoolAttr.get(True),
                "algorithm": ir.StringAttr.get("online"),
            },
            ip=ip,
        )

        # ---- Step 5: 替换 —— MLIR 自动重连所有 use sites ----
        rewriter.replace_op(softmax_op, list(fused_op.results))
        return None  # None = 匹配成功


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _extract_float_const(value: ir.Value, *, default: float = 1.0) -> float:
    """从 torch.constant.float 操作中提取浮点值。"""
    if not isinstance(value, ir.OpResult):
        return default
    op = value.owner
    if op.name != "torch.constant.float":
        return default
    attrs = dict(op.attributes)
    if "value" not in attrs:
        return default
    return float(ir.FloatAttr(attrs["value"]).value)


def _extract_int_const(value: ir.Value, *, default: int = -1) -> int:
    """从 torch.constant.int 操作中提取整数值。"""
    if not isinstance(value, ir.OpResult):
        return default
    op = value.owner
    if op.name != "torch.constant.int":
        return default
    attrs = dict(op.attributes)
    if "value" not in attrs:
        return default
    return int(ir.IntegerAttr(attrs["value"]))


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def create_attention_fusion_patterns(
    ctx: ir.Context,
) -> rewrite.FrozenRewritePatternSet:
    """创建并冻结 attention 融合 pattern 集合。"""
    patterns = rewrite.RewritePatternSet(ctx)
    patterns.add(
        "torch.aten.softmax.int",
        attention_fusion_pattern,
        benefit=10,
    )
    return patterns.freeze()


def run_attention_fusion_pass(module: ir.Module) -> bool:
    """
    在 MLIR Module 上运行 attention 融合 pass。

    Args:
        module: torch-mlir 导出的 ir.Module（Torch dialect）

    Returns:
        True 如果 pass 成功执行（无论是否有匹配），False 如果发生内部错误。

    Side Effect:
        module 中的 softmax→where→mul 子图被原地替换为
        custom.fused_scaled_masked_softmax 融合操作。
    """
    module.context.allow_unregistered_dialects = True
    frozen = create_attention_fusion_patterns(module.context)
    try:
        rewrite.walk_and_apply_patterns(module.operation, frozen)
    except RuntimeError:
        # walk_and_apply_patterns 在某些 IR 结构上会触发 std::bad_cast
        # （C++ binding 已知问题），当所有 pattern 均未匹配时可能发生。
        # 此时 IR 未被修改，可安全忽略。
        return False
    return True
