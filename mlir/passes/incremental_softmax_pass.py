"""
增量计算（Online Softmax）重写 Pass — MLIR 原生实现
===================================================

在分解后的 Torch dialect IR 上，匹配标准 3-pass softmax 模式：
  %max, _ = torch.aten.max.dim         %x, dim, keepdim=true
  %sub    = torch.aten.sub.Tensor       %x, %max, 1.0
  %exp    = torch.aten.exp              %sub
  %sum    = torch.aten.sum.dim_IntList  %exp, [dim], keepdim=true, none
  %div    = torch.aten.div.Tensor       %exp, %sum

重写为 2-pass online softmax：
  %result = "custom.online_softmax"(%x) { dim = -1 }

算法转换：
  标准 softmax（3 次全局扫描）:
    Pass 1: max = reduce_max(x)
    Pass 2: exp_vals = exp(x - max), sum = reduce_sum(exp_vals)
    Pass 3: out = exp_vals / sum

  Online softmax（2 次全局扫描，增量/流式）:
    Pass 1: 同时计算 running_max 和 running_sum（单次扫描）
            max_new = max(max_old, block_max)
            sum_new = sum_old * exp(max_old - max_new) + block_exp_sum
    Pass 2: out = exp(x - final_max) / final_sum

使用方式：
  1. 先用 torch-decompose-complex-ops 将 softmax.int 分解为 5 个组件操作
  2. 再运行本 pass 匹配分解后的模式并重写

API 对应关系（Python bindings ↔ C++ MLIR）:
  rewrite.RewritePatternSet.add()       ↔ mlir::RewritePatternSet::add<Pattern>
  rewrite.PatternRewriter.replace_op()  ↔ mlir::PatternRewriter::replaceOp()
  ir.OpResult.result_number             ↔ mlir::OpResult::getResultNumber()
"""

from torch_mlir import ir, rewrite, passmanager


# ---------------------------------------------------------------------------
# Pattern callback
# ---------------------------------------------------------------------------

def online_softmax_rewrite(
    div_op: ir.Operation,
    rewriter: rewrite.PatternRewriter,
):
    """
    从 torch.aten.div.Tensor 反向匹配标准 softmax 的 5-op 分解模式。

    匹配路径（反向 def-use chain）:
      div.operands[0].owner → exp
      div.operands[1].owner → sum.dim_IntList
      exp.operands[0].owner → sub.Tensor
      sum.operands[0].owner → exp           (同一个 exp)
      sub.operands[0]       → input_tensor  (原始输入)
      sub.operands[1].owner → max.dim
      max.operands[0]       → input_tensor  (同一个输入，验证一致性)

    返回 None 表示匹配成功，返回非 None 表示不匹配。
    """
    with div_op.context, ir.Location.unknown(context=div_op.context):

        # ---- Step 1: div 的两个操作数 → exp 和 sum ----
        exp_val = div_op.operands[0]
        sum_val = div_op.operands[1]

        if not isinstance(exp_val, ir.OpResult) or not isinstance(sum_val, ir.OpResult):
            return "div operands not OpResult"

        exp_op = exp_val.owner
        sum_op = sum_val.owner

        if exp_op.name != "torch.aten.exp":
            return "div.operands[0] not from exp"
        if sum_op.name != "torch.aten.sum.dim_IntList":
            return "div.operands[1] not from sum.dim_IntList"

        # ---- Step 2: sum 的输入应为同一个 exp 的输出 ----
        sum_input = sum_op.operands[0]
        if sum_input != exp_val:
            return "sum input not from same exp output"

        # ---- Step 3: exp 的输入 → sub.Tensor ----
        sub_val = exp_op.operands[0]
        if not isinstance(sub_val, ir.OpResult):
            return "exp input not OpResult"
        sub_op = sub_val.owner
        if sub_op.name != "torch.aten.sub.Tensor":
            return "exp input not from sub.Tensor"

        # ---- Step 4: sub 的第二个操作数 → max.dim ----
        max_val = sub_op.operands[1]
        if not isinstance(max_val, ir.OpResult):
            return "sub.operands[1] not OpResult"
        max_op = max_val.owner
        if max_op.name != "torch.aten.max.dim":
            return "sub.operands[1] not from max.dim"
        # max.dim 返回 (values, indices)，sub 使用的应为 result[0]
        if max_val.result_number != 0:
            return "sub uses max.dim indices instead of values"

        # ---- Step 5: 一致性验证 —— sub 和 max 的输入应为同一个 tensor ----
        softmax_input = sub_op.operands[0]
        max_input = max_op.operands[0]
        if softmax_input != max_input:
            return "sub and max operate on different inputs"

        # ---- Step 6: 提取 reduction dim ----
        dim_int = _extract_int_const(max_op.operands[1], default=-1)

        # ---- Step 7: 创建 online softmax 融合操作 ----
        ip = rewriter.ip
        online_op = ir.Operation.create(
            "custom.online_softmax",
            results=[div_op.results[0].type],
            operands=[softmax_input],
            attributes={
                "dim": ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(64), dim_int
                ),
                "algorithm": ir.StringAttr.get("online_2pass"),
            },
            ip=ip,
        )

        # ---- Step 8: 替换 div（最终输出） ----
        rewriter.replace_op(div_op, list(online_op.results))
        return None  # 匹配成功


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

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

def create_online_softmax_patterns(
    ctx: ir.Context,
) -> rewrite.FrozenRewritePatternSet:
    """创建并冻结 online softmax 重写 pattern 集合。"""
    patterns = rewrite.RewritePatternSet(ctx)
    patterns.add(
        "torch.aten.div.Tensor",
        online_softmax_rewrite,
        benefit=10,
    )
    return patterns.freeze()


def decompose_softmax(module: ir.Module) -> None:
    """
    使用 torch-mlir 内置 pass 将 torch.aten.softmax.int 分解为组件操作。

    分解结果:
      softmax.int → max.dim + sub.Tensor + exp + sum.dim_IntList + div.Tensor
    """
    pm = passmanager.PassManager.parse(
        "builtin.module(func.func(torch-decompose-complex-ops))",
        context=module.context,
    )
    pm.run(module.operation)


def run_online_softmax_pass(module: ir.Module) -> bool:
    """
    在 MLIR Module 上运行 online softmax 重写 pass。

    完整流程:
      1. 分解 softmax.int 为 max + sub + exp + sum + div
      2. 匹配分解后的 5-op 标准 softmax 模式
      3. 替换为 custom.online_softmax（2-pass 增量计算）

    Args:
        module: torch-mlir 导出的 ir.Module（Torch dialect）

    Returns:
        True 如果 pass 成功执行，False 如果发生内部错误。

    Side Effect:
        module 中的标准 softmax 子图被原地替换为
        custom.online_softmax 操作。
    """
    module.context.allow_unregistered_dialects = True

    # Step 1: 分解 softmax
    decompose_softmax(module)

    # Step 2: 匹配分解后的模式并重写
    frozen = create_online_softmax_patterns(module.context)
    try:
        rewrite.walk_and_apply_patterns(module.operation, frozen)
    except RuntimeError:
        return False
    return True
