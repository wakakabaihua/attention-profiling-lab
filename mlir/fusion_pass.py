"""
Attention Fusion Pass（Python MLIR 实现）
=========================================

在 MLIR Torch dialect IR 上实现 attention 融合 Pass。

核心逻辑（与 C++ MLIR RewritePattern 对应）:

1. 模式匹配 (Pattern Matching)
   从 softmax 操作反向追溯 SSA 定义链，识别 scale → mask → softmax 子图

2. 子图验证 (Subgraph Validation)
   确认操作之间通过 SSA 值直接连接（def-use chain）

3. 操作替换 (Operation Replacement)
   将匹配到的子图替换为单一 custom.fused_scaled_masked_softmax 操作

4. 死代码消除 (Dead Code Elimination)
   被融合的中间操作和辅助操作标记为死代码

本模块是对 C++ MLIR ``OpRewritePattern`` 的 Python 教学实现。
真正的编译器 pass 会使用 MLIR 的 C++ RewritePatternSet + PatternRewriter，
但核心算法（反向追溯 + 模式匹配 + 替换）完全相同。
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set


# =====================================================================
# 数据结构
# =====================================================================

@dataclass
class FusionCandidate:
    """
    表示一个可融合的 attention 子图。

    由融合 Pass 的模式匹配阶段产生，包含:
    - 核心操作 (scale, mask, softmax)
    - 辅助操作 (mask 生成、常量)
    - 数据流信息 (输入/输出 SSA 变量)
    """

    # 核心操作（融合目标）
    scale_op: Dict             # mul.Scalar 操作
    mask_op: Dict              # where.ScalarSelf / masked_fill 操作
    softmax_op: Dict           # softmax.int 操作

    # 辅助操作（死代码消除目标）
    auxiliary_ops: List[Dict] = field(default_factory=list)

    # 数据流
    scores_input: str = ""     # 原始输入 SSA 变量
    probs_output: str = ""     # 最终输出 SSA 变量
    scale_value: str = ""      # 缩放常量值

    # 统计
    total_ops_fused: int = 0   # 被消除的操作总数


# =====================================================================
# Attention Fusion Pass
# =====================================================================

class AttentionFusionPass:
    """
    Attention 融合 Pass。

    模式匹配规则（Torch dialect）::

        %scaled = torch.aten.mul.Scalar   %scores, %scale        ← SCALE
        ... (mask generation: ones, arange, triu, ...) ...
        %masked = torch.aten.where.ScalarSelf %mask, -inf, %scaled ← MASK
        %probs  = torch.aten.softmax.int  %masked, %dim, %none    ← SOFTMAX

                            ↓↓↓  融合  ↓↓↓

        %probs = "custom.fused_scaled_masked_softmax"(%scores, %scale) {
            softmax_dim = -1, is_causal = true
        }

    对应 C++ 实现中的 ``AttentionFusionPattern : OpRewritePattern<...>``
    """

    def __init__(self):
        self.candidates: List[FusionCandidate] = []
        self._op_map: Dict[str, object] = {}  # result_var → IROperation

    # -----------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------

    def run(self, ir_text: str, ops: list) -> List[FusionCandidate]:
        """
        在 IR 上运行融合 Pass，返回所有找到的融合候选。

        Args:
            ir_text: MLIR IR 全文
            ops:     解析后的 IROperation 列表（来自 export_attention_ir.parse_torch_ir）
        """
        # 建立 SSA result_var → op 映射（模拟 MLIR 的 Value→DefiningOp）
        self._op_map = {op.result_var: op for op in ops if op.result_var}

        candidates: List[FusionCandidate] = []

        # 模式入口: 从 softmax 操作反向追溯
        softmax_ops = [op for op in ops if op.category == "softmax"]

        for softmax in softmax_ops:
            candidate = self._match_from_softmax(softmax, ops)
            if candidate:
                candidates.append(candidate)

        self.candidates = candidates
        return candidates

    def generate_fused_ir(self, ir_text: str, candidate: FusionCandidate) -> str:
        """
        生成融合后的 IR 文本。

        将 scale + mask_gen + where + softmax 替换为
        单一 ``custom.fused_scaled_masked_softmax`` 操作。
        """
        # 收集所有要移除的行
        lines_to_remove: Set[str] = set()
        lines_to_remove.add(candidate.scale_op["line"])
        lines_to_remove.add(candidate.mask_op["line"])
        lines_to_remove.add(candidate.softmax_op["line"])
        for aux in candidate.auxiliary_ops:
            lines_to_remove.add(aux["line"])

        lines = ir_text.split("\n")
        new_lines: List[str] = []
        fused_inserted = False

        for line in lines:
            stripped = line.strip()

            if stripped in lines_to_remove:
                # 在 scale 操作的位置插入融合操作
                if not fused_inserted and stripped == candidate.scale_op["line"]:
                    indent = len(line) - len(line.lstrip())
                    sp = " " * indent

                    # 推断类型
                    in_type = self._infer_type(candidate.scores_input, ir_text)
                    out_type = self._infer_type(candidate.probs_output, ir_text)

                    new_lines.append(
                        f"{sp}// ===== AttentionFusionPass: "
                        f"融合 scale + causal_mask + softmax ====="
                    )
                    new_lines.append(
                        f"{sp}// 原始: {candidate.scale_op['name']} → "
                        f"{candidate.mask_op['name']} → "
                        f"{candidate.softmax_op['name']}"
                    )
                    new_lines.append(
                        f"{sp}// 消除 {candidate.total_ops_fused} 个操作 → "
                        f"1 个融合操作"
                    )
                    new_lines.append(
                        f'{sp}{candidate.probs_output} = '
                        f'"custom.fused_scaled_masked_softmax"'
                        f"({candidate.scores_input}, "
                        f"{candidate.scale_value}) {{"
                    )
                    new_lines.append(f"{sp}    softmax_dim = -1 : i64,")
                    new_lines.append(f"{sp}    is_causal = true,")
                    new_lines.append(
                        f'{sp}    fusion_source = "attention_fusion_pass_v1"'
                    )
                    new_lines.append(
                        f"{sp}}} : ({in_type}, f32) -> {out_type}"
                    )
                    fused_inserted = True
                # else: 跳过此行（死代码消除）
            else:
                new_lines.append(line)

        return "\n".join(new_lines)

    # -----------------------------------------------------------------
    # 模式匹配（内部方法）
    # -----------------------------------------------------------------

    def _match_from_softmax(self, softmax_op, all_ops) -> Optional[FusionCandidate]:
        """
        从 softmax 操作反向追溯，匹配 scale → mask → softmax 模式。

        对应 C++ 中的 matchAndRewrite() 方法:
        1. 从 softmax（pattern 的最后一个操作）开始
        2. 追溯 softmax 的输入 → 应为 where/masked_fill
        3. 追溯 where 的 tensor 输入 → 应为 mul.Scalar (scale)
        4. 验证整个子图的连接性
        """
        if not softmax_op.operand_vars:
            return None

        # ---- Step 1: softmax 的输入应来自 where/masked_fill ----
        softmax_input = softmax_op.operand_vars[0]
        mask_op = self._op_map.get(softmax_input)
        if not mask_op or mask_op.category != "mask_apply":
            return None

        # ---- Step 2: where.ScalarSelf 的 tensor 输入应来自 scale ----
        # where.ScalarSelf(%condition, %scalar_self, %other_tensor)
        #   %condition = mask (bool tensor)
        #   %scalar_self = -inf (fill value)
        #   %other_tensor = scaled scores → 来自 mul.Scalar
        if len(mask_op.operand_vars) < 3:
            return None

        scaled_var = mask_op.operand_vars[2]  # other_tensor (scaled scores)
        scale_op = self._op_map.get(scaled_var)

        if not scale_op or scale_op.category != "scale":
            return None

        # ---- Step 3: 提取参数 ----
        if not scale_op.operand_vars:
            return None

        scores_input = scale_op.operand_vars[0]  # 原始 scores
        scale_const_var = (
            scale_op.operand_vars[1] if len(scale_op.operand_vars) > 1 else "?"
        )
        scale_value = self._get_constant_value(scale_const_var)

        # ---- Step 4: 追溯辅助操作（mask 生成链） ----
        core_vars = {
            softmax_op.result_var,
            mask_op.result_var,
            scale_op.result_var,
        }
        auxiliary_ops: List[object] = []

        # mask condition 的生产者链
        mask_cond_var = mask_op.operand_vars[0]
        auxiliary_ops.extend(
            self._trace_producers(mask_cond_var, exclude=core_vars)
        )

        # -inf 常量
        neg_inf_var = mask_op.operand_vars[1] if len(mask_op.operand_vars) > 1 else None
        if neg_inf_var:
            neg_inf_op = self._op_map.get(neg_inf_var)
            if neg_inf_op and neg_inf_op not in auxiliary_ops:
                auxiliary_ops.append(neg_inf_op)

        # softmax 的常量操作数 (dim, none)
        for var in softmax_op.operand_vars[1:]:
            const_op = self._op_map.get(var)
            if (
                const_op
                and const_op.category == "constant"
                and const_op not in auxiliary_ops
            ):
                auxiliary_ops.append(const_op)

        # scale 的常量操作数
        if scale_const_var != "?":
            sc_op = self._op_map.get(scale_const_var)
            if sc_op and sc_op not in auxiliary_ops:
                auxiliary_ops.append(sc_op)

        # ---- Step 5: 构建 FusionCandidate ----
        return FusionCandidate(
            scale_op=_op_to_dict(scale_op),
            mask_op=_op_to_dict(mask_op),
            softmax_op=_op_to_dict(softmax_op),
            auxiliary_ops=[_op_to_dict(a) for a in auxiliary_ops],
            scores_input=scores_input,
            probs_output=softmax_op.result_var,
            scale_value=scale_value,
            total_ops_fused=3 + len(auxiliary_ops),
        )

    def _get_constant_value(self, var: str) -> str:
        """提取常量变量的数值。"""
        op = self._op_map.get(var)
        if not op:
            return "?"
        m = re.search(r"(-?\d+\.?\d*(?:e[+-]?\d+)?)", op.full_line)
        if m:
            return m.group(1)
        return "?"

    def _trace_producers(
        self, var: str, exclude: Set[str], visited: Optional[Set[str]] = None
    ) -> List:
        """
        递归追溯变量的所有生产者操作。

        类似于 MLIR 中沿 def-use chain 的反向遍历。
        """
        if visited is None:
            visited = set()
        if var in visited or var in exclude:
            return []
        visited.add(var)

        result = []
        op = self._op_map.get(var)
        if op:
            result.append(op)
            for operand_var in op.operand_vars:
                result.extend(
                    self._trace_producers(operand_var, exclude, visited)
                )
        return result

    def _infer_type(self, var: str, ir_text: str) -> str:
        """从 IR 文本推断 SSA 变量的类型。"""
        # 函数参数类型
        if var.startswith("%arg"):
            m = re.search(re.escape(var) + r":\s*(\S+)", ir_text)
            if m:
                return m.group(1).rstrip(",)")

        # 操作结果类型（-> type 部分）
        op = self._op_map.get(var)
        if op:
            m = re.search(r"->\s*(\S+)\s*$", op.full_line)
            if m:
                return m.group(1)

        return "!torch.vtensor<[?,?,?,?],f32>"


# =====================================================================
# Linalg 层面融合分析
# =====================================================================

def analyze_linalg_fusion(linalg_ops: List[Dict]) -> Dict:
    """
    分析 Linalg dialect 的融合机会。

    在 linalg 层面，attention 的可融合操作包括:
      scale (mulf) + mask (select) + softmax_max + softmax_sub
      + softmax_exp + softmax_sum + softmax_div
    = 7 个 linalg.generic → 1 个融合 kernel

    返回:
        包含 fusible/non-fusible 操作数和分类信息的字典
    """
    FUSIBLE_CATEGORIES = {
        "scale", "mask/where",
        "softmax_max", "softmax_sub", "softmax_exp",
        "softmax_sum", "softmax_div",
    }

    fusible = [g for g in linalg_ops if g.get("category") in FUSIBLE_CATEGORIES]
    non_fusible = [g for g in linalg_ops if g.get("category") not in FUSIBLE_CATEGORIES]

    return {
        "total_generics": len(linalg_ops),
        "fusible_count": len(fusible),
        "non_fusible_count": len(non_fusible),
        "fusible_ops": fusible,
        "non_fusible_ops": non_fusible,
        "categories": sorted({g.get("category", "unknown") for g in linalg_ops}),
    }


# =====================================================================
# 辅助函数
# =====================================================================

def _op_to_dict(op) -> Dict:
    """将 IROperation 转换为普通字典（用于序列化）。"""
    return {
        "name": op.name,
        "result_var": op.result_var,
        "index": op.index,
        "line": op.full_line,
        "category": op.category,
    }
