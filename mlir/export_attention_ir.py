"""
Attention MLIR IR 导出与解析工具
================================

使用 torch-mlir 将 PyTorch attention 子操作导出为 MLIR IR，
支持 Torch dialect 和 Linalg on Tensors dialect 两个层级。

提供两个可导出模型:
  ScaleMaskSoftmax  — 仅融合目标区域 (scale + mask + softmax)
  FullAttention     — 完整 attention (QK^T + scale + mask + softmax + PV)

以及 IR 解析工具，将 MLIR IR 文本转换为结构化数据，供融合 Pass 使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch_mlir.fx import export_and_import


# =====================================================================
# 可导出的 Attention 模型
# =====================================================================

class ScaleMaskSoftmax(nn.Module):
    """
    仅包含 attention 中可融合的核心操作: scale + causal_mask + softmax。

    对应 ManualAttention.forward 中的步骤 2-4:
      scores = scores * scale           # 步骤 2: scale
      scores = scores.masked_fill(...)   # 步骤 3: causal mask
      probs = F.softmax(scores, dim=-1)  # 步骤 4: softmax

    这三个操作在 baseline 中是独立的 CUDA kernel，是融合的主要目标。
    """

    def __init__(self, head_dim: int = 64, seq_len: int = 128):
        super().__init__()
        self.scale = head_dim ** -0.5   # 0.125 when head_dim=64
        self.seq_len = seq_len

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        # 步骤 2: Scale（逐元素乘法 kernel）
        scores = scores * self.scale

        # 步骤 3: Causal Mask
        T = self.seq_len
        mask = torch.triu(
            torch.ones(T, T, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

        # 步骤 4: Softmax（归约 kernel）
        return F.softmax(scores, dim=-1)


class FullAttention(nn.Module):
    """
    完整 attention 计算: QK^T → scale → mask → softmax → PV matmul。

    导出后可在 MLIR 中清楚看到两个 matmul 之间的可融合区域。
    """

    def __init__(self, head_dim: int = 64, seq_len: int = 128):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.seq_len = seq_len

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # QK^T matmul（cublas kernel）
        scores = torch.matmul(q, k.transpose(-2, -1))

        # ---- 可融合区域开始 ----
        scores = scores * self.scale
        T = self.seq_len
        mask = torch.triu(
            torch.ones(T, T, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        # ---- 可融合区域结束 ----

        # PV matmul（cublas kernel）
        return torch.matmul(probs, v)


# =====================================================================
# MLIR 导出函数
# =====================================================================

def export_to_torch_dialect(model: nn.Module, *example_inputs):
    """
    将 PyTorch 模型导出为 Torch dialect MLIR Module。

    返回:
        MLIR Module 对象（Torch dialect）
    """
    return export_and_import(model, *example_inputs)


def export_to_linalg(model: nn.Module, *example_inputs):
    """
    将 PyTorch 模型导出为 Linalg on Tensors dialect MLIR Module。

    Linalg dialect 更接近底层硬件表示，每个 linalg.generic 操作
    对应一个潜在的 GPU kernel。

    返回:
        MLIR Module 对象（Linalg dialect），失败时返回 None
    """
    try:
        return export_and_import(
            model, *example_inputs, output_type="linalg-on-tensors"
        )
    except Exception as e:
        print(f"  ⚠️  Linalg 降级失败: {e}")
        return None


def get_ir_text(module) -> str:
    """获取 MLIR IR 的文本表示。"""
    return str(module)


def save_ir(ir_text: str, filepath: str):
    """保存 IR 文本到文件。"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ir_text, encoding="utf-8")


# =====================================================================
# Torch Dialect IR 解析
# =====================================================================

@dataclass
class IROperation:
    """解析后的 MLIR 操作。"""

    index: int                    # 操作序号
    name: str                     # 操作名称 (如 torch.aten.mul.Scalar)
    result_var: str               # 结果 SSA 变量 (如 %0)
    operand_vars: List[str]       # 操作数 SSA 变量列表
    full_line: str                # 完整 IR 行文本
    category: str = ""            # 语义分类标签


def parse_torch_ir(ir_text: str) -> List[IROperation]:
    """
    解析 Torch dialect IR 文本，提取所有操作的结构化信息。

    返回:
        IROperation 列表，每个元素包含操作名称、SSA 变量、分类等信息，
        可直接用于融合 Pass 的模式匹配。
    """
    ops: List[IROperation] = []
    idx = 0

    for line in ir_text.split("\n"):
        stripped = line.strip()

        # 跳过非操作行
        if (
            not stripped
            or stripped.startswith("//")
            or stripped.startswith("module")
            or stripped.startswith("func.func")
            or stripped.startswith("#map")
            or stripped in ("}", "{")
        ):
            continue

        # 匹配: %result = op_name operands : types
        m = re.match(r"(%\S+)\s*=\s*\"?([^\"(\s]+)\"?\s*(.*)", stripped)
        if m:
            result_var, op_name, rest = m.groups()

            # 提取操作数 SSA 变量（: 之前的部分）
            type_split = rest.split(" : ", 1)
            operands_part = type_split[0] if len(type_split) > 1 else rest
            operand_vars = re.findall(r"%[\w.\-]+", operands_part)

            ops.append(
                IROperation(
                    index=idx,
                    name=op_name,
                    result_var=result_var,
                    operand_vars=operand_vars,
                    full_line=stripped,
                    category=_categorize_torch_op(op_name),
                )
            )
            idx += 1

        elif stripped.startswith("return"):
            operand_vars = re.findall(r"%[\w.\-]+", stripped)
            ops.append(
                IROperation(
                    index=idx,
                    name="return",
                    result_var="",
                    operand_vars=operand_vars,
                    full_line=stripped,
                    category="control",
                )
            )
            idx += 1

    return ops


def _categorize_torch_op(name: str) -> str:
    """将 Torch dialect 操作名称映射到语义分类。"""
    n = name.lower()
    # 注意顺序: matmul 要在 mul 之前检查（因为 matmul 包含 mul）
    if "matmul" in n or n.endswith(".mm") or n.endswith(".bmm"):
        return "matmul"
    elif "mul" in n:
        return "scale"
    elif "where" in n or "masked_fill" in n:
        return "mask_apply"
    elif "softmax" in n:
        return "softmax"
    elif "constant" in n:
        return "constant"
    elif any(kw in n for kw in ("ones", "triu", "arange", "unsqueeze")):
        return "mask_gen"
    elif any(kw in n for kw in ("sub", "ge.", "logical", "cmpi")):
        return "mask_gen"
    elif "listconstruct" in n or "prim" in n:
        return "auxiliary"
    else:
        return "other"


# =====================================================================
# Linalg Dialect IR 解析
# =====================================================================

def parse_linalg_ir(ir_text: str) -> List[Dict]:
    """
    解析 Linalg dialect IR，提取每个 linalg.generic 操作的结构信息。

    重点识别每个 generic 操作内部的算术/数学操作（mulf, exp, divf 等），
    用于分析 linalg 层面的融合机会。

    返回:
        字典列表，每个字典包含:
          - inner_ops: 内部算术操作列表
          - iterator_types: 迭代器类型 (parallel/reduction)
          - category: 语义分类
    """
    generics: List[Dict] = []
    current: Optional[Dict] = None

    for line in ir_text.split("\n"):
        stripped = line.strip()

        # 检测 linalg.generic 或 linalg.fill 开始
        if "linalg.generic" in stripped:
            current = {
                "header": stripped[:80] + ("..." if len(stripped) > 80 else ""),
                "inner_ops": [],
                "iterator_types": [],
                "is_reduction": False,
            }
            it_match = re.search(r'iterator_types\s*=\s*\[(.*?)\]', stripped)
            if it_match:
                types = re.findall(r'"(\w+)"', it_match.group(1))
                current["iterator_types"] = types
                current["is_reduction"] = "reduction" in types
            generics.append(current)

        elif "linalg.fill" in stripped:
            current = {
                "header": stripped[:80],
                "inner_ops": ["linalg.fill"],
                "iterator_types": [],
                "is_reduction": False,
            }
            generics.append(current)
            current = None  # fill 是单行操作

        elif current is not None:
            # 收集 generic body 中的算术/数学操作
            for op_name in [
                "arith.mulf", "arith.addf", "arith.subf", "arith.divf",
                "arith.maximumf", "arith.cmpf", "arith.cmpi",
                "arith.select", "arith.index_cast",
                "math.exp", "math.erf",
                "linalg.index",
            ]:
                if op_name in stripped:
                    current["inner_ops"].append(op_name)

            if "linalg.yield" in stripped:
                current = None  # generic body 结束

    # 分类每个 generic
    for g in generics:
        g["category"] = _categorize_linalg_generic(g)

    return generics


def _categorize_linalg_generic(g: Dict) -> str:
    """根据内部操作和迭代器类型分类 linalg.generic。"""
    inner = set(g["inner_ops"])

    if "arith.mulf" in inner and "arith.divf" not in inner and "arith.maximumf" not in inner and "arith.select" not in inner:
        return "scale"
    elif "arith.maximumf" in inner:
        return "softmax_max"
    elif "arith.select" in inner:
        return "mask/where"
    elif "arith.subf" in inner and "math.exp" not in inner:
        return "softmax_sub"
    elif "math.exp" in inner:
        return "softmax_exp"
    elif "arith.addf" in inner and g.get("is_reduction"):
        return "softmax_sum"
    elif "arith.divf" in inner:
        return "softmax_div"
    elif "linalg.index" in inner or "arith.index_cast" in inner:
        return "indexing"
    elif "arith.cmpi" in inner:
        return "comparison"
    elif "linalg.fill" in inner:
        return "fill"
    else:
        return "other"


# =====================================================================
# 快速自测
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  📦 Attention MLIR IR 导出测试")
    print("=" * 60)

    model = ScaleMaskSoftmax(head_dim=64, seq_len=32)
    example = torch.randn(1, 12, 32, 32)

    module = export_to_torch_dialect(model, example)
    ir_text = get_ir_text(module)

    ops = parse_torch_ir(ir_text)
    print(f"\n  Torch dialect 操作清单（共 {len(ops)} 个）:\n")
    for op in ops:
        marker = "🟡" if op.category in ("scale", "mask_apply", "softmax") else "  "
        print(f"    {marker} [{op.index:2d}] {op.category:12s} │ {op.name}")

    print(f"\n  ✅ 解析完成")
