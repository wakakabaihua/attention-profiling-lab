"""
Passes — Validation Pass
==========================
检查 IRGraph 的合法性。

检查项:
    1. 无环（通过拓扑排序验证）
    2. 无悬空引用（每个 input 名称都是已知节点）
    3. 算子输入数匹配 OpSpec 要求
    4. FUSED_SCALE_MASK_SOFTMAX 属性完整性
    5. 图中存在至少一个 OUTPUT 节点

验证通过时返回 ValidationResult(ok=True)，
验证失败时返回 ValidationResult(ok=False, errors=[...])。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from compiler.ir.graph import IRGraph

from compiler.ir.ops import OpType, OP_REGISTRY


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.ok:
            return f"ValidationResult(ok=True, warnings={len(self.warnings)})"
        return f"ValidationResult(ok=False, errors={self.errors})"

    def raise_if_invalid(self) -> None:
        """若验证失败则抛出 ValueError，包含所有错误信息。"""
        if not self.ok:
            msg = "IR validation failed:\n" + "\n".join(f"  - {e}" for e in self.errors)
            raise ValueError(msg)


class ValidationPass:
    """
    图合法性检查 pass。

    Usage:
        result = ValidationPass().run(graph)
        result.raise_if_invalid()
    """

    def run(self, graph: "IRGraph") -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        # 1. 无环检查
        try:
            graph.topological_sort()
        except ValueError as e:
            errors.append(f"Cycle detected: {e}")

        # 2. 存在至少一个 OUTPUT 节点
        if not graph.get_output_nodes():
            warnings.append("Graph has no OUTPUT node; may be an intermediate representation.")

        node_names = {n.name for n in graph.nodes}

        for node in graph.nodes:
            # 3. 悬空引用检查
            for inp in node.inputs:
                if inp not in node_names:
                    errors.append(
                        f"Node '{node.name}' references unknown input '{inp}'"
                    )

            # 4. 算子输入数检查
            spec = OP_REGISTRY.get(node.op_type)
            if spec is not None and spec.num_inputs != -1:
                # INPUT 节点允许多余输入（FX 有时生成占位 INPUT 包含多引用）
                if node.op_type not in (OpType.INPUT, OpType.OUTPUT):
                    if len(node.inputs) != spec.num_inputs:
                        errors.append(
                            f"Node '{node.name}' ({node.op_type.name}) expects "
                            f"{spec.num_inputs} inputs but got {len(node.inputs)}"
                        )

            # 5. FUSED_SCALE_MASK_SOFTMAX 属性完整性
            if node.op_type == OpType.FUSED_SCALE_MASK_SOFTMAX:
                for required_attr in ("scale_factor", "is_causal", "softmax_dim"):
                    if required_attr not in node.attrs:
                        errors.append(
                            f"FUSED_SCALE_MASK_SOFTMAX node '{node.name}' "
                            f"missing required attribute '{required_attr}'"
                        )

        return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)
