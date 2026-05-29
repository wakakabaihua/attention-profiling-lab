"""
Backends — Reference Backend
==============================
使用 PyTorch eager 模式执行 IRGraph。

职责:
    - 将图中每个节点对应的操作映射到 PyTorch 原生 op 并执行
    - FusedScaleMaskSoftmax 节点用 PyTorch 逐步执行（scale + mask + softmax）
    - 作为 correctness 基准，验证其他 backend 的数值一致性

这是最简单也最可靠的 backend，不依赖 Triton 或 MLIR。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if TYPE_CHECKING:
    from compiler.lowering.pipeline import CompilationArtifact

from compiler.ir.ops import OpType
from compiler.ir.graph import IRGraph, IRNode


class ReferenceBackend:
    """
    PyTorch eager reference backend。

    execute() 接受 CompilationArtifact 和输入 tensor，
    沿拓扑顺序逐个节点执行并返回最终输出。
    """

    def execute(
        self,
        artifact: "CompilationArtifact",
        *inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行 fused_ir 图（reference 语义：fused 节点用 PyTorch 展开执行）。

        Args:
            artifact: CompilationArtifact（包含 fused_ir 和 backend 选择）
            *inputs:  与图 INPUT 节点一一对应的输入 tensor

        Returns:
            input 节点中的输出 tensor
        """
        return self.run_graph(artifact.fused_ir, *inputs)

    def run_graph(self, graph: IRGraph, *inputs: torch.Tensor) -> torch.Tensor:
        """直接在 IRGraph 上执行（不需要 artifact）。"""
        input_nodes = graph.get_input_nodes()
        if len(inputs) != len(input_nodes):
            raise ValueError(
                f"Graph expects {len(input_nodes)} inputs but got {len(inputs)}"
            )

        # 构建 value map：node_name -> tensor
        value_map: Dict[str, torch.Tensor] = {}
        for node, tensor in zip(input_nodes, inputs):
            value_map[node.name] = tensor

        ordered = graph.topological_sort()
        for node in ordered:
            if node.op_type == OpType.INPUT:
                continue  # 已填充
            if node.op_type == OpType.OUTPUT:
                # 最终输出：取第一个输入的 tensor
                if node.inputs and node.inputs[0] in value_map:
                    value_map[node.name] = value_map[node.inputs[0]]
                continue

            result = self._execute_node(node, value_map)
            if result is not None:
                value_map[node.name] = result

        # 返回最后一个 OUTPUT 节点的值（或最后一个非 OUTPUT 节点）
        output_nodes = graph.get_output_nodes()
        if output_nodes:
            last_output = output_nodes[-1]
            if last_output.name in value_map:
                return value_map[last_output.name]
            if last_output.inputs and last_output.inputs[0] in value_map:
                return value_map[last_output.inputs[0]]
        # fallback: 返回最后计算的 tensor
        return list(value_map.values())[-1]

    # ──────────────────────────────────────────────────
    # 节点执行
    # ──────────────────────────────────────────────────

    def _execute_node(
        self,
        node: IRNode,
        value_map: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        def get(name: str) -> torch.Tensor:
            if name not in value_map:
                raise KeyError(f"Value '{name}' not found in value_map during execution of '{node.name}'")
            return value_map[name]

        if node.op_type == OpType.MATMUL:
            a = get(node.inputs[0])
            b = get(node.inputs[1])
            return torch.matmul(a, b)

        if node.op_type == OpType.SCALE:
            x = get(node.inputs[0])
            scale = float(node.attrs.get("scale_factor", 1.0))
            return x * scale

        if node.op_type == OpType.MASK:
            x = get(node.inputs[0])
            T = x.shape[-1]
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            mask_val = float(node.attrs.get("mask_value", float("-inf")))
            return x.masked_fill(mask, mask_val)

        if node.op_type == OpType.SOFTMAX:
            x = get(node.inputs[0])
            dim = int(node.attrs.get("dim", -1))
            return F.softmax(x, dim=dim)

        if node.op_type == OpType.FUSED_SCALE_MASK_SOFTMAX:
            # Reference：展开执行（等价于三步 PyTorch op）
            x = get(node.inputs[0])
            scale = float(node.attrs.get("scale_factor", 1.0))
            is_causal = bool(node.attrs.get("is_causal", True))
            dim = int(node.attrs.get("softmax_dim", -1))
            mask_val = float(node.attrs.get("mask_value", float("-inf")))

            x = x * scale
            if is_causal:
                T = x.shape[-1]
                mask = torch.triu(
                    torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
                )
                x = x.masked_fill(mask, mask_val)
            return F.softmax(x, dim=dim)

        if node.op_type == OpType.ATTENTION_SCORE:
            q = get(node.inputs[0])
            k = get(node.inputs[1])
            v = get(node.inputs[2])
            scale = float(node.attrs.get("scale_factor", q.shape[-1] ** -0.5))
            return F.scaled_dot_product_attention(q, k, v, scale=scale)

        raise NotImplementedError(
            f"ReferenceBackend: no implementation for op type '{node.op_type.name}'"
        )
