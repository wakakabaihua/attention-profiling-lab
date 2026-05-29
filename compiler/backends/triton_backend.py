"""
Backends — Triton Backend
===========================
调用已有 Triton fused kernel 执行 FUSED_SCALE_MASK_SOFTMAX 节点。

职责:
    - 从 CompilationArtifact 中读取 TritonKernelSpec
    - 调用 models/triton_attention.py 中的 triton_fused_scale_mask_softmax
    - 处理非融合节点时回退到 ReferenceBackend

这一层明确了内部 IR -> Triton kernel 调用的边界：
    TritonKernelSpec (scale, is_causal, dim) 完全确定了 kernel 行为，
    与 Stage 2 手写 Triton kernel 的调用路径保持一致。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, List, TYPE_CHECKING

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if TYPE_CHECKING:
    from compiler.lowering.pipeline import CompilationArtifact

from compiler.ir.ops import OpType
from compiler.ir.graph import IRGraph, IRNode
from compiler.lowering.to_triton import TritonKernelSpec, lower_to_triton_specs
from compiler.backends.reference_backend import ReferenceBackend


class TritonBackend:
    """
    Triton fused kernel backend。

    对 FUSED_SCALE_MASK_SOFTMAX 节点调用 Triton kernel；
    其余节点委托 ReferenceBackend 执行。
    """

    def __init__(self):
        self._ref_backend = ReferenceBackend()

    def execute(
        self,
        artifact: "CompilationArtifact",
        *inputs: torch.Tensor,
    ) -> torch.Tensor:
        """执行 fused_ir 图：FUSED 节点用 Triton，其余用 PyTorch。"""
        return self._run_graph(artifact.fused_ir, artifact.triton_specs, *inputs)

    def _run_graph(
        self,
        graph: IRGraph,
        specs: List[TritonKernelSpec],
        *inputs: torch.Tensor,
    ) -> torch.Tensor:
        # 建立 spec 索引：node_name -> TritonKernelSpec
        spec_map: Dict[str, TritonKernelSpec] = {s.node_name: s for s in specs}

        input_nodes = graph.get_input_nodes()
        if len(inputs) != len(input_nodes):
            raise ValueError(
                f"Graph expects {len(input_nodes)} inputs but got {len(inputs)}"
            )

        value_map: Dict[str, torch.Tensor] = {}
        for node, tensor in zip(input_nodes, inputs):
            value_map[node.name] = tensor

        ordered = graph.topological_sort()
        for node in ordered:
            if node.op_type == OpType.INPUT:
                continue
            if node.op_type == OpType.OUTPUT:
                if node.inputs and node.inputs[0] in value_map:
                    value_map[node.name] = value_map[node.inputs[0]]
                continue

            if node.op_type == OpType.FUSED_SCALE_MASK_SOFTMAX and node.name in spec_map:
                spec = spec_map[node.name]
                inp = value_map[node.inputs[0]]
                result = self._call_triton_kernel(inp, spec)
            else:
                # 非融合节点：委托 reference 执行
                result = self._ref_backend._execute_node(node, value_map)

            if result is not None:
                value_map[node.name] = result

        output_nodes = graph.get_output_nodes()
        if output_nodes:
            last = output_nodes[-1]
            if last.name in value_map:
                return value_map[last.name]
            if last.inputs and last.inputs[0] in value_map:
                return value_map[last.inputs[0]]
        return list(value_map.values())[-1]

    def _call_triton_kernel(
        self, scores: torch.Tensor, spec: TritonKernelSpec
    ) -> torch.Tensor:
        """调用 Stage 2 的 Triton fused kernel。"""
        # 延迟导入 Triton 相关代码（避免非 CUDA 环境下报错）
        from models.triton_attention import triton_fused_scale_mask_softmax

        if not scores.is_cuda:
            raise RuntimeError(
                "TritonBackend requires CUDA tensors. "
                f"Got device: {scores.device}"
            )
        return triton_fused_scale_mask_softmax(scores, spec.scale_factor)
