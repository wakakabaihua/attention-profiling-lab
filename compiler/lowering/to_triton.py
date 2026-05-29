"""
Lowering — Internal IR to Triton Kernel Mapping
==================================================
将 FUSED_SCALE_MASK_SOFTMAX 内部节点的属性映射为 Triton kernel 调用参数。

职责:
    - 从 FUSED_SCALE_MASK_SOFTMAX 节点提取编译期常量（scale, is_causal, dim）
    - 生成 TritonKernelSpec（kernel 调用规格），由 TritonBackend 使用
    - 不直接执行 kernel，只做"IR 属性 -> kernel 参数"的转换

这一层明确了内部 IR 与 Triton backend 之间的边界。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from compiler.ir.graph import IRGraph, IRNode

from compiler.ir.ops import OpType


@dataclass
class TritonKernelSpec:
    """
    描述一个可执行 Triton kernel 调用所需的全部参数。

    由 to_triton 转换层生成，由 TritonBackend 消费。
    """

    node_name: str          # 对应的融合节点名称
    scale_factor: float     # attention scale (1/sqrt(head_dim))
    is_causal: bool         # 是否应用因果遮罩
    softmax_dim: int        # softmax 归约维度（通常为 -1）
    mask_value: float       # 遮罩填充值（通常为 -inf）

    def __repr__(self) -> str:
        return (
            f"TritonKernelSpec(node={self.node_name!r}, "
            f"scale={self.scale_factor}, is_causal={self.is_causal}, "
            f"softmax_dim={self.softmax_dim})"
        )


def lower_to_triton_specs(graph: "IRGraph") -> List[TritonKernelSpec]:
    """
    遍历图，为所有 FUSED_SCALE_MASK_SOFTMAX 节点生成 TritonKernelSpec。

    Args:
        graph: 经过 fusion pass 后的 IRGraph

    Returns:
        TritonKernelSpec 列表（按拓扑顺序）
    """
    try:
        ordered = graph.topological_sort()
    except ValueError:
        ordered = graph.nodes

    specs: List[TritonKernelSpec] = []
    for node in ordered:
        if node.op_type != OpType.FUSED_SCALE_MASK_SOFTMAX:
            continue
        spec = _node_to_spec(node)
        specs.append(spec)
    return specs


def _node_to_spec(node: "IRNode") -> TritonKernelSpec:
    """从单个 FUSED_SCALE_MASK_SOFTMAX 节点提取 TritonKernelSpec。"""
    attrs = node.attrs
    return TritonKernelSpec(
        node_name=node.name,
        scale_factor=float(attrs.get("scale_factor", 1.0)),
        is_causal=bool(attrs.get("is_causal", True)),
        softmax_dim=int(attrs.get("softmax_dim", -1)),
        mask_value=float(attrs.get("mask_value", float("-inf"))),
    )
