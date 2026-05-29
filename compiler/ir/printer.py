"""
Internal IR — Printer
=======================
将内部 IR 图打印为可读的文本格式，用于调试和 before/after 对比。

输出格式示例:
    ──────────────────────────────────────────────
    IRGraph: attention_score (6 nodes)
    ──────────────────────────────────────────────
    %scores_input  = INPUT()                       -> [-1, 12, 128, 128]
    %scale_0       = SCALE(%scores_input)          -> [-1, 12, 128, 128]  {scale_factor=0.125}
    %mask_0        = MASK(%scale_0)                -> [-1, 12, 128, 128]  {is_causal=True}
    %softmax_0     = SOFTMAX(%mask_0)              -> [-1, 12, 128, 128]  {dim=-1}
    %output        = OUTPUT(%softmax_0)
    ──────────────────────────────────────────────
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compiler.ir.graph import IRGraph, IRNode


_SEPARATOR = "─" * 60


def print_ir(graph: "IRGraph", title: str = "") -> str:
    """
    将 IRGraph 格式化为可读字符串并打印到控制台。

    Args:
        graph:  要打印的 IRGraph
        title:  可选标题前缀（用于 before/after 对比）

    Returns:
        格式化后的字符串（同时打印到 stdout）
    """
    lines = _format_ir(graph, title)
    output = "\n".join(lines)
    print(output)
    return output


def format_ir(graph: "IRGraph", title: str = "") -> str:
    """返回格式化后的 IR 字符串（不打印）。"""
    return "\n".join(_format_ir(graph, title))


def _format_ir(graph: "IRGraph", title: str = "") -> list[str]:
    lines: list[str] = [_SEPARATOR]

    header = f"IRGraph: {graph.name} ({graph.num_nodes} nodes)"
    if title:
        header = f"[{title}] {header}"
    lines.append(header)
    lines.append(_SEPARATOR)

    try:
        ordered = graph.topological_sort()
    except ValueError:
        # 图有环时按插入顺序打印
        ordered = graph.nodes

    for node in ordered:
        lines.append(_format_node(node))

    lines.append(_SEPARATOR)
    return lines


def _format_node(node: "IRNode") -> str:
    # 输入列表
    if node.inputs:
        inputs_str = "(" + ", ".join(f"%{i}" for i in node.inputs) + ")"
    else:
        inputs_str = "()"

    # 操作部分
    op_part = f"%{node.name:<20} = {node.op_type.name}{inputs_str}"

    # 形状部分
    if node.output_shape is not None:
        shape_str = f"  -> {node.output_shape.dims}"
    else:
        shape_str = ""

    # 属性部分
    if node.attrs:
        attrs_parts = [f"{k}={v!r}" for k, v in sorted(node.attrs.items())]
        attrs_str = f"  {{{', '.join(attrs_parts)}}}"
    else:
        attrs_str = ""

    return op_part + shape_str + attrs_str


def diff_ir(before: "IRGraph", after: "IRGraph") -> str:
    """
    打印两个 IR 图的对比摘要（节点数变化 + 新增/删除节点）。

    Returns:
        对比字符串（同时打印到 stdout）
    """
    before_names = {n.name for n in before.nodes}
    after_names = {n.name for n in after.nodes}

    added = after_names - before_names
    removed = before_names - after_names

    lines = [
        _SEPARATOR,
        f"IR Diff: {before.name} -> {after.name}",
        f"  nodes: {before.num_nodes} -> {after.num_nodes} "
        f"(+{len(added)} added, -{len(removed)} removed)",
    ]

    if removed:
        lines.append("  removed: " + ", ".join(sorted(removed)))
    if added:
        lines.append("  added:   " + ", ".join(sorted(added)))

    lines.append(_SEPARATOR)
    output = "\n".join(lines)
    print(output)
    return output
