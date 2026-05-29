"""
TVM Relax Importer
===================
将内部 IRGraph 转换为 TVM Relax IRModule，供 TVMBackend 编译执行。

核心思路:
    - FUSED_SCALE_MASK_SOFTMAX → relax.op.multiply + relax.op.nn.softmax 序列
    - 支持因果遮罩（is_causal=True）：用 tril + where 填充 -inf
    - 输出 IRModule 可直接用 relax.build(mod, target="cuda") 编译

这一层明确了内部 IR 与 TVM 编译器之间的边界：
    IRGraph (内部节点) → tvm.IRModule (Relax 函数)
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

try:
    import tvm
    import tvm.relax as relax
    from tvm.relax import BlockBuilder
    _TVM_AVAILABLE = True
except ImportError:
    _TVM_AVAILABLE = False

if TYPE_CHECKING:
    from tvm import IRModule
    from compiler.ir.graph import IRGraph, IRNode

from compiler.ir.ops import OpType


def _require_tvm() -> None:
    if not _TVM_AVAILABLE:
        raise ImportError(
            "TVM is not installed. Build from source and configure "
            "LD_LIBRARY_PATH=/data/github/tvm/build/lib."
        )


def lower_to_relax(
    graph: "IRGraph",
    input_shape: Tuple[int, int, int, int] = (1, 12, 128, 128),
) -> "IRModule":
    """
    将 IRGraph 降低为 TVM Relax IRModule。

    生成的模块包含一个 Relax 函数 `main`，签名:
        main(scores: Tensor[B,H,T,T, fp16]) -> Tensor[B,H,T,T, fp16]

    Args:
        graph: 经过 FusionPass 的 IRGraph（含 FUSED_SCALE_MASK_SOFTMAX 节点）
        input_shape: (B, H, T_q, T_k) 具体形状，用于 TVM 静态编译

    Returns:
        tvm.IRModule（包含 `main` Relax 函数）

    Raises:
        ImportError: TVM 未安装时抛出
        ValueError: 图中找不到 FUSED_SCALE_MASK_SOFTMAX 节点时抛出
    """
    _require_tvm()

    fused_node = _find_fused_node(graph)
    if fused_node is None:
        raise ValueError(
            "IRGraph contains no FUSED_SCALE_MASK_SOFTMAX node. "
            "Run FusionPass before calling lower_to_relax()."
        )

    scale = float(fused_node.attrs.get("scale_factor", 1.0))
    is_causal = bool(fused_node.attrs.get("is_causal", True))
    softmax_dim = int(fused_node.attrs.get("softmax_dim", -1))

    return _build_relax_module(scale, is_causal, softmax_dim, input_shape)


def _find_fused_node(graph: "IRGraph") -> Optional["IRNode"]:
    """返回图中第一个 FUSED_SCALE_MASK_SOFTMAX 节点，找不到则返回 None。"""
    for node in graph.nodes:
        if node.op_type == OpType.FUSED_SCALE_MASK_SOFTMAX:
            return node
    return None


def _build_relax_module(
    scale: float,
    is_causal: bool,
    softmax_dim: int,
    shape: Tuple[int, int, int, int],
) -> "IRModule":
    """
    构建 Relax IRModule（具体形状版本）。

    函数签名:
        main(scores: Tensor[B, H, T, T, fp16]) -> Tensor[B, H, T, T, fp16]

    计算:
        scaled = scores * scale
        if is_causal:
            masked = where(tril_mask, scaled, -inf)
        else:
            masked = scaled
        return softmax(masked, axis=softmax_dim)
    """
    B, H, T_q, T_k = shape

    bb = BlockBuilder()
    scores_var = relax.Var("scores", relax.TensorStructInfo([B, H, T_q, T_k], "float16"))

    with bb.function("main", params=[scores_var]):
        with bb.dataflow():
            # 1. Scale
            scale_const = relax.const(scale, dtype="float16")
            scaled = bb.emit(relax.op.multiply(scores_var, scale_const))

            # 2. 因果遮罩（可选）
            if is_causal:
                # 下三角为 1，上三角为 0；用 where 把上三角位置填 -inf
                ones = bb.emit(relax.op.ones([B, H, T_q, T_k], "float16"))
                tril_mask = bb.emit(relax.op.tril(ones))
                bool_mask = bb.emit(relax.op.astype(tril_mask, "bool"))
                neg_inf_const = relax.const(float("-inf"), dtype="float16")
                neg_inf_broad = bb.emit(
                    relax.op.broadcast_to(neg_inf_const, [B, H, T_q, T_k])
                )
                masked = bb.emit(relax.op.where(bool_mask, scaled, neg_inf_broad))
            else:
                masked = scaled

            # 3. Softmax（axis=-1 对应最后一维 T_k）
            axis = softmax_dim if softmax_dim >= 0 else (len(shape) - 1)
            output = bb.emit(relax.op.nn.softmax(masked, axis=axis))

            gv = bb.emit_output(output)
        bb.emit_func_output(gv)

    return bb.get()


def print_relax_ir(
    graph: "IRGraph",
    input_shape: Tuple[int, int, int, int] = (1, 12, 128, 128),
) -> str:
    """将 IRGraph 转换为 Relax IR 文本，用于调试。"""
    _require_tvm()
    mod = lower_to_relax(graph, input_shape=input_shape)
    return mod.script()
