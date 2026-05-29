"""
Backends — TVM Relax Backend
==============================
使用 TVM Relax 编译并执行 FUSED_SCALE_MASK_SOFTMAX 节点。

流程:
    1. lower_to_relax(graph) → tvm.IRModule
    2. relax.build(mod, target="cuda") → Executable
    3. relax.VirtualMachine(ex, dev).run() → Tensor

这一层明确了内部 IR 与 TVM 编译器之间的边界：
    CompilationArtifact (IRGraph) → Relax IRModule → GPU Executable
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if TYPE_CHECKING:
    from compiler.lowering.pipeline import CompilationArtifact

from compiler.ir.ops import OpType
from compiler.ir.graph import IRGraph


class TVMBackend:
    """
    TVM Relax 后端。

    对 FUSED_SCALE_MASK_SOFTMAX 节点使用 TVM Relax 编译执行；
    如果 TVM 不可用或图不含融合节点，则委托 ReferenceBackend。

    Args:
        target: TVM 编译目标，默认 "cuda"。
        cache_compiled: 是否缓存编译结果（相同形状重复执行时避免重编译）。
    """

    def __init__(
        self,
        target: str = "cuda",
        cache_compiled: bool = True,
    ):
        self._target_str = target
        self._cache = {} if cache_compiled else None
        self._tvm_available = self._check_tvm()

    @staticmethod
    def _check_tvm() -> bool:
        try:
            import tvm  # noqa: F401
            import tvm.relax  # noqa: F401
            return True
        except ImportError:
            return False

    def execute(
        self,
        artifact: "CompilationArtifact",
        *inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行 fused_ir 图：FUSED 节点用 TVM Relax，其余情况委托 Reference。
        """
        if not self._tvm_available:
            from compiler.backends.reference_backend import ReferenceBackend
            return ReferenceBackend().execute(artifact, *inputs)

        # 找融合节点
        fused_node = self._find_fused_node(artifact.fused_ir)
        if fused_node is None:
            from compiler.backends.reference_backend import ReferenceBackend
            return ReferenceBackend().execute(artifact, *inputs)

        if len(inputs) == 0:
            raise ValueError("TVMBackend.execute() requires at least one input tensor.")

        scores = inputs[0]
        if not scores.is_cuda:
            raise RuntimeError(
                f"TVMBackend requires CUDA tensors. Got device: {scores.device}"
            )

        return self._run_via_tvm(artifact.fused_ir, scores)

    def _run_via_tvm(self, graph: IRGraph, scores: torch.Tensor) -> torch.Tensor:
        """调用 TVM Relax 编译链执行。"""
        import tvm
        import tvm.relax as relax
        from tvm_integration.relax_importer import lower_to_relax

        shape: Tuple[int, int, int, int] = tuple(scores.shape)  # type: ignore[assignment]
        cache_key = (id(graph), shape)

        if self._cache is not None and cache_key in self._cache:
            vm, dev = self._cache[cache_key]
        else:
            mod = lower_to_relax(graph, input_shape=shape)
            target = tvm.target.Target(self._target_str)
            ex = relax.build(mod, target)
            dev = tvm.cuda(scores.device.index or 0)
            vm = relax.VirtualMachine(ex, dev)
            if self._cache is not None:
                self._cache[cache_key] = (vm, dev)

        # PyTorch CUDA Tensor → TVM Tensor (via DLPack, zero-copy)
        from tvm import runtime as tvmrt
        inp_tvm = tvmrt.from_dlpack(scores)
        out_tvm = vm["main"](inp_tvm)

        # TVM Tensor → PyTorch Tensor (via DLPack)
        return torch.from_dlpack(out_tvm)

    @staticmethod
    def _find_fused_node(graph: IRGraph):
        for node in graph.nodes:
            if node.op_type == OpType.FUSED_SCALE_MASK_SOFTMAX:
                return node
        return None
