"""
Compare All Backends — End-to-End Benchmark
=============================================
对同一 attention 输入在所有可用 backend 上运行 benchmark，
生成跨阶段对比报告（Stage 1 ~ Stage 4）。

支持的 backend:
    baseline    — ManualAttention（unfused，PyTorch eager）
    sdpa        — PyTorch scaled_dot_product_attention（FlashAttention）
    triton      — Stage 2 Triton fused kernel（手写）
    compiler    — Stage 4 Mini Compiler Pipeline（reference backend）
    compiler_triton — Stage 4 + Triton backend

用法:
    python benchmarks/compare_all_backends.py [--seq_len 128] [--n_iter 100]

输出:
    - 控制台 benchmark 表格
    - reports/compiler_pipeline_benchmark_<timestamp>.md
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.mini_transformer import TransformerConfig, ManualAttention
from compiler.ir.ops import OpType
from compiler.ir.graph import IRShape, IRNode, IRGraph
from compiler.passes.fusion import ScaleMaskSoftmaxFusionPass
from compiler.lowering.pipeline import CompilerPipeline
from compiler.runtime.benchmark_runner import BenchmarkRunner, BenchmarkReport, BackendStats


# ─────────────────────────────────────────────────────────────────────
# 手动构建 IR 图（跳过 FX trace 依赖）
# ─────────────────────────────────────────────────────────────────────

def build_sms_ir(scale: float, seq_len: int, batch: int, heads: int) -> IRGraph:
    shape = IRShape([batch, heads, seq_len, seq_len])
    g = IRGraph(name="scale_mask_softmax")
    g.add_node(IRNode("scores", OpType.INPUT, output_shape=shape))
    g.add_node(IRNode("scale_0", OpType.SCALE, inputs=["scores"],
                      attrs={"scale_factor": scale}, output_shape=shape))
    g.add_node(IRNode("mask_0", OpType.MASK, inputs=["scale_0"],
                      attrs={"is_causal": True, "mask_value": float("-inf")},
                      output_shape=shape))
    g.add_node(IRNode("softmax_0", OpType.SOFTMAX, inputs=["mask_0"],
                      attrs={"dim": -1}, output_shape=shape))
    g.add_node(IRNode("output", OpType.OUTPUT, inputs=["softmax_0"]))
    return g


# ─────────────────────────────────────────────────────────────────────
# 各 Backend 封装为统一 callable
# ─────────────────────────────────────────────────────────────────────

def make_baseline(scale: float):
    """ManualAttention 中的 scale + mask + softmax（PyTorch unfused）。"""
    def fn(scores: torch.Tensor) -> torch.Tensor:
        T = scores.shape[-1]
        mask = torch.triu(
            torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1
        )
        out = F.softmax(scores.masked_fill(mask, float("-inf")) * scale, dim=-1)
        return out
    return fn


def make_sdpa():
    """PyTorch SDPA（作为对比用），需要 Q/K/V 形式。"""
    def fn(scores: torch.Tensor) -> torch.Tensor:
        # scores 形状为 (B, H, T, T)，此处用 fake Q/K 重建 SDPA 调用
        # 仅用于 latency 对比，不保证数值严格等价于 SMS 路径
        B, H, T, _ = scores.shape
        D = 64
        q = torch.randn(B, H, T, D, device=scores.device, dtype=scores.dtype)
        k = torch.randn(B, H, T, D, device=scores.device, dtype=scores.dtype)
        v = torch.randn(B, H, T, D, device=scores.device, dtype=scores.dtype)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return fn


def make_triton(scale: float):
    """Stage 2 Triton fused kernel。"""
    from models.triton_attention import triton_fused_scale_mask_softmax
    def fn(scores: torch.Tensor) -> torch.Tensor:
        return triton_fused_scale_mask_softmax(scores, scale)
    return fn


def make_compiler_reference(ir: IRGraph):
    """Stage 4 Mini Compiler Pipeline，Reference backend（FUSED 节点展开执行）。"""
    from compiler.backends.reference_backend import ReferenceBackend
    pipeline = CompilerPipeline(backend="reference")
    artifact = pipeline.compile_ir(ir)
    ref = ReferenceBackend()
    def fn(scores: torch.Tensor) -> torch.Tensor:
        return ref.execute(artifact, scores)
    return fn


def make_compiler_triton(ir: IRGraph):
    """Stage 4 Mini Compiler Pipeline，Triton backend。"""
    from compiler.backends.triton_backend import TritonBackend
    pipeline = CompilerPipeline(backend="triton")
    artifact = pipeline.compile_ir(ir)
    tb = TritonBackend()
    def fn(scores: torch.Tensor) -> torch.Tensor:
        return tb.execute(artifact, scores)
    return fn


def make_compiler_tvm(ir: IRGraph):
    """Stage 4 Mini Compiler Pipeline，TVM Relax backend。"""
    from compiler.backends.tvm_backend import TVMBackend
    pipeline = CompilerPipeline(backend="reference")
    artifact = pipeline.compile_ir(ir)
    tvm_b = TVMBackend()
    def fn(scores: torch.Tensor) -> torch.Tensor:
        return tvm_b.execute(artifact, scores)
    return fn


# ─────────────────────────────────────────────────────────────────────
# Benchmark 工具
# ─────────────────────────────────────────────────────────────────────

def benchmark_fn(fn, scores, n_iter=100, warmup=10, extra_warmup=0):
    """对 fn 进行 CUDA event 计时 benchmark。

    Args:
        warmup:       基础 warmup 次数（功能预热，触发 JIT / kernel cache）
        extra_warmup: 额外稳定化 warmup 次数（用于 Python 层 IR 遍历 / TVM VM
                      等高抖动路径，消除前几次调用的调度开销）
    """
    is_cuda = scores.is_cuda
    for _ in range(warmup + extra_warmup):
        out = fn(scores)
    if is_cuda:
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_iter):
        if is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(scores)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
        else:
            import time
            t0 = time.perf_counter()
            out = fn(scores)
            latencies.append((time.perf_counter() - t0) * 1000)

    stats = BackendStats(backend="", latency_ms=latencies)
    return out, stats


def check_correctness(out, ref, atol=1e-3):
    try:
        return torch.allclose(out.float(), ref.float(), atol=atol, rtol=1e-3)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────

def run_comparison(
    seq_len: int = 128,
    batch: int = 1,
    heads: int = 12,
    head_dim: int = 64,
    n_iter: int = 100,
    warmup: int = 10,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    save_report: bool = True,
) -> BenchmarkReport:
    scale = head_dim ** -0.5
    print(f"\nCompare All Backends — SMS Attention")
    print(f"Config: B={batch}, H={heads}, T={seq_len}, head_dim={head_dim}")
    print(f"dtype={dtype}, device={device}, n_iter={n_iter}")

    scores = torch.randn(batch, heads, seq_len, seq_len,
                         device=device, dtype=dtype)

    ir = build_sms_ir(scale, seq_len, batch, heads)

    # 定义所有 backend（不导入 Triton 若不在 CUDA 上）
    backends: Dict[str, callable] = {}
    backends["baseline"] = make_baseline(scale)
    backends["compiler (ref)"] = make_compiler_reference(ir)

    if device == "cuda":
        try:
            backends["triton (stage2)"] = make_triton(scale)
        except Exception as e:
            print(f"  [warn] Triton import failed: {e}")

        try:
            backends["compiler (triton)"] = make_compiler_triton(ir)
        except Exception as e:
            print(f"  [warn] compiler+triton backend failed: {e}")

        try:
            backends["compiler (tvm)"] = make_compiler_tvm(ir)
        except Exception as e:
            print(f"  [warn] compiler+tvm backend failed: {e}")

    # Python IR 遍历 / TVM VM 路径需要更多 warmup 来稳定 std
    # 参见 reports/compiler_pipeline_benchmark_*.md 的"数据异常说明"
    EXTRA_WARMUP = {
        "compiler (ref)": 40,   # Python dict lookup + IR dispatch 路径
        "compiler (tvm)": 40,   # TVM VirtualMachine + DLPack 首次调度
    }

    # 跑 reference baseline
    ref_out, ref_stats = benchmark_fn(backends["baseline"], scores, n_iter, warmup)

    report = BenchmarkReport(
        title="Compare All Backends: scale + causal_mask + softmax",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        baseline_backend="baseline",
    )
    report.stats["baseline"] = ref_stats
    report.stats["baseline"].backend = "baseline"
    report.correctness["baseline"] = True
    report.max_abs_err["baseline"] = 0.0

    for name, fn in backends.items():
        if name == "baseline":
            continue
        try:
            extra = EXTRA_WARMUP.get(name, 0)
            out, stats = benchmark_fn(fn, scores, n_iter, warmup, extra_warmup=extra)
            stats.backend = name
            report.stats[name] = stats
            correct = check_correctness(out, ref_out.to(out.device))
            report.correctness[name] = correct
            err = float((out.float() - ref_out.float()).abs().max())
            report.max_abs_err[name] = err
        except Exception as e:
            print(f"  [error] {name}: {e}")

    report.print_table()

    if save_report:
        _save_report(report)

    return report


def _save_report(report: BenchmarkReport) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"compiler_pipeline_benchmark_{timestamp}.md"
    out_path.write_text(report.to_markdown(), encoding="utf-8")
    print(f"\nReport saved: {out_path.relative_to(_PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all compiler pipeline backends")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if args.fp32 else torch.float16

    run_comparison(
        seq_len=args.seq_len,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
        n_iter=args.n_iter,
        warmup=args.warmup,
        dtype=dtype,
        device=device,
    )
