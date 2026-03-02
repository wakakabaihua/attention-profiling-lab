"""
Triton 融合 Attention Profiling
================================
使用 PyTorch Profiler 对 Triton 融合注意力 kernel 进行性能分析。
将 TritonAttention 嵌入 MiniTransformerBlock，与 baseline / SDPA / compiled 进行对比。

输出：
  1. 按 CUDA 时间排序的控制台表格。
  2. Chrome trace JSON → traces/triton_trace.json

关键对比点：
  - ManualAttention: scale + mask + softmax 各自独立 kernel
  - TritonAttention: scale + mask + softmax 融合为 1 个 Triton kernel
  - 两者都保留 cublas matmul（QK^T, PV），只改中间部分

用法：
    python benchmarks/profile_triton.py [--seq_len 128] [--hidden_size 768] \
           [--num_heads 12] [--batch_size 1] [--warmup 10] [--repeat 20]
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.mini_transformer import MiniTransformerBlock, TransformerConfig
from models.triton_attention import TritonAttention


class TritonTransformerBlock(MiniTransformerBlock):
    """
    使用 Triton 融合 attention 的 TransformerBlock。

    继承 MiniTransformerBlock，仅将 attn_fn 替换为 TritonAttention。
    其他部分（LayerNorm、MLP、残差连接）保持不变。
    """

    def __init__(self, config=None):
        # 先用 ManualAttention 初始化，然后替换
        super().__init__(config, use_sdpa=False)
        self.attn_fn = TritonAttention(config or TransformerConfig())


def parse_args():
    p = argparse.ArgumentParser(description="Triton 融合注意力性能分析")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10, help="预热迭代次数")
    p.add_argument("--repeat", type=int, default=20, help="profiling 迭代次数")
    p.add_argument("--trace_dir", type=str, default="traces")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = TransformerConfig(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    print("=" * 60)
    print("  Triton 融合 Attention Profiling")
    print("=" * 60)
    print(f"  hidden_size : {cfg.hidden_size}")
    print(f"  num_heads   : {cfg.num_heads}")
    print(f"  seq_len     : {cfg.seq_len}")
    print(f"  batch_size  : {cfg.batch_size}")
    print(f"  dtype       : {cfg.dtype}")
    print(f"  预热次数  : {args.warmup}")
    print(f"  重复次数  : {args.repeat}")
    print("=" * 60)
    print(f"\n  💡 融合策略：")
    print(f"     QK^T matmul    → cublas（保留）")
    print(f"     scale+mask+softmax → Triton fused kernel（融合）")
    print(f"     PV matmul      → cublas（保留）")

    # ---- 模型 ----
    model = TritonTransformerBlock(cfg).to(cfg.device).to(cfg.dtype)
    model.eval()

    x = torch.randn(
        cfg.batch_size, cfg.seq_len, cfg.hidden_size,
        device=cfg.device, dtype=cfg.dtype,
    )

    # ---- 预热 ----
    print(f"\n⏳ 正在预热（{args.warmup} 次迭代）...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
    torch.cuda.synchronize()
    print("✅ 预热完成。")

    # ---- 性能分析 ----
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "triton_trace.json"

    print(f"\n🔬 正在进行性能分析（{args.repeat} 次迭代）...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(args.repeat):
                with record_function("transformer_block_triton"):
                    _ = model(x)
        torch.cuda.synchronize()

    # ---- 结果输出 ----
    print("\n" + "=" * 60)
    print("  🔍 Kernel 级别统计（按 CUDA 时间排序）")
    print("=" * 60)
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=30
        )
    )

    # ---- Kernel 统计摘要 ----
    print("\n" + "=" * 60)
    print("  📊 Kernel 统计摘要（Triton 融合）")
    print("=" * 60)
    events = prof.key_averages()
    cuda_events = [e for e in events if e.device_time_total > 0]
    total_cuda_time = sum(e.device_time_total for e in cuda_events)
    small_kernels = [e for e in cuda_events if e.device_time_total / max(e.count, 1) < 50]

    print(f"  不同 CUDA 算子总数    : {len(cuda_events)}")
    print(f"  CUDA 总时间          : {total_cuda_time / 1e3:.2f} ms")
    print(f"  小 kernel（<50μs 均值）: {len(small_kernels)}")

    if small_kernels:
        small_time = sum(e.device_time_total for e in small_kernels)
        print(f"  小 kernel 时间占比    : {small_time / total_cuda_time * 100:.1f}%")

    # ---- 识别 Triton kernel ----
    triton_events = [e for e in cuda_events if "triton" in e.key.lower()
                     or "_fused_scale_mask_softmax" in e.key.lower()]
    if triton_events:
        print(f"\n  🔶 Triton 融合 Kernel：")
        for e in triton_events:
            avg_us = e.device_time_total / max(e.count, 1)
            print(f"    {e.key:<55s}  调用={e.count:>4d}  "
                  f"均值={avg_us:>7.1f}μs  总计={e.device_time_total:>8.1f}μs")
    else:
        print(f"\n  ⚠️ 未检测到 Triton kernel（可能名称不匹配）")

    # ---- 对比提示 ----
    print("\n" + "=" * 60)
    print("  📊 与其他版本的预期对比")
    print("=" * 60)
    print("  baseline:  scale / mask / softmax 各 1 个 kernel × N 次")
    print("  triton:    scale+mask+softmax 融合为 1 个 kernel × N 次")
    print(f"\n  本次 kernel 总数: {len(cuda_events)}")
    print(f"  本次总耗时:       {total_cuda_time / 1e3:.2f} ms")

    # ---- 导出 trace ----
    prof.export_chrome_trace(str(trace_path))
    print(f"\n💾 Chrome trace 已保存 → {trace_path}")

    print("\n✅ 四组对比：")
    print("   python benchmarks/profile_attention.py    → baseline_trace.json")
    print("   python benchmarks/profile_flash_attn.py   → sdpa_trace.json")
    print("   python benchmarks/profile_compiled.py     → compiled_trace.json")
    print("   python benchmarks/profile_triton.py       → triton_trace.json  ← 当前")


if __name__ == "__main__":
    main()
