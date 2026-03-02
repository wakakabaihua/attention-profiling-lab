"""
基线注意力 Profiling
====================
使用 PyTorch Profiler 对 MiniTransformerBlock 的 *手写*（未融合）注意力进行性能分析。
输出：
  1. 按 CUDA 时间排序的控制台表格。
  2. Chrome trace JSON → traces/baseline_trace.json
  3. （可选）TensorBoard 日志 → traces/tb_baseline/

用法：
    python benchmarks/profile_attention.py [--seq_len 128] [--hidden_size 768] \
           [--num_heads 12] [--batch_size 1] [--warmup 10] [--repeat 20]
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# 允许从项目根目录导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.mini_transformer import MiniTransformerBlock, TransformerConfig


def parse_args():
    p = argparse.ArgumentParser(description="基线注意力性能分析")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10, help="预热迭代次数")
    p.add_argument("--repeat", type=int, default=20, help="profiling 迭代次数")
    p.add_argument("--trace_dir", type=str, default="traces")
    p.add_argument("--use_tb", action="store_true", help="导出 TensorBoard 日志")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- 配置 ----
    cfg = TransformerConfig(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    print("=" * 60)
    print("  基线注意力 Profiling（手写 / 未融合）")
    print("=" * 60)
    print(f"  hidden_size : {cfg.hidden_size}")
    print(f"  num_heads   : {cfg.num_heads}")
    print(f"  seq_len     : {cfg.seq_len}")
    print(f"  batch_size  : {cfg.batch_size}")
    print(f"  dtype       : {cfg.dtype}")
    print(f"  预热次数  : {args.warmup}")
    print(f"  重复次数  : {args.repeat}")
    print("=" * 60)

    # ---- 模型 ----
    model = MiniTransformerBlock(cfg, use_sdpa=False).to(cfg.device).to(cfg.dtype)
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
    trace_path = trace_dir / "baseline_trace.json"

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
                with record_function("transformer_block"):
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

    # ---- Kernel 数量与小 kernel 分析 ----
    print("\n" + "=" * 60)
    print("  📊 Kernel 统计摘要")
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
        print("\n  小 kernel（可融合候选）：")
        for e in sorted(small_kernels, key=lambda x: x.device_time_total, reverse=True)[:10]:
            avg_us = e.device_time_total / max(e.count, 1)
            print(f"    {e.key:<40s}  调用={e.count:>4d}  均值={avg_us:>7.1f}μs")

    # ---- 内存分配事件 ----
    print("\n" + "=" * 60)
    print("  🧠 内存分配摘要")
    print("=" * 60)
    mem_events = [e for e in events if e.cpu_memory_usage != 0 or e.device_memory_usage != 0]
    for e in mem_events[:15]:
        print(
            f"    {e.key:<40s}  CPU={e.cpu_memory_usage:>10d} B  "
            f"CUDA={e.device_memory_usage:>10d} B"
        )

    # ---- 导出 trace ----
    prof.export_chrome_trace(str(trace_path))
    print(f"\n💾 Chrome trace 已保存 → {trace_path}")

    if args.use_tb:
        tb_dir = trace_dir / "tb_baseline"
        tb_dir.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(tb_dir / "trace.json"))
        print(f"📈 TensorBoard trace 目录 → {tb_dir}")

    print("\n✅ 性能分析完成。使用以下方式打开 trace：")
    print(f"   chrome://tracing  →  加载 {trace_path}")
    print("   或：perfetto.dev  →  打开 trace 文件")


if __name__ == "__main__":
    main()
