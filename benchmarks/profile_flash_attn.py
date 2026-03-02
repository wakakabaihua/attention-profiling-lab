"""
FlashAttention / SDPA Profiling
================================
使用 PyTorch SDPA（scaled_dot_product_attention）对 MiniTransformerBlock 进行性能分析，
其内部会调度 FlashAttention 或内存高效后端。

与基线版本（profile_attention.py）对比，衡量融合带来的收益。

用法：
    python benchmarks/profile_flash_attn.py [--seq_len 128] [--hidden_size 768] \
           [--num_heads 12] [--batch_size 1] [--warmup 10] [--repeat 20]
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.mini_transformer import MiniTransformerBlock, TransformerConfig


def parse_args():
    p = argparse.ArgumentParser(description="SDPA / FlashAttention 性能分析")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=20)
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
    print("  SDPA / FlashAttention Profiling（融合版本）")
    print("=" * 60)
    print(f"  hidden_size : {cfg.hidden_size}")
    print(f"  num_heads   : {cfg.num_heads}")
    print(f"  seq_len     : {cfg.seq_len}")
    print(f"  batch_size  : {cfg.batch_size}")
    print(f"  dtype       : {cfg.dtype}")
    print("=" * 60)

    # 检查 SDPA 后端可用性
    print("\n📋 SDPA 后端可用性：")
    if hasattr(torch.backends, "cuda"):
        print(f"  Flash Attention  : {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"  内存高效后端    : {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"  Math（回退）     : {torch.backends.cuda.math_sdp_enabled()}")

    # ---- 模型 ----
    model = MiniTransformerBlock(cfg, use_sdpa=True).to(cfg.device).to(cfg.dtype)
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
    trace_path = trace_dir / "sdpa_trace.json"

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
                with record_function("transformer_block_sdpa"):
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

    # ---- Kernel 数量对比 ----
    events = prof.key_averages()
    cuda_events = [e for e in events if e.device_time_total > 0]
    total_cuda_time = sum(e.device_time_total for e in cuda_events)

    print("\n" + "=" * 60)
    print("  📊 Kernel 统计摘要（SDPA）")
    print("=" * 60)
    print(f"  不同 CUDA 算子总数    : {len(cuda_events)}")
    print(f"  CUDA 总时间          : {total_cuda_time / 1e3:.2f} ms")

    # ---- 导出 trace ----
    prof.export_chrome_trace(str(trace_path))
    print(f"\n💾 Chrome trace 已保存 → {trace_path}")
    print("\n✅ 与基线对比：")
    print("   python benchmarks/profile_attention.py   → baseline_trace.json")
    print("   python benchmarks/profile_flash_attn.py  → sdpa_trace.json")


if __name__ == "__main__":
    main()
