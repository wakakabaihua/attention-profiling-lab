"""
torch.compile Profiling
========================
使用 torch.compile（Inductor 后端）编译 MiniTransformerBlock 并进行性能分析。
将 kernel 数量和延迟与基线版本、SDPA 版本进行对比。

用法：
    python benchmarks/profile_compiled.py [--seq_len 128] [--hidden_size 768] \
           [--num_heads 12] [--batch_size 1] [--warmup 20] [--repeat 20] \
           [--backend inductor]
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.mini_transformer import MiniTransformerBlock, TransformerConfig


def parse_args():
    p = argparse.ArgumentParser(description="torch.compile 性能分析")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20,
                   help="额外预热次数（包含编译）")
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--trace_dir", type=str, default="traces")
    p.add_argument("--backend", type=str, default="inductor",
                   choices=["inductor", "eager", "aot_eager", "cudagraphs"],
                   help="torch.compile 后端")
    p.add_argument("--mode", type=str, default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help="torch.compile 模式")
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
    print("  torch.compile 性能分析")
    print("=" * 60)
    print(f"  hidden_size : {cfg.hidden_size}")
    print(f"  num_heads   : {cfg.num_heads}")
    print(f"  seq_len     : {cfg.seq_len}")
    print(f"  batch_size  : {cfg.batch_size}")
    print(f"  dtype       : {cfg.dtype}")
    print(f"  后端        : {args.backend}")
    print(f"  模式        : {args.mode}")
    print("=" * 60)

    # ---- 模型 ----
    model = MiniTransformerBlock(cfg, use_sdpa=False).to(cfg.device).to(cfg.dtype)
    model.eval()

    # ---- 编译 ----
    print(f"\n🔧 正在编译，后端='{args.backend}'，模式='{args.mode}' ...")
    compiled_model = torch.compile(model, backend=args.backend, mode=args.mode)

    x = torch.randn(
        cfg.batch_size, cfg.seq_len, cfg.hidden_size,
        device=cfg.device, dtype=cfg.dtype,
    )

    # ---- 预热（包含编译） ----
    print(f"\n⏳ 正在预热（{args.warmup} 次迭代，包含 JIT 编译）...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = compiled_model(x)
    torch.cuda.synchronize()
    print("✅ 预热 + 编译完成。")

    # ---- 性能分析 ----
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "compiled_trace.json"

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
                with record_function("transformer_block_compiled"):
                    _ = compiled_model(x)
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

    events = prof.key_averages()
    cuda_events = [e for e in events if e.device_time_total > 0]
    total_cuda_time = sum(e.device_time_total for e in cuda_events)

    print("\n" + "=" * 60)
    print("  📊 Kernel 统计摘要（torch.compile）")
    print("=" * 60)
    print(f"  不同 CUDA 算子总数    : {len(cuda_events)}")
    print(f"  CUDA 总时间          : {total_cuda_time / 1e3:.2f} ms")

    # ---- 导出 trace ----
    prof.export_chrome_trace(str(trace_path))
    print(f"\n💾 Chrome trace 已保存 → {trace_path}")

    print("\n✅ 三组对比：")
    print("   python benchmarks/profile_attention.py    → baseline_trace.json")
    print("   python benchmarks/profile_flash_attn.py   → sdpa_trace.json")
    print("   python benchmarks/profile_compiled.py     → compiled_trace.json")


if __name__ == "__main__":
    main()
