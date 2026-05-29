"""
Triton Kernel 性能调优实验
==========================

目标：系统性地探索 Triton kernel 的调优参数，理解 GPU 硬件特性如何影响性能。

背景知识
--------

GPU 上一个 Triton kernel 的性能取决于几个关键参数：

1. **BLOCK_T（tiling / 分块大小）**
   - 控制每个 program（线程块）一次处理多少列
   - 太小 → 循环次数多、launch overhead 大、无法充分利用寄存器/向量化
   - 太大 → 寄存器压力大、occupancy 降低（每个 SM 能同时跑的 program 变少）
   - 当 BLOCK_T >= seq_len 时，循环只执行一次（无循环开销）

2. **num_warps（每个线程块的 warp 数）**
   - 一个 warp = 32 个线程。num_warps=4 → 128 个线程/block
   - 更多 warp → 更好的延迟隐藏（一个 warp 等内存时，其他 warp 可以计算）
   - 更多 warp → 与 BLOCK_T 的比例要匹配，否则部分 warp 闲置
   - RTX 4090 每个 SM 最多 48 个 warp（1536 线程）

3. **num_stages（流水线级数 / software pipelining）**
   - 控制全局内存加载的 double buffering 级数
   - 更多 stage → 更好地隐藏内存延迟（加载下一批数据时处理当前数据）
   - 更多 stage → 需要更多共享内存（每个 stage 一个 buffer）
   - RTX 4090 每个 SM 有 100KB 共享内存

4. **数据类型（fp16 vs fp32）**
   - fp16: 2 字节/元素，带宽翻倍，Tensor Core 可加速
   - fp32: 4 字节/元素，精度更高，带宽减半

实验设计
--------
对每个参数组合 (BLOCK_T, num_warps, num_stages, seq_len, dtype)：
  1. 用 triton.testing.do_bench() 测量 kernel 执行时间
  2. 计算有效带宽（GB/s）和相对加速比
  3. 分析哪些参数组合最优及原因

用法
----
    python benchmarks/tune_triton_kernel.py                  # 运行全部实验
    python benchmarks/tune_triton_kernel.py --quick           # 快速模式（参数空间缩小）
    python benchmarks/tune_triton_kernel.py --seq_len 256     # 指定序列长度
    python benchmarks/tune_triton_kernel.py --autotune        # 使用 Triton 内置 autotune
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import triton
import triton.language as tl
import argparse
import time
from itertools import product


# =====================================================================
# 实验1：BLOCK_T 扫描 — 分块大小对性能的影响
# =====================================================================

@triton.jit
def _tunable_fused_softmax_fwd(
    scores_ptr, output_ptr,
    seq_len,
    scale: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """与原版 _fused_scale_mask_softmax_fwd 相同逻辑，BLOCK_T 作为调优参数。"""
    pid = tl.program_id(0)
    batch_head_idx = pid // seq_len
    row_idx = pid % seq_len
    row_offset = batch_head_idx * seq_len * seq_len + row_idx * seq_len
    col_offsets = tl.arange(0, BLOCK_T)

    # Pass 1: max
    max_val = float("-inf")
    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        causal_mask = cols <= row_idx
        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(causal_mask & mask, x, float("-inf"))
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Pass 2: exp + sum
    sum_exp = 0.0
    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        causal_mask = cols <= row_idx
        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(causal_mask & mask, x, float("-inf"))
        x = tl.exp(x - max_val)
        x = tl.where(causal_mask & mask, x, 0.0)
        sum_exp += tl.sum(x, axis=0)

    # Pass 3: normalize + store
    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        causal_mask = cols <= row_idx
        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(causal_mask & mask, x, float("-inf"))
        x = tl.exp(x - max_val)
        x = tl.where(causal_mask & mask, x, 0.0)
        x = x / (sum_exp + 1e-6)
        tl.store(output_ptr + row_offset + cols, x, mask=mask)


@triton.jit
def _tunable_online_softmax_fwd(
    scores_ptr, output_ptr,
    seq_len,
    scale: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """与原版 _online_softmax_fwd 相同逻辑，BLOCK_T 作为调优参数。"""
    pid = tl.program_id(0)
    batch_head_idx = pid // seq_len
    row_idx = pid % seq_len
    row_offset = batch_head_idx * seq_len * seq_len + row_idx * seq_len
    col_offsets = tl.arange(0, BLOCK_T)

    # Pass 1: online max + sum
    max_val = float("-inf")
    sum_exp = 0.0
    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        causal_mask = cols <= row_idx
        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(causal_mask & mask, x, float("-inf"))

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max)
        exp_x = tl.exp(x - new_max)
        exp_x = tl.where(causal_mask & mask, exp_x, 0.0)
        sum_exp += tl.sum(exp_x, axis=0)
        max_val = new_max

    # Pass 2: normalize + store
    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        causal_mask = cols <= row_idx
        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(causal_mask & mask, x, float("-inf"))
        x = tl.exp(x - max_val)
        x = tl.where(causal_mask & mask, x, 0.0)
        x = x / (sum_exp + 1e-6)
        tl.store(output_ptr + row_offset + cols, x, mask=mask)


# =====================================================================
# 实验2：@triton.autotune — 让 Triton 自动搜索最优配置
# =====================================================================

def get_autotune_configs():
    """返回 Triton autotune 搜索的配置空间。"""
    configs = []
    for block_t in [32, 64, 128, 256]:
        for num_warps in [1, 2, 4, 8]:
            for num_stages in [1, 2, 3, 4]:
                configs.append(
                    triton.Config(
                        {"BLOCK_T": block_t},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=get_autotune_configs(),
    key=["seq_len"],  # 当 seq_len 变化时重新搜索
)
@triton.jit
def _autotuned_online_softmax_fwd(
    scores_ptr, output_ptr,
    seq_len,
    scale: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """使用 @triton.autotune 自动搜索最优 (BLOCK_T, num_warps, num_stages)。"""
    pid = tl.program_id(0)
    batch_head_idx = pid // seq_len
    row_idx = pid % seq_len
    row_offset = batch_head_idx * seq_len * seq_len + row_idx * seq_len
    col_offsets = tl.arange(0, BLOCK_T)

    max_val = float("-inf")
    sum_exp = 0.0
    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        causal_mask = cols <= row_idx
        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(causal_mask & mask, x, float("-inf"))
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max)
        exp_x = tl.exp(x - new_max)
        exp_x = tl.where(causal_mask & mask, exp_x, 0.0)
        sum_exp += tl.sum(exp_x, axis=0)
        max_val = new_max

    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        causal_mask = cols <= row_idx
        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(causal_mask & mask, x, float("-inf"))
        x = tl.exp(x - max_val)
        x = tl.where(causal_mask & mask, x, 0.0)
        x = x / (sum_exp + 1e-6)
        tl.store(output_ptr + row_offset + cols, x, mask=mask)


# =====================================================================
# 辅助函数
# =====================================================================

def run_kernel(kernel_fn, scores_3d, output, BH, T, scale, BLOCK_T,
               num_warps=4, num_stages=2):
    """运行指定的 Triton kernel。"""
    grid = (BH * T,)
    kernel_fn[grid](
        scores_3d, output, T, scale,
        BLOCK_T=BLOCK_T,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def benchmark_kernel(kernel_fn, scores_3d, output, BH, T, scale, BLOCK_T,
                     num_warps=4, num_stages=2, warmup=25, rep=100):
    """
    使用 triton.testing.do_bench 精确计时。

    triton.testing.do_bench 内部做了：
    1. GPU 预热（warmup ms）
    2. 多次重复测量（rep 次）
    3. 返回中位数执行时间（毫秒）
    4. 自动处理 CUDA 同步

    这比 torch.profiler 更适合单 kernel 的微基准测试（micro-benchmark）。
    """
    grid = (BH * T,)

    def kernel_call():
        kernel_fn[grid](
            scores_3d, output, T, scale,
            BLOCK_T=BLOCK_T,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    ms = triton.testing.do_bench(kernel_call, warmup=warmup, rep=rep)
    return ms


def calc_bandwidth(T, BH, dtype_bytes, ms, num_passes):
    """
    计算有效内存带宽（GB/s）。

    每个 program 处理 1 行 T 个元素：
      - 每个 pass 读取 T 个元素
      - 最后一个 pass 写入 T 个元素
      - 总数据量 = BH * T * (num_passes_read * T + T_write) * dtype_bytes

    RTX 4090 理论峰值带宽：~1008 GB/s (GDDR6X)
    """
    total_bytes = BH * T * ((num_passes) * T + T) * dtype_bytes  # reads + writes
    bandwidth_gbs = total_bytes / (ms * 1e-3) / 1e9
    return bandwidth_gbs


# =====================================================================
# 实验运行器
# =====================================================================

def experiment_block_size(args):
    """实验1：固定其他参数，扫描 BLOCK_T 大小。"""
    print("\n" + "=" * 70)
    print("  实验1：BLOCK_T（分块大小）扫描")
    print("=" * 70)
    print("""
  原理：BLOCK_T 控制每个线程块一次处理多少列。
  - 当 BLOCK_T >= seq_len 时，内层循环只执行 1 次（无循环开销）
  - 当 BLOCK_T < seq_len 时，需要多次循环迭代（有额外的循环控制 + 可能的寄存器 spill）
  - BLOCK_T 必须是 2 的幂（Triton 要求向量化宽度对齐）
  - 太大的 BLOCK_T 会导致寄存器压力增大，occupancy 降低
""")

    B, H = args.batch_size, args.num_heads
    D = args.hidden_size // args.num_heads
    BH = B * H
    scale = D ** -0.5
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    dtype_bytes = 2 if args.dtype == "fp16" else 4

    results_3pass = []
    results_online = []

    for T in args.seq_lens:
        print(f"\n  --- seq_len = {T} ---")
        scores = torch.randn(BH, T, T, device="cuda", dtype=dtype)
        output = torch.empty_like(scores)

        block_sizes = [bs for bs in [16, 32, 64, 128, 256, 512, 1024] if bs >= 16]

        print(f"  {'BLOCK_T':>8} | {'3-pass (µs)':>12} | {'Online (µs)':>12} | "
              f"{'3p BW (GB/s)':>13} | {'OL BW (GB/s)':>13} | {'num_iters':>10} | 备注")
        print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*13}-+-{'-'*13}-+-{'-'*10}-+-----")

        best_3p = (float('inf'), 0)
        best_ol = (float('inf'), 0)

        for block_t in block_sizes:
            num_iters = (T + block_t - 1) // block_t
            note = ""

            if block_t >= T:
                note = "单次迭代（无循环开销）"
            elif block_t < 32:
                note = "太小：向量化效率低"

            try:
                ms_3p = benchmark_kernel(
                    _tunable_fused_softmax_fwd, scores, output,
                    BH, T, scale, block_t,
                    num_warps=4, num_stages=2,
                    warmup=args.warmup, rep=args.repeat
                )
                bw_3p = calc_bandwidth(T, BH, dtype_bytes, ms_3p, 3)

                ms_ol = benchmark_kernel(
                    _tunable_online_softmax_fwd, scores, output,
                    BH, T, scale, block_t,
                    num_warps=4, num_stages=2,
                    warmup=args.warmup, rep=args.repeat
                )
                bw_ol = calc_bandwidth(T, BH, dtype_bytes, ms_ol, 2)

                us_3p = ms_3p * 1000
                us_ol = ms_ol * 1000

                if us_3p < best_3p[0]:
                    best_3p = (us_3p, block_t)
                if us_ol < best_ol[0]:
                    best_ol = (us_ol, block_t)

                print(f"  {block_t:>8} | {us_3p:>12.2f} | {us_ol:>12.2f} | "
                      f"{bw_3p:>13.1f} | {bw_ol:>13.1f} | {num_iters:>10} | {note}")

                results_3pass.append({
                    "seq_len": T, "block_t": block_t, "us": us_3p,
                    "bw": bw_3p, "num_iters": num_iters
                })
                results_online.append({
                    "seq_len": T, "block_t": block_t, "us": us_ol,
                    "bw": bw_ol, "num_iters": num_iters
                })

            except Exception as e:
                print(f"  {block_t:>8} | {'FAILED':>12} | {'FAILED':>12} | "
                      f"{'':>13} | {'':>13} | {num_iters:>10} | {e}")

        print(f"\n  最优 3-pass: BLOCK_T={best_3p[1]}, {best_3p[0]:.2f} µs")
        print(f"  最优 Online: BLOCK_T={best_ol[1]}, {best_ol[0]:.2f} µs")

    return results_3pass, results_online


def experiment_num_warps(args):
    """实验2：固定 BLOCK_T，扫描 num_warps 数量。"""
    print("\n" + "=" * 70)
    print("  实验2：num_warps（线程块内 warp 数）扫描")
    print("=" * 70)
    print("""
  原理：num_warps 控制每个线程块（CTA）内的并行度。
  - 1 warp = 32 个 CUDA 线程
  - num_warps=4 → 128 线程/block, num_warps=8 → 256 线程/block
  - 更多 warp → 更好的延迟隐藏（latency hiding）
    → 当一个 warp 等待全局内存加载时，调度器可以切换到另一个 warp 执行计算
  - 但 warp 数也影响寄存器分配：更多 warp → 每个 warp 可用寄存器更少
  - RTX 4090（SM 8.9）每个 SM 有 65536 个 32位寄存器，最多 48 个 warp
  - 关键指标：occupancy = 实际活跃 warp 数 / SM 最大 warp 数
""")

    B, H = args.batch_size, args.num_heads
    D = args.hidden_size // args.num_heads
    BH = B * H
    scale = D ** -0.5
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    dtype_bytes = 2 if args.dtype == "fp16" else 4

    warp_counts = [1, 2, 4, 8, 16]

    for T in args.seq_lens:
        BLOCK_T = triton.next_power_of_2(T)
        scores = torch.randn(BH, T, T, device="cuda", dtype=dtype)
        output = torch.empty_like(scores)

        print(f"\n  --- seq_len = {T}, BLOCK_T = {BLOCK_T} ---")
        print(f"  {'num_warps':>10} | {'threads':>8} | {'Online (µs)':>12} | "
              f"{'BW (GB/s)':>10} | {'相对':>6} | 分析")
        print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*6}-+-----")

        baseline_us = None

        for nw in warp_counts:
            threads = nw * 32
            note = ""

            # 推断行为
            if threads > BLOCK_T:
                note = "warp > 数据宽度 → 部分 warp 闲置"
            elif threads < 32:
                note = "低于单 warp"

            try:
                ms = benchmark_kernel(
                    _tunable_online_softmax_fwd, scores, output,
                    BH, T, scale, BLOCK_T,
                    num_warps=nw, num_stages=2,
                    warmup=args.warmup, rep=args.repeat
                )
                us = ms * 1000
                bw = calc_bandwidth(T, BH, dtype_bytes, ms, 2)

                if baseline_us is None:
                    baseline_us = us
                relative = baseline_us / us

                print(f"  {nw:>10} | {threads:>8} | {us:>12.2f} | "
                      f"{bw:>10.1f} | {relative:>6.2f}x | {note}")

            except Exception as e:
                print(f"  {nw:>10} | {threads:>8} | {'FAILED':>12} | "
                      f"{'':>10} | {'':>6} | {e}")


def experiment_num_stages(args):
    """实验3：固定 BLOCK_T 和 num_warps，扫描 num_stages。"""
    print("\n" + "=" * 70)
    print("  实验3：num_stages（流水线级数）扫描")
    print("=" * 70)
    print("""
  原理：num_stages 控制 software pipelining 的深度（loop unrolling + prefetch）。
  - num_stages=1: 无 prefetch，每次循环等待加载完成后再计算
  - num_stages=2: double buffering，加载下一批数据的同时计算当前数据
  - num_stages=3+: 更深流水线，需要更多共享内存（每级一个 buffer）

  关键权衡：
  - 更多 stages → 更好的延迟隐藏 → 但每级需要 BLOCK_T × dtype_bytes 的共享内存
  - RTX 4090 每个 SM 有 100KB 共享内存
  - 如果 BLOCK_T=128, fp16 → 每级 256 bytes，4 级 = 1KB（很小，不是瓶颈）
  - 如果 BLOCK_T=1024, fp32 → 每级 4KB，4 级 = 16KB（仍可接受）

  注意：当 BLOCK_T >= seq_len 时（单次循环），num_stages 几乎没有影响，
  因为没有循环迭代可以流水线化。stages 只在多次循环迭代时起作用。
""")

    B, H = args.batch_size, args.num_heads
    D = args.hidden_size // args.num_heads
    BH = B * H
    scale = D ** -0.5
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    dtype_bytes = 2 if args.dtype == "fp16" else 4

    stage_counts = [1, 2, 3, 4, 5]

    for T in args.seq_lens:
        # 使用较小的 BLOCK_T 以触发多次循环迭代（才能看到 stages 的效果）
        test_blocks = [32, 64] if T >= 128 else [triton.next_power_of_2(T)]

        for BLOCK_T in test_blocks:
            num_iters = (T + BLOCK_T - 1) // BLOCK_T
            scores = torch.randn(BH, T, T, device="cuda", dtype=dtype)
            output = torch.empty_like(scores)

            print(f"\n  --- seq_len = {T}, BLOCK_T = {BLOCK_T} ({num_iters} iterations) ---")

            smem_per_stage = BLOCK_T * dtype_bytes
            print(f"  共享内存每级: {smem_per_stage} bytes")

            print(f"  {'num_stages':>11} | {'smem (bytes)':>13} | {'Online (µs)':>12} | "
                  f"{'BW (GB/s)':>10} | {'相对':>6}")
            print(f"  {'-'*11}-+-{'-'*13}-+-{'-'*12}-+-{'-'*10}-+-{'-'*6}")

            baseline_us = None

            for ns in stage_counts:
                total_smem = ns * smem_per_stage

                try:
                    ms = benchmark_kernel(
                        _tunable_online_softmax_fwd, scores, output,
                        BH, T, scale, BLOCK_T,
                        num_warps=4, num_stages=ns,
                        warmup=args.warmup, rep=args.repeat
                    )
                    us = ms * 1000
                    bw = calc_bandwidth(T, BH, dtype_bytes, ms, 2)

                    if baseline_us is None:
                        baseline_us = us
                    relative = baseline_us / us

                    print(f"  {ns:>11} | {total_smem:>13} | {us:>12.2f} | "
                          f"{bw:>10.1f} | {relative:>6.2f}x")

                except Exception as e:
                    print(f"  {ns:>11} | {total_smem:>13} | {'FAILED':>12} | "
                          f"{'':>10} | {'':>6} | {e}")


def experiment_dtype(args):
    """实验4：fp16 vs fp32 性能对比。"""
    print("\n" + "=" * 70)
    print("  实验4：数据类型（fp16 vs fp32）性能对比")
    print("=" * 70)
    print("""
  原理：
  - fp16 (2 bytes): 带宽需求减半，RTX 4090 理论峰值 ~1008 GB/s
    → 同等带宽下可以处理 2x 数据量
  - fp32 (4 bytes): 精度更高，带宽需求翻倍
  - 如果 kernel 是 memory-bound（受带宽限制），fp16 应接近 2x 加速
  - 如果 kernel 是 compute-bound（受计算限制），fp16 加速取决于 Tensor Core 利用
""")

    B, H = args.batch_size, args.num_heads
    D = args.hidden_size // args.num_heads
    BH = B * H
    scale = D ** -0.5

    for T in args.seq_lens:
        BLOCK_T = triton.next_power_of_2(T)
        print(f"\n  --- seq_len = {T}, BLOCK_T = {BLOCK_T} ---")
        print(f"  {'dtype':>8} | {'bytes/elem':>11} | {'Online (µs)':>12} | "
              f"{'BW (GB/s)':>10} | {'加速比':>6}")
        print(f"  {'-'*8}-+-{'-'*11}-+-{'-'*12}-+-{'-'*10}-+-{'-'*6}")

        baseline_us = None
        for dtype, dtype_name, dbytes in [(torch.float16, "fp16", 2), (torch.float32, "fp32", 4)]:
            scores = torch.randn(BH, T, T, device="cuda", dtype=dtype)
            output = torch.empty_like(scores)

            ms = benchmark_kernel(
                _tunable_online_softmax_fwd, scores, output,
                BH, T, scale, BLOCK_T,
                num_warps=4, num_stages=2,
                warmup=args.warmup, rep=args.repeat
            )
            us = ms * 1000
            bw = calc_bandwidth(T, BH, dbytes, ms, 2)

            if baseline_us is None:
                baseline_us = us
            relative = baseline_us / us

            print(f"  {dtype_name:>8} | {dbytes:>11} | {us:>12.2f} | "
                  f"{bw:>10.1f} | {relative:>6.2f}x")


def experiment_seq_len_scaling(args):
    """实验5：不同 seq_len 下的 kernel 缩放行为。"""
    print("\n" + "=" * 70)
    print("  实验5：序列长度缩放分析")
    print("=" * 70)
    print("""
  原理：
  - kernel 总工作量 = B*H*T 行，每行处理 T 列 → O(T²) 的数据量
  - grid 大小 = B*H*T → 更长序列 = 更多 program = 更好的 SM 利用率
  - 当 BLOCK_T = next_power_of_2(T) 时：
    T=64  → BLOCK_T=64,  单次循环
    T=128 → BLOCK_T=128, 单次循环
    T=256 → BLOCK_T=256, 单次循环（可能寄存器压力增大）
    T=512 → BLOCK_T=512, 单次循环（更大寄存器压力）
  - 问题：T 增大后，是否应该用更小的 BLOCK_T + 多次循环？
""")

    B, H = args.batch_size, args.num_heads
    D = args.hidden_size // args.num_heads
    BH = B * H
    scale = D ** -0.5
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    dtype_bytes = 2 if args.dtype == "fp16" else 4

    seq_lens = [64, 128, 256, 512, 1024]

    print(f"\n  策略A：BLOCK_T = next_power_of_2(T)（当前默认）")
    print(f"  {'seq_len':>8} | {'BLOCK_T':>8} | {'grid_size':>10} | {'Online (µs)':>12} | "
          f"{'BW (GB/s)':>10} | {'µs/row':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")

    for T in seq_lens:
        BLOCK_T = triton.next_power_of_2(T)
        scores = torch.randn(BH, T, T, device="cuda", dtype=dtype)
        output = torch.empty_like(scores)

        try:
            ms = benchmark_kernel(
                _tunable_online_softmax_fwd, scores, output,
                BH, T, scale, BLOCK_T,
                num_warps=4, num_stages=2,
                warmup=args.warmup, rep=args.repeat
            )
            us = ms * 1000
            bw = calc_bandwidth(T, BH, dtype_bytes, ms, 2)
            us_per_row = us / (BH * T)

            print(f"  {T:>8} | {BLOCK_T:>8} | {BH*T:>10} | {us:>12.2f} | "
                  f"{bw:>10.1f} | {us_per_row:>8.4f}")
        except Exception as e:
            print(f"  {T:>8} | {BLOCK_T:>8} | {BH*T:>10} | {'FAILED':>12} | {e}")

    print(f"\n  策略B：固定 BLOCK_T=128，大 seq_len 用多次循环")
    print(f"  {'seq_len':>8} | {'BLOCK_T':>8} | {'iterations':>10} | {'Online (µs)':>12} | "
          f"{'BW (GB/s)':>10} | {'µs/row':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")

    BLOCK_T_FIXED = 128
    for T in seq_lens:
        num_iters = (T + BLOCK_T_FIXED - 1) // BLOCK_T_FIXED
        scores = torch.randn(BH, T, T, device="cuda", dtype=dtype)
        output = torch.empty_like(scores)

        try:
            ms = benchmark_kernel(
                _tunable_online_softmax_fwd, scores, output,
                BH, T, scale, BLOCK_T_FIXED,
                num_warps=4, num_stages=2,
                warmup=args.warmup, rep=args.repeat
            )
            us = ms * 1000
            bw = calc_bandwidth(T, BH, dtype_bytes, ms, 2)
            us_per_row = us / (BH * T)

            print(f"  {T:>8} | {BLOCK_T_FIXED:>8} | {num_iters:>10} | {us:>12.2f} | "
                  f"{bw:>10.1f} | {us_per_row:>8.4f}")
        except Exception as e:
            print(f"  {T:>8} | {BLOCK_T_FIXED:>8} | {num_iters:>10} | {'FAILED':>12} | {e}")


def experiment_combined_sweep(args):
    """实验6：联合参数扫描 — 找到全局最优配置。"""
    print("\n" + "=" * 70)
    print("  实验6：联合参数扫描（BLOCK_T × num_warps × num_stages）")
    print("=" * 70)
    print("""
  同时扫描所有参数的交叉组合，找到全局最优。
  这模拟了 @triton.autotune 在做的事情，但我们手动控制并记录每个组合。
""")

    B, H = args.batch_size, args.num_heads
    D = args.hidden_size // args.num_heads
    BH = B * H
    scale = D ** -0.5
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    block_sizes = [32, 64, 128, 256] if not args.quick else [64, 128]
    warp_counts = [2, 4, 8] if not args.quick else [4, 8]
    stage_counts = [1, 2, 3] if not args.quick else [2, 3]

    for T in args.seq_lens:
        scores = torch.randn(BH, T, T, device="cuda", dtype=dtype)
        output = torch.empty_like(scores)

        print(f"\n  --- seq_len = {T} ---")
        results = []

        total = len(block_sizes) * len(warp_counts) * len(stage_counts)
        print(f"  搜索空间: {total} 种配置\n")
        print(f"  {'BLOCK_T':>8} {'warps':>6} {'stages':>7} | {'µs':>10} | {'BW GB/s':>9} | 排名")
        print(f"  {'-'*8} {'-'*6} {'-'*7}-+-{'-'*10}-+-{'-'*9}-+----")

        for bt, nw, ns in product(block_sizes, warp_counts, stage_counts):
            try:
                ms = benchmark_kernel(
                    _tunable_online_softmax_fwd, scores, output,
                    BH, T, scale, bt,
                    num_warps=nw, num_stages=ns,
                    warmup=args.warmup, rep=args.repeat
                )
                us = ms * 1000
                results.append((bt, nw, ns, us))
            except Exception:
                pass

        results.sort(key=lambda x: x[3])

        for rank, (bt, nw, ns, us) in enumerate(results, 1):
            dtype_bytes = 2 if args.dtype == "fp16" else 4
            bw = calc_bandwidth(T, BH, dtype_bytes, us / 1000, 2)
            marker = " ← 最优" if rank == 1 else ""
            if rank <= 5 or rank == len(results):
                print(f"  {bt:>8} {nw:>6} {ns:>7} | {us:>10.2f} | {bw:>9.1f} | #{rank}{marker}")
            elif rank == 6:
                print(f"  {'...':>8} {'':>6} {'':>7} | {'...':>10} | {'...':>9} | ...")

        if results:
            best = results[0]
            worst = results[-1]
            print(f"\n  最优: BLOCK_T={best[0]}, num_warps={best[1]}, num_stages={best[2]} → {best[3]:.2f} µs")
            print(f"  最差: BLOCK_T={worst[0]}, num_warps={worst[1]}, num_stages={worst[2]} → {worst[3]:.2f} µs")
            print(f"  最优/最差比: {worst[3]/best[3]:.2f}x")


def experiment_autotune(args):
    """实验7：使用 @triton.autotune 自动搜索最优配置。"""
    print("\n" + "=" * 70)
    print("  实验7：@triton.autotune 自动搜索")
    print("=" * 70)
    print("""
  Triton 内置的 @triton.autotune 装饰器会：
  1. 在给定的配置列表中逐一尝试每种 (BLOCK_T, num_warps, num_stages) 组合
  2. 用 do_bench 测量每种配置的执行时间
  3. 缓存并选择最快的配置（对同一 key 值只搜索一次）

  与手动扫描的区别：
  - 自动搜索由 Triton runtime 驱动，结果可缓存
  - 搜索结果可通过 kernel.best_config 查看
""")

    B, H = args.batch_size, args.num_heads
    D = args.hidden_size // args.num_heads
    BH = B * H
    scale = D ** -0.5
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    for T in args.seq_lens:
        scores = torch.randn(BH, T, T, device="cuda", dtype=dtype)
        output = torch.empty_like(scores)
        grid = (BH * T,)

        print(f"\n  --- seq_len = {T} --- 搜索 {len(get_autotune_configs())} 种配置...")

        # 清除之前的缓存（如有）
        _autotuned_online_softmax_fwd.cache.clear()

        # 第一次调用触发搜索
        _autotuned_online_softmax_fwd[grid](scores, output, T, scale)

        # 获取最优配置
        best_config = _autotuned_online_softmax_fwd.best_config
        print(f"  Autotune 选择的最优配置:")
        print(f"    BLOCK_T    = {best_config.kwargs.get('BLOCK_T', 'N/A')}")
        print(f"    num_warps  = {best_config.num_warps}")
        print(f"    num_stages = {best_config.num_stages}")

        # 用选定配置测量最终性能
        ms = triton.testing.do_bench(
            lambda: _autotuned_online_softmax_fwd[grid](scores, output, T, scale),
            warmup=args.warmup, rep=args.repeat
        )
        us = ms * 1000
        dtype_bytes = 2 if args.dtype == "fp16" else 4
        bw = calc_bandwidth(T, BH, dtype_bytes, ms, 2)
        print(f"    执行时间   = {us:.2f} µs")
        print(f"    有效带宽   = {bw:.1f} GB/s")
        print(f"    (RTX 4090 峰值 ~1008 GB/s → 利用率 {bw/1008*100:.1f}%)")


def generate_report(args):
    """生成调优结果的 Markdown 报告。"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report_dir) / f"triton_tuning_{timestamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = f"""# Triton Kernel 性能调优报告

> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> GPU: RTX 4090 (128 SMs, SM 8.9)
> Triton: {triton.__version__}
> CUDA: {torch.version.cuda}

## 硬件参数

| 参数 | RTX 4090 |
|------|---------|
| SM 数量 | 128 |
| 每 SM 寄存器数 | 65536 × 32bit |
| 每 SM 共享内存 | 100 KB |
| 每 SM 最大 warp 数 | 48 |
| 全局内存带宽 | ~1008 GB/s |
| Compute Capability | 8.9 |

## 调优参数说明

### BLOCK_T（分块大小）
- 控制每个线程块处理的列数
- 太小 → 循环次数多，向量化效率低
- 太大 → 寄存器压力大，occupancy 下降
- 当 BLOCK_T >= seq_len 时，循环只执行一次

### num_warps（warp 数量）
- 控制线程块内并行度（1 warp = 32 线程）
- 更多 warp → 更好的延迟隐藏
- 需要与 BLOCK_T 匹配（warp 过多会闲置）

### num_stages（流水线级数）
- 控制全局内存加载的 prefetch 深度
- 只在多次循环迭代时有效
- 每级需要额外共享内存

## 实验配置

| 参数 | 值 |
|------|---|
| batch_size | {args.batch_size} |
| num_heads | {args.num_heads} |
| hidden_size | {args.hidden_size} |
| head_dim | {args.hidden_size // args.num_heads} |
| dtype | {args.dtype} |
| seq_lens | {args.seq_lens} |
| warmup | {args.warmup} |
| repeat | {args.repeat} |

---

*注意：具体实验数据请参考终端输出。此报告提供实验框架和参数说明。*
*运行 `python benchmarks/tune_triton_kernel.py` 获取完整数据。*
"""
    report_path.write_text(report, encoding="utf-8")
    # 也写入 latest 版本
    latest_path = Path(args.report_dir) / "triton_tuning_latest.md"
    latest_path.write_text(report, encoding="utf-8")
    print(f"\n  报告已保存: {report_path}")
    return report_path


# =====================================================================
# 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Triton Kernel 性能调优实验")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="指定单一序列长度（默认测试多种）")
    parser.add_argument("--warmup", type=int, default=25,
                        help="triton.testing.do_bench warmup 次数")
    parser.add_argument("--repeat", type=int, default=100,
                        help="triton.testing.do_bench 重复次数")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：缩小搜索空间")
    parser.add_argument("--autotune", action="store_true",
                        help="运行 Triton autotune 实验")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "block", "warps", "stages", "dtype",
                                 "scaling", "sweep", "autotune"],
                        help="选择要运行的实验")
    parser.add_argument("--report_dir", type=str, default="reports")
    args = parser.parse_args()

    if args.seq_len:
        args.seq_lens = [args.seq_len]
    elif args.quick:
        args.seq_lens = [128, 256]
    else:
        args.seq_lens = [128, 256, 512]

    print("=" * 70)
    print("  Triton Kernel 性能调优实验")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Triton: {triton.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  B={args.batch_size}, H={args.num_heads}, D={args.hidden_size // args.num_heads}")
    print(f"  dtype: {args.dtype}")
    print(f"  seq_lens: {args.seq_lens}")
    print(f"  模式: {'快速' if args.quick else '完整'}")

    exp = args.experiment

    if exp in ("all", "block"):
        experiment_block_size(args)
    if exp in ("all", "warps"):
        experiment_num_warps(args)
    if exp in ("all", "stages"):
        experiment_num_stages(args)
    if exp in ("all", "dtype"):
        experiment_dtype(args)
    if exp in ("all", "scaling"):
        experiment_seq_len_scaling(args)
    if exp in ("all", "sweep"):
        experiment_combined_sweep(args)
    if exp in ("all", "autotune") or args.autotune:
        experiment_autotune(args)

    generate_report(args)

    print("\n" + "=" * 70)
    print("  调优实验完成！")
    print("=" * 70)
    print("""
  下一步建议：
  1. 对比各实验结果，找到最优参数组合
  2. 将最优参数回写到 models/triton_attention.py
  3. 使用 Nsight Compute 深入分析瓶颈：
     ncu --set full python benchmarks/tune_triton_kernel.py --experiment block --seq_len 128
  4. 关注以下 Nsight Compute 指标：
     - sm__warps_active.avg.pct_of_peak_sustained_active (occupancy)
     - dram__bytes.sum.per_second (内存带宽)
     - sm__instruction_throughput.avg.pct_of_peak_sustained_active (计算吞吐)
""")


if __name__ == "__main__":
    main()
