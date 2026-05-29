"""
Nsight Compute 专用的 Triton kernel profiling 脚本。

用法：
    /usr/local/cuda-11.7/nsight-compute-2022.2.0/ncu \
        --section SpeedOfLight \
        --section ComputeWorkloadAnalysis \
        --section MemoryWorkloadAnalysis \
        --section Occupancy \
        --section SchedulerStatistics \
        --section WarpStateStatistics \
        --section LaunchStatistics \
        -k _tunable --launch-count 3 \
        python benchmarks/ncu_profile_kernel.py

    # 输出到文件后用 ncu-ui 查看：
    /usr/local/cuda-11.7/nsight-compute-2022.2.0/ncu \
        --set full -o reports/ncu_softmax \
        -k _tunable --launch-count 3 \
        python benchmarks/ncu_profile_kernel.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import triton
import triton.language as tl


@triton.jit
def _tunable_online_softmax_fwd(
    scores_ptr, output_ptr,
    seq_len,
    scale: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
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


def main():
    T = 128
    B, H, D = 1, 12, 64
    BH = B * H
    scale = D ** -0.5
    dtype = torch.float16

    scores = torch.randn(BH, T, T, device="cuda", dtype=dtype)
    output = torch.empty_like(scores)
    BLOCK_T = triton.next_power_of_2(T)
    grid = (BH * T,)

    # Warmup (JIT compile)
    _tunable_online_softmax_fwd[grid](scores, output, T, scale, BLOCK_T=BLOCK_T, num_warps=2, num_stages=1)
    torch.cuda.synchronize()

    # Profiled runs — ncu will capture these
    for _ in range(3):
        _tunable_online_softmax_fwd[grid](
            scores, output, T, scale,
            BLOCK_T=BLOCK_T, num_warps=2, num_stages=1,
        )
    torch.cuda.synchronize()
    print("Done. ncu should have captured kernel launches.")


if __name__ == "__main__":
    main()
