"""
Triton 融合注意力 Kernel
========================
手写 Triton fused attention kernel，将 scale + causal mask + softmax 融合为
单个 GPU kernel，消除中间结果的全局内存读写和多次 kernel launch 开销。

这是第二阶段的核心实验：用 Triton DSL 验证编译优化假设
（即第一阶段 profiling 发现的 attention 子操作碎片化问题）。

提供三种实现：
1. TritonAttention — 三遍扫描的基础融合（scale + mask + softmax）
2. OnlineTritonAttention — Online Softmax 两遍扫描版本（减少一次全局内存加载）
3. 两者均保留 QK^T 和 PV matmul 使用 cublas，只替换中间的 softmax 部分

用法：
    from models.triton_attention import TritonAttention, OnlineTritonAttention
    attn = TritonAttention(config)        # 三遍版本
    attn = OnlineTritonAttention(config)  # Online Softmax 两遍版本
    out = attn(q, k, v)   # (B, H, T, D)
"""

import torch
import torch.nn as nn

import triton
import triton.language as tl

from dataclasses import dataclass
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =====================================================================
# Triton Kernel：融合 scale + causal mask + softmax
# =====================================================================

@triton.jit
def _fused_scale_mask_softmax_fwd(
    # 指针
    scores_ptr,     # 输入：原始 attention scores (B*H, T, T)
    output_ptr,     # 输出：softmax 结果 (B*H, T, T)
    # 维度
    seq_len,        # T
    # 参数
    scale: tl.constexpr,          # 1/sqrt(head_dim)
    # Block 大小
    BLOCK_T: tl.constexpr,        # 分块大小（沿 T 维度）
):
    """
    对 attention scores 执行融合的 scale + causal mask + softmax。
    每个 program 处理一行 (batch_head_idx, row_idx)。
    """
    # program id: 每个 instance 处理一行
    pid = tl.program_id(0)
    batch_head_idx = pid // seq_len
    row_idx = pid % seq_len

    # 基地址偏移
    row_offset = batch_head_idx * seq_len * seq_len + row_idx * seq_len

    # 分块处理列
    col_offsets = tl.arange(0, BLOCK_T)

    # ---- 第一遍：找最大值（数值稳定性） ----
    max_val = float("-inf")
    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        # 因果遮罩：只保留 col <= row 的元素
        causal_mask = cols <= row_idx

        # 加载 scores
        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)

        # 融合操作 1：scale
        x = x * scale

        # 融合操作 2：causal mask（将未来位置设为 -inf）
        x = tl.where(causal_mask & mask, x, float("-inf"))

        # 更新 running max
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # ---- 第二遍：计算 exp 和 sum ----
    sum_exp = 0.0
    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        causal_mask = cols <= row_idx

        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(causal_mask & mask, x, float("-inf"))

        # softmax: exp(x - max)
        x = tl.exp(x - max_val)
        # 屏蔽无效位置
        x = tl.where(causal_mask & mask, x, 0.0)
        sum_exp += tl.sum(x, axis=0)

    # ---- 第三遍：归一化并写出 ----
    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        causal_mask = cols <= row_idx

        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(causal_mask & mask, x, float("-inf"))

        x = tl.exp(x - max_val)
        x = tl.where(causal_mask & mask, x, 0.0)

        # 归一化
        x = x / (sum_exp + 1e-6)

        tl.store(output_ptr + row_offset + cols, x, mask=mask)


def triton_fused_scale_mask_softmax(scores: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Python 封装：对 attention scores 执行融合的 scale + causal mask + softmax。

    参数：
        scores: (B, H, T, T) 或 (B*H, T, T) 的原始 attention scores
        scale:  1/sqrt(head_dim) 缩放因子

    返回：
        (B, H, T, T) softmax 概率矩阵
    """
    original_shape = scores.shape
    if scores.ndim == 4:
        B, H, T, _ = scores.shape
        scores_3d = scores.reshape(B * H, T, T)
    else:
        scores_3d = scores
        T = scores.shape[-1]

    BH = scores_3d.shape[0]
    output = torch.empty_like(scores_3d)

    # Grid: 每行一个 program (BH * T 个 program)
    grid = (BH * T,)

    # 选择 BLOCK_T: 下一个 2 的幂
    BLOCK_T = triton.next_power_of_2(T)
    BLOCK_T = max(BLOCK_T, 16)  # 最小 16

    _fused_scale_mask_softmax_fwd[grid](
        scores_3d, output,
        T,
        scale,
        BLOCK_T=BLOCK_T,
    )

    return output.reshape(original_shape)


# =====================================================================
# TritonAttention 模块（nn.Module 封装）
# =====================================================================

class TritonFusedScaleMaskSoftmax(nn.Module):
    """
    Triton 融合 scale + mask + softmax 模块。

    仅替换 ManualAttention 中的步骤 2–4（scale → mask → softmax），
    保留 QK^T 和 PV matmul 使用 PyTorch cublas。
    这样可以精确测量融合这三个子操作的收益。
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = head_dim ** -0.5

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        参数：
            scores: (B, H, T, T) 原始 QK^T scores（未缩放）

        返回：
            (B, H, T, T) softmax 概率
        """
        return triton_fused_scale_mask_softmax(scores, self.scale)


class TritonAttention(nn.Module):
    """
    使用 Triton 融合 kernel 的注意力模块。

    计算流程：
    1. QK^T matmul（cublas）
    2. scale + causal_mask + softmax（Triton 融合 kernel）
    3. PV matmul（cublas）

    与 ManualAttention 的区别：
    - ManualAttention：步骤 2、3、4 各自是独立 CUDA kernel
    - TritonAttention：步骤 2、3、4 融合为单个 Triton kernel
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.fused_softmax = TritonFusedScaleMaskSoftmax(self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        参数：
            q, k, v: (B, H, T, D)

        返回：
            (B, H, T, D) attention 输出
        """
        # 步骤 1：QK^T（使用 cublas 矩阵乘法）
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)

        # 步骤 2-4：融合 scale + causal mask + softmax（Triton kernel）
        attn_probs = self.fused_softmax(attn_scores)

        # 步骤 5：PV matmul（使用 cublas 矩阵乘法）
        attn_output = torch.matmul(attn_probs, v)  # (B, H, T, D)

        return attn_output


# =====================================================================
# Triton Kernel：Online Softmax（两遍扫描）
# =====================================================================
# 经典三遍 vs Online 两遍对比：
#
#   三遍版本 (_fused_scale_mask_softmax_fwd):
#     第1遍: load → scale → mask → 求 max
#     第2遍: load → scale → mask → exp(x-max) → 求 sum
#     第3遍: load → scale → mask → exp(x-max) → /sum → store
#     全局内存加载: 3×T
#
#   Online 两遍版本 (_online_softmax_fwd):
#     第1遍: load → scale → mask → 同时维护 running max 和 running sum
#     第2遍: load → scale → mask → exp(x-max) → /sum → store
#     全局内存加载: 2×T（减少 33%）
#
#   关键技巧：当遇到新 max' > max 时，修正已累积的 sum：
#     sum_new = sum_old × exp(max_old − max_new) + exp(x − max_new)

@triton.jit
def _online_softmax_fwd(
    # 指针
    scores_ptr,     # 输入：原始 attention scores (B*H, T, T)
    output_ptr,     # 输出：softmax 结果 (B*H, T, T)
    # 维度
    seq_len,        # T
    # 参数
    scale: tl.constexpr,          # 1/sqrt(head_dim)
    # Block 大小
    BLOCK_T: tl.constexpr,        # 分块大小（沿 T 维度）
):
    """
    Online Softmax 融合 kernel（两遍扫描）。

    使用 Milakov & Gimelshein (2018) 的 Online Softmax 算法：
      第1遍：同时计算 running max 和 running sum(exp)
      第2遍：归一化输出
    相比三遍版本减少 1 次全局内存加载。
    """
    # program id: 每个 instance 处理一行
    pid = tl.program_id(0)
    batch_head_idx = pid // seq_len
    row_idx = pid % seq_len

    # 基地址偏移
    row_offset = batch_head_idx * seq_len * seq_len + row_idx * seq_len

    # 分块处理列
    col_offsets = tl.arange(0, BLOCK_T)

    # ---- 第一遍：Online Softmax —— 同时维护 running max 和 running sum ----
    max_val = float("-inf")
    sum_exp = 0.0

    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        causal_mask = cols <= row_idx

        # 加载 + scale + causal mask（三个操作融合在寄存器中）
        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(causal_mask & mask, x, float("-inf"))

        # Online 更新：先算本块最大值
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)

        # 修正已有的 sum_exp：旧的累积值要乘以 exp(old_max - new_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max)

        # 累加本块的 exp(x - new_max)
        exp_x = tl.exp(x - new_max)
        exp_x = tl.where(causal_mask & mask, exp_x, 0.0)
        sum_exp += tl.sum(exp_x, axis=0)

        # 更新 max
        max_val = new_max

    # ---- 第二遍：归一化并写出 ----
    for col_start in range(0, seq_len, BLOCK_T):
        cols = col_start + col_offsets
        mask = cols < seq_len
        causal_mask = cols <= row_idx

        x = tl.load(scores_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(causal_mask & mask, x, float("-inf"))

        x = tl.exp(x - max_val)
        x = tl.where(causal_mask & mask, x, 0.0)

        # 归一化
        x = x / (sum_exp + 1e-6)

        tl.store(output_ptr + row_offset + cols, x, mask=mask)


def triton_online_softmax(scores: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Python 封装：使用 Online Softmax 算法的融合 scale + mask + softmax。

    与 triton_fused_scale_mask_softmax 功能相同，但全局内存加载减少 33%。

    参数：
        scores: (B, H, T, T) 或 (B*H, T, T) 的原始 attention scores
        scale:  1/sqrt(head_dim) 缩放因子

    返回：
        (B, H, T, T) softmax 概率矩阵
    """
    original_shape = scores.shape
    if scores.ndim == 4:
        B, H, T, _ = scores.shape
        scores_3d = scores.reshape(B * H, T, T)
    else:
        scores_3d = scores
        T = scores.shape[-1]

    BH = scores_3d.shape[0]
    output = torch.empty_like(scores_3d)

    grid = (BH * T,)

    BLOCK_T = triton.next_power_of_2(T)
    BLOCK_T = max(BLOCK_T, 16)

    _online_softmax_fwd[grid](
        scores_3d, output,
        T,
        scale,
        BLOCK_T=BLOCK_T,
    )

    return output.reshape(original_shape)


# =====================================================================
# OnlineTritonAttention 模块
# =====================================================================

class OnlineFusedScaleMaskSoftmax(nn.Module):
    """
    Online Softmax 版本的融合 scale + mask + softmax。

    与 TritonFusedScaleMaskSoftmax 相比：
    - 全局内存加载从 3×T 减少到 2×T
    - 使用 running max + running sum 技巧合并第1遍和第2遍
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = head_dim ** -0.5

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return triton_online_softmax(scores, self.scale)


class OnlineTritonAttention(nn.Module):
    """
    使用 Online Softmax 的 Triton 注意力模块。

    计算流程与 TritonAttention 相同：
    1. QK^T matmul（cublas）
    2. scale + causal_mask + softmax（Online Softmax Triton kernel，两遍扫描）
    3. PV matmul（cublas）

    三种 Attention 实现对比：
    ┌──────────────────────┬───────────┬────────────────────────────┐
    │ 实现                 │ 全局 load │ softmax 遍数               │
    ├──────────────────────┼───────────┼────────────────────────────┤
    │ ManualAttention      │ 各 1 次   │ 3 个独立 kernel            │
    │ TritonAttention      │ 3×T       │ 1 kernel, 三遍扫描         │
    │ OnlineTritonAttention│ 2×T       │ 1 kernel, 两遍扫描（本版本）│
    └──────────────────────┴───────────┴────────────────────────────┘
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.fused_softmax = OnlineFusedScaleMaskSoftmax(self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        参数：
            q, k, v: (B, H, T, D)

        返回：
            (B, H, T, D) attention 输出
        """
        # 步骤 1：QK^T
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # 步骤 2-4：Online Softmax 融合 kernel
        attn_probs = self.fused_softmax(attn_scores)

        # 步骤 5：PV matmul
        attn_output = torch.matmul(attn_probs, v)

        return attn_output


# =====================================================================
# 正确性验证
# =====================================================================

def verify_correctness():
    """对比 Triton 融合实现（三遍 + Online 两遍）与 PyTorch 原生实现的数值一致性。"""
    print("=" * 60)
    print("  🧪 Triton 融合 Attention 正确性验证")
    print("=" * 60)

    torch.manual_seed(42)

    B, H, T, D = 2, 12, 128, 64
    device = "cuda"
    dtype = torch.float16

    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)

    scale = D ** -0.5

    # ---- PyTorch 参考实现 ----
    scores_ref = torch.matmul(q, k.transpose(-2, -1))
    scores_ref = scores_ref * scale
    mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    scores_ref = scores_ref.masked_fill(mask, float("-inf"))
    probs_ref = torch.softmax(scores_ref, dim=-1)
    out_ref = torch.matmul(probs_ref, v)

    # ---- Triton 三遍融合实现 ----
    scores_triton = torch.matmul(q, k.transpose(-2, -1))
    probs_triton = triton_fused_scale_mask_softmax(scores_triton, scale)
    out_triton = torch.matmul(probs_triton, v)

    # ---- Triton Online Softmax 两遍实现 ----
    scores_online = torch.matmul(q, k.transpose(-2, -1))
    probs_online = triton_online_softmax(scores_online, scale)
    out_online = torch.matmul(probs_online, v)

    rtol = 1e-2
    atol = 1e-2
    all_pass = True

    # ---- 对比 1: 三遍版本 vs PyTorch ----
    print(f"\n  📌 三遍 Triton vs PyTorch：")
    prob_diff = (probs_ref - probs_triton).abs()
    out_diff = (out_ref - out_triton).abs()
    print(f"    Softmax  最大误差: {prob_diff.max().item():.6e}  均值: {prob_diff.mean().item():.6e}")
    print(f"    输出     最大误差: {out_diff.max().item():.6e}  均值: {out_diff.mean().item():.6e}")
    ok1 = torch.allclose(out_ref, out_triton, rtol=rtol, atol=atol)
    print(f"    {'✅ 通过' if ok1 else '❌ 失败'}")
    all_pass &= ok1

    # ---- 对比 2: Online Softmax 两遍版本 vs PyTorch ----
    print(f"\n  📌 Online Softmax（两遍）vs PyTorch：")
    prob_diff2 = (probs_ref - probs_online).abs()
    out_diff2 = (out_ref - out_online).abs()
    print(f"    Softmax  最大误差: {prob_diff2.max().item():.6e}  均值: {prob_diff2.mean().item():.6e}")
    print(f"    输出     最大误差: {out_diff2.max().item():.6e}  均值: {out_diff2.mean().item():.6e}")
    ok2 = torch.allclose(out_ref, out_online, rtol=rtol, atol=atol)
    print(f"    {'✅ 通过' if ok2 else '❌ 失败'}")
    all_pass &= ok2

    # ---- 对比 3: 三遍 vs Online Softmax（互相一致性） ----
    print(f"\n  📌 三遍 vs Online Softmax（一致性）：")
    cross_diff = (probs_triton - probs_online).abs()
    print(f"    Softmax  最大误差: {cross_diff.max().item():.6e}  均值: {cross_diff.mean().item():.6e}")
    ok3 = torch.allclose(probs_triton, probs_online, rtol=rtol, atol=atol)
    print(f"    {'✅ 通过' if ok3 else '❌ 失败'}")
    all_pass &= ok3

    print(f"\n  {'✅ 全部正确性验证通过' if all_pass else '❌ 存在验证失败'}（rtol={rtol}, atol={atol}）")

    return all_pass


if __name__ == "__main__":
    verify_correctness()
