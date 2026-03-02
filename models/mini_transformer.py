"""
最小 Transformer Block —— 用于注意力机制性能分析实验。

本模块提供单层、可配置的 Transformer Block，
专为生成清晰的 CUDA kernel trace 而设计，便于 profiling 与编译优化分析。

设计目标：
- 将每个子操作（QKV matmul、softmax、scale、mask）暴露为独立 kernel，
  方便 profiling 识别可融合的机会。
- 同时支持手写 attention 和 PyTorch SDPA（FlashAttention），用于 A/B 对比。
- 保持代码最小化、自包含。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """MiniTransformerBlock 的配置参数。"""

    hidden_size: int = 768
    num_heads: int = 12
    seq_len: int = 128
    batch_size: int = 1
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    use_causal_mask: bool = True
    dropout: float = 0.0


class ManualAttention(nn.Module):
    """
    手写缩放点积注意力。

    故意不做融合，使每个子操作（scale、mask、softmax、dropout、matmul）
    在 profiler timeline 中表现为独立的 CUDA kernel。
    这是编译器融合应当改进的"基线"版本。
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.dropout
        self.use_causal_mask = config.use_causal_mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: (B, H, T, D)
        scale = self.head_dim ** -0.5

        # 步骤 1：QK^T（矩阵乘法 kernel）
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)

        # 步骤 2：缩放（逐元素 kernel）
        attn_scores = attn_scores * scale

        # 步骤 3：因果遮罩（逐元素 kernel）
        if self.use_causal_mask:
            T = q.size(-2)
            mask = torch.triu(
                torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # 步骤 4：softmax（归约 kernel）
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 步骤 5：dropout（逐元素 kernel，仅训练时启用）
        if self.dropout > 0.0 and self.training:
            attn_probs = F.dropout(attn_probs, p=self.dropout)

        # 步骤 6：PV 矩阵乘法（matmul kernel）
        attn_output = torch.matmul(attn_probs, v)  # (B, H, T, D)

        return attn_output


class SDPAttention(nn.Module):
    """
    PyTorch scaled_dot_product_attention (SDPA) 封装。

    在可用时使用融合的 FlashAttention / 内存高效后端。
    作为"已优化"的参考版本用于对比。
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dropout = config.dropout
        self.use_causal_mask = config.use_causal_mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.use_causal_mask,
        )


class MiniTransformerBlock(nn.Module):
    """
    单层 Transformer Block：LayerNorm → Attention → 残差 → LayerNorm → MLP → 残差。

    参数
    ----------
    config : TransformerConfig
        模型 / profiling 配置。
    use_sdpa : bool
        若为 True，使用 PyTorch SDPA（融合注意力）。
        若为 False，使用手写 ManualAttention（未融合，用于 profiling）。
    """

    def __init__(self, config: Optional[TransformerConfig] = None, use_sdpa: bool = False):
        super().__init__()
        if config is None:
            config = TransformerConfig()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        # ---------- 注意力层 ----------
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        if use_sdpa:
            self.attn_fn = SDPAttention(config)
        else:
            self.attn_fn = ManualAttention(config)

        # ---------- 前馈网络 ----------
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False)
        self.fc2 = nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False)

        # ---------- 层归一化 ----------
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # ===== 自注意力 =====
        residual = x
        x = self.ln1(x)

        qkv = self.qkv(x)  # (B, T, 3C)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # 每个 (B, T, H, D)

        # 转置为 (B, H, T, D) 以进行注意力计算
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = self.attn_fn(q, k, v)  # (B, H, T, D)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        x = self.out_proj(attn_output)
        x = x + residual

        # ===== 前馈网络 MLP =====
        residual = x
        x = self.ln2(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x + residual

        return x


# ---------------------------------------------------------------------------
# 快速冒烟测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = TransformerConfig()
    model = MiniTransformerBlock(cfg, use_sdpa=False).to(cfg.device).to(cfg.dtype)
    model.eval()
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size,
                     device=cfg.device, dtype=cfg.dtype)
    with torch.no_grad():
        y = model(x)
    print(f"输入:  {x.shape}")
    print(f"输出: {y.shape}")
    print("✅ MiniTransformerBlock 冒烟测试通过。")
