# MLIR Attention Fusion 实验

> 本目录包含 attention 子操作在 MLIR 层面的表示与融合 Pass 设计。

## 文件说明

| 文件 | 说明 |
|------|------|
| `attention_unfused.mlir` | 融合前：scale、mask、softmax 各为独立 `linalg.generic` 操作 |
| `attention_fused.mlir` | 融合后：合并为 `custom.fused_scaled_masked_softmax` 单一操作 |
| `attention_fusion_pass.mlir` | Fusion Pass 设计文档（模式匹配 + 替换规则 + lowering 路径） |

## 核心思路

### 问题（来自第一阶段 Profiling）

在 baseline 版本中，attention 的中间步骤被拆分为多个独立 CUDA kernel：

```
[QK^T matmul] → [scale kernel] → [mask kernel] → [softmax-max] → [softmax-exp-sum] → [softmax-norm] → [PV matmul]
                 ↑________________________________↑
                        可融合区域（5 个 kernel → 1 个）
```

每个 kernel 之间存在：
- **launch overhead**：CPU→GPU 调度延迟（~5μs/次）
- **全局内存读写**：中间 tensor 写入 → 重新读取（~48KB/次 × 4 次）

### 解决方案

设计 `AttentionFusionPass`，将 5 个操作融合为 1 个：

```
融合前（5 个 linalg.generic）         融合后（1 个 custom op）
┌──────────────────────┐           ┌──────────────────────────────────────┐
│ scale (mulf)         │           │                                      │
│     ↓                │           │  custom.fused_scaled_masked_softmax  │
│ mask (select)        │    →→→    │                                      │
│     ↓                │           │  内部使用 online softmax 算法         │
│ max (reduction)      │           │  中间结果在寄存器/shared memory       │
│     ↓                │           │                                      │
│ exp+sum (map+reduce) │           └──────────────────────────────────────┘
│     ↓                │
│ normalize (divf)     │
└──────────────────────┘
```

### 从 FX Graph 到 MLIR 的映射

```
PyTorch FX Node          →    MLIR Operation
─────────────────────────────────────────────
aten.matmul              →    linalg.batch_matmul
aten.mul (scale)         →    linalg.generic { arith.mulf }
aten.masked_fill         →    linalg.generic { arith.select }
aten._softmax            →    linalg.generic { math.exp, arith.divf }
aten.layer_norm          →    linalg.generic { mean + var + norm }
aten.gelu                →    linalg.generic { math.erf + arith.mulf }
aten.add (residual)      →    linalg.generic { arith.addf }
```

### Lowering 路径

```
custom.fused_scaled_masked_softmax
        │
        ├─→ Triton Lowering
        │     └─→ _fused_scale_mask_softmax_fwd kernel
        │          (见 models/triton_attention.py)
        │
        ├─→ CUDA C++ Lowering
        │     └─→ 手写 CUDA kernel（shared memory + online softmax）
        │
        └─→ CPU Lowering（测试）
              └─→ 标量循环展开
```

## 验证结果

| 指标 | 融合前 (baseline) | 融合后 (Triton) | 变化 |
|------|:--:|:--:|:--:|
| Attention 子操作 kernel 数 | 5 | 1 | -80% |
| 数值最大误差 (fp16) | — | 4.88e-4 | ✅ 可接受 |

完整的性能对比数据见 `reports/trace_analysis_latest.md`。

## 关联文件

- `models/triton_attention.py` — Triton 融合 kernel 实现（lowering 目标）
- `benchmarks/export_fx_graph.py` — FX Graph 导出（融合前的计算图分析）
- `benchmarks/profile_triton.py` — 融合后的 profiling
- `benchmarks/analyze_trace.py` — 融合前后性能对比
