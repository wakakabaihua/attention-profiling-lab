# MLIR Attention Fusion Pass 分析报告

> 生成时间: 2026-05-29 12:51:30
> 环境: PyTorch 2.12.0.dev20260301+cu126
> 模型: ScaleMaskSoftmax (B=1, H=12, T=128, D=64)

## 数据来源说明

本报告中的每个数据点均标注来源：

| 标记 | 含义 | 说明 |
|------|------|------|
| 📊 实测 | 本次程序实测 | 由本脚本真实执行 torch-mlir 导出、IR 解析、融合 Pass 产生 |
| 📂 Stage1 | Stage 1 GPU 实测 | 来自 `traces/*.json` 中 PyTorch Profiler 的真实 GPU kernel 数据 |
| 📐 IR推导 | 从 IR 逻辑推导 | 基于 MLIR IR 结构推导（如中间 tensor 数 = 核心操作数 - 1） |
| ⚠️ 估算 | 理论计算 | 基于 tensor 形状和 GPU 架构参数估算，非实测 |

## 实验概述

本实验是 attention-profiling-lab 第三阶段（MLIR 融合 Pass），
展示编译器如何在 MLIR IR 层面自动识别并融合 attention 子操作。

## Torch Dialect 分析

### 操作清单

| # | MLIR Operation | 分类 | 融合 |
|---|----------------|------|------|
| 0 | `torch.constant.float` | constant |  |
| 1 | `torch.aten.mul.Scalar` | scale | 🟡 |
| 2 | `torch.constant.int` | constant |  |
| 3 | `torch.constant.int` | constant |  |
| 4 | `torch.prim.ListConstruct` | auxiliary |  |
| 5 | `torch.constant.int` | constant |  |
| 6 | `torch.constant.none` | constant |  |
| 7 | `torch.constant.device` | constant |  |
| 8 | `torch.constant.bool` | constant |  |
| 9 | `torch.aten.ones` | mask_gen |  |
| 10 | `torch.constant.int` | constant |  |
| 11 | `torch.constant.none` | constant |  |
| 12 | `torch.constant.none` | constant |  |
| 13 | `torch.constant.device` | constant |  |
| 14 | `torch.constant.bool` | constant |  |
| 15 | `torch.aten.arange` | mask_gen |  |
| 16 | `torch.constant.int` | constant |  |
| 17 | `torch.aten.unsqueeze` | mask_gen |  |
| 18 | `torch.constant.int` | constant |  |
| 19 | `torch.constant.none` | constant |  |
| 20 | `torch.constant.none` | constant |  |
| 21 | `torch.constant.device` | constant |  |
| 22 | `torch.constant.bool` | constant |  |
| 23 | `torch.aten.arange` | mask_gen |  |
| 24 | `torch.constant.int` | constant |  |
| 25 | `torch.aten.unsqueeze` | mask_gen |  |
| 26 | `torch.constant.int` | constant |  |
| 27 | `torch.aten.sub.Tensor` | mask_gen |  |
| 28 | `torch.constant.int` | constant |  |
| 29 | `torch.aten.ge.Scalar` | mask_gen |  |
| 30 | `torch.aten.logical_and` | mask_gen |  |
| 31 | `torch.constant.float` | constant |  |
| 32 | `torch.aten.where.ScalarSelf` | mask_apply | 🟡 |
| 33 | `torch.constant.int` | constant |  |
| 34 | `torch.constant.none` | constant |  |
| 35 | `torch.aten.softmax.int` | softmax | 🟡 |
| 36 | `return` | control |  |

### 融合模式匹配结果

```
  %arg0 (scores)
      │
      ▼
  [torch.aten.mul.Scalar]  ← scale = 1.250000e-01
      │
      ▼
  [torch.aten.where.ScalarSelf]  ← causal mask (triu → -inf)
      │
      ▼
  [torch.aten.softmax.int]  ← softmax(dim=-1)
      │
      ▼
  %11 (probs)
```

- 核心操作: 3 个 (scale + mask + softmax)
- 辅助操作: 33 个 (mask 生成 + 常量)
- 总计消除: 36 个操作 → 1 个融合操作

### 融合后 IR

```mlir
%11 = "custom.fused_scaled_masked_softmax"(%arg0, 1.250000e-01) {
    softmax_dim = -1 : i64,
    is_causal = true,
    fusion_source = "attention_fusion_pass_v1"
}
```

## Linalg Dialect 分析

| # | 内部操作 | 迭代类型 | 分类 | 融合 |
|---|----------|----------|------|------|
| 0 | `arith.mulf` | parallel, parallel, parallel | scale | 🟡 |
| 1 | `linalg.index, arith.index_cast` | parallel | indexing |  |
| 2 | `(empty)` | parallel, parallel | other |  |
| 3 | `arith.cmpi` | parallel, parallel | comparison |  |
| 4 | `(empty)` | parallel, parallel | other |  |
| 5 | `arith.select` | parallel, parallel, parallel | mask/where | 🟡 |
| 6 | `linalg.fill` | n/a | fill |  |
| 7 | `linalg.fill` | n/a | fill |  |
| 8 | `linalg.index, arith.index_cast, arith.maximumf` | parallel, parallel, parallel | softmax_max | 🟡 |
| 9 | `arith.subf` | parallel, parallel, parallel | softmax_sub | 🟡 |
| 10 | `math.exp` | parallel, parallel, parallel | softmax_exp | 🟡 |
| 11 | `linalg.fill` | n/a | fill |  |
| 12 | `arith.addf` | parallel, parallel, parallel | softmax_sum | 🟡 |
| 13 | `arith.divf` | parallel, parallel, parallel | softmax_div | 🟡 |

- 总 linalg.generic: 14
- 可融合: 7
- 不可融合: 7
- 融合后: 8

## 融合效果汇总

| 数据来源 | 指标 | 融合前 | 融合后 | 变化 |
|----------|------|:------:|:------:|:----:|
| 📊 实测 | Torch dialect 操作数 | 37 | 2 | -95% |
| 📊 实测 | 核心计算操作 | 3 | 1 | -67% |
| 📐 IR推导 | 中间 tensor | 2 | 0 | -100% |
| 📐 IR推导 | 全局内存读写 | 4 次 | 0 次 | -100% |
| 📊 实测 | Linalg generic 数 | 14 | 8 | -43% |
| ⚠️ 估算 | 中间 tensor 内存 | 1536 KB | 0 KB | -100% |

> **📐 IR推导说明**: 中间 tensor 数 = 核心操作数(3) - 1 = 2；全局内存读写 = 中间 tensor 数 × 2 (每个需写出+重读) = 4

> **⚠️ 估算说明**: 中间 tensor 内存 = 1×12×128×128×2bytes × 2 × 2(写+读) = 1536 KB，基于 tensor 形状理论计算，未实测

## Stage 1 GPU Profiling 实测数据 (交叉验证)

> 以下数据全部来自 `traces/` 目录中的真实 GPU profiling trace

### Baseline Attention 子操作

| Kernel 类别 | 启动次数 | 耗时 (μs) | 数据来源 |
|------------|:--------:|:---------:|----------|
| softmax | 30 | 77.0 | 📂 Stage1 实测 |
| mask_triu | 30 | 56.5 | 📂 Stage1 实测 |
| mask_fill | 30 | 61.8 | 📂 Stage1 实测 |
| **合计** | **90** | **195.2** | 📂 占总 kernel 时间 11.5% |

### 全流水线对比 — Stage 1 实测

> ⚠️ **测量范围: 完整 Attention 流水线** (QK^T → scale → mask → softmax → ·V 及所有辅助 kernel)
> Triton 融合仅替换了其中 softmax 部分，matmul 等其他 kernel 不变，
> 而 softmax 子操作仅占总时间 11.5%，因此全流水线加速比较小。

| 版本 | 总 kernel 数 | 总耗时 (μs) | 加速比 | 数据来源 |
|------|:-----------:|:----------:|:------:|----------|
| Baseline (全流水线) | 540 | 1696.4 | 1.00× | 📂 Stage1 实测 |
| SDPA | 330 | 1440.5 | 1.18× | 📂 Stage1 实测 |
| Triton-3pass | 420 | 1507.5 | 1.13× | 📂 Stage1 实测 |
| Triton-Online | 420 | 1502.0 | 1.13× | 📂 Stage1 实测 |

## 多版本融合 GPU 实测验证 — 仅 ScaleMaskSoftmax 部分

> 以下数据全部来自本次实验 GPU 实测 (PyTorch Profiler)
>
> ⚠️ **测量范围: 仅 ScaleMaskSoftmax 模块** (scale → causal_mask → softmax)，
> **不包含** QK^T 和 ·V 矩阵乘法。因此加速比反映的是 **softmax 子操作本身**的融合收益，
> 而非完整 Attention 流水线的端到端加速。

### 多版本对比（仅 softmax 子操作）

| 版本 | CUDA kernel 数 | CUDA 总耗时 (μs) | μs/iter | 加速比 | 数据来源 |
|------|:--------------:|:----------------:|:-------:|:------:|----------|
| 融合前 (独立 kernel) | 282 | 1339.0 | 66.9 | 1.00× | 📊 实测 |
| MLIR 融合 (compile) | 101 | 570.1 | 28.5 | 2.35× | 📊 实测 |
| MLIR 自编译 (our pass) | 40 | 100.9 | 5.0 | 13.27× | 📊 实测 |
| Triton 3-pass | 40 | 101.6 | 5.1 | 13.17× | 📊 实测 |
| Triton Online | 40 | 100.5 | 5.0 | 13.32× | 📊 实测 |

### 融合前 (独立 kernel) — Top Kernels

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `unfused` | 20 | 936.8 | 46.8 | 📊 实测 |
| `aten::_softmax` | 20 | 51.3 | 2.6 | 📊 实测 |
| `void (anonymous namespace)::softmax_warp_forw` | 20 | 51.3 | 2.6 | 📊 实测 |
| `aten::masked_fill_` | 20 | 38.6 | 1.9 | 📊 实测 |
| `void at::native::elementwise_kernel<128, 2, a` | 20 | 38.6 | 1.9 | 📊 实测 |

### MLIR 融合 (compile) — Top Kernels

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `## Call CompiledFxGraph ffrjlhrwxgd3lebb5yttj` | 20 | 208.8 | 10.4 | 📊 实测 |
| `aten::_foreach_copy_` | 20 | 134.4 | 6.7 | 📊 实测 |
| `void at::native::(anonymous namespace)::multi` | 20 | 134.4 | 6.7 | 📊 实测 |
| `Torch-Compiled Region: 0/0` | 20 | 42.8 | 2.1 | 📊 实测 |
| `triton_per_fused__softmax_exp_masked_fill_mul` | 20 | 42.8 | 2.1 | 📊 实测 |

### MLIR 自编译 (our pass) — Top Kernels

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `_mlir_compiled_fused_softmax_kernel` | 20 | 50.4 | 2.5 | 📊 实测 |
| `mlir_compiled` | 20 | 50.4 | 2.5 | 📊 实测 |

### Triton 3-pass — Top Kernels

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `_fused_scale_mask_softmax_fwd` | 20 | 50.8 | 2.5 | 📊 实测 |
| `triton_3pass` | 20 | 50.8 | 2.5 | 📊 实测 |

### Triton Online — Top Kernels

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `_online_softmax_fwd` | 20 | 50.3 | 2.5 | 📊 实测 |
| `triton_online` | 20 | 50.3 | 2.5 | 📊 实测 |

> **说明**:
> - **四个版本是四种独立实现**，不是叠加组合。每个版本单独运行 ScaleMaskSoftmax 并测量。
> - **MLIR 融合 (compile)**: `torch.compile` 编译器自动融合，等价于 MLIR fusion pass 在 IR 层面识别的优化
> - **Triton 3-pass**: Stage 2 手写 Triton kernel（三遍扫描: max → exp+sum → div）
> - **Triton Online**: Stage 2 手写 Triton kernel（两遍在线算法: running max+sum → div）
> - 所有版本实现相同功能: scale → causal_mask → softmax
>
> **与 Stage 1 全流水线数据的区别**:
> - Stage 1 测量的是 **完整 Attention 流水线**（含 matmul），softmax 仅占 ~11.6%，所以 Triton 加速比仅 1.12×
> - 本节测量的是 **仅 ScaleMaskSoftmax 模块**，加速比反映 softmax 本身的融合收益 (13×+)
> - 两组数据不可直接对比加速比，因为基线和测量范围完全不同

## 全流水线 GPU 实测 — 完整 FullAttention

> 以下数据全部来自本次实验 GPU 实测 (PyTorch Profiler)
>
> **测量范围: 完整 Attention 流水线** (QK^T → scale → mask → softmax → ·V)，
> 与 Stage 1 trace 数据测量范围一致，加速比可直接对比。

### 六版本对比（全流水线）

| 版本 | CUDA kernel 数 | CUDA 总耗时 (μs) | μs/iter | 加速比 | 数据来源 |
|------|:--------------:|:----------------:|:-------:|:------:|----------|
| 原始 FullAttention | 341 | 2917.3 | 145.9 | 1.00× | 📊 实测 |
| MLIR 融合 (compile) | 140 | 1761.9 | 88.1 | 1.66× | 📊 实测 |
| MLIR 自编译 (our pass) | 121 | 2509.6 | 125.5 | 1.16× | 📊 实测 |
| Triton 3-pass | 121 | 2177.8 | 108.9 | 1.34× | 📊 实测 |
| Triton Online | 121 | 2222.9 | 111.1 | 1.31× | 📊 实测 |
| MLIR + Triton 3-pass | 140 | 1890.3 | 94.5 | 1.54× | 📊 实测 |
| MLIR + Triton Online | 140 | 1886.7 | 94.3 | 1.55× | 📊 实测 |
| compile + MLIR 自编译 | 140 | 1854.8 | 92.7 | 1.57× | 📊 实测 |

### 原始 FullAttention — Top Kernels (全流水线)

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `unfused` | 20 | 1617.7 | 80.9 | 📊 实测 |
| `aten::bmm` | 40 | 445.4 | 11.1 | 📊 实测 |
| `ampere_sgemm_128x128_nn` | 20 | 271.5 | 13.6 | 📊 实测 |
| `ampere_sgemm_128x128_tn` | 20 | 174.0 | 8.7 | 📊 实测 |
| `aten::_softmax` | 20 | 51.6 | 2.6 | 📊 实测 |

### MLIR 融合 (compile) — Top Kernels (全流水线)

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `## Call CompiledFxGraph f7iho6uc4bykoe6cvf2yd` | 20 | 807.4 | 40.4 | 📊 实测 |
| `aten::bmm` | 40 | 443.6 | 11.1 | 📊 实测 |
| `ampere_sgemm_128x128_nn` | 20 | 270.9 | 13.5 | 📊 实测 |
| `ampere_sgemm_128x128_tn` | 20 | 172.6 | 8.6 | 📊 实测 |
| `triton_per_fused__softmax_exp_masked_fill_mul` | 20 | 33.7 | 1.7 | 📊 实测 |

### MLIR 自编译 (our pass) — Top Kernels (全流水线)

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `mlir_compiled` | 20 | 1559.9 | 78.0 | 📊 实测 |
| `aten::bmm` | 40 | 445.0 | 11.1 | 📊 实测 |
| `ampere_sgemm_128x128_nn` | 20 | 271.2 | 13.6 | 📊 实测 |
| `ampere_sgemm_128x128_tn` | 20 | 173.8 | 8.7 | 📊 实测 |
| `_mlir_compiled_fused_softmax_kernel` | 20 | 51.1 | 2.6 | 📊 实测 |

### Triton 3-pass — Top Kernels (全流水线)

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `triton_3pass` | 20 | 1231.7 | 61.6 | 📊 实测 |
| `aten::bmm` | 40 | 443.4 | 11.1 | 📊 实测 |
| `ampere_sgemm_128x128_nn` | 20 | 270.6 | 13.5 | 📊 实测 |
| `ampere_sgemm_128x128_tn` | 20 | 172.8 | 8.6 | 📊 实测 |
| `_fused_scale_mask_softmax_fwd` | 20 | 50.7 | 2.5 | 📊 实测 |

### Triton Online — Top Kernels (全流水线)

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `triton_online` | 20 | 1276.2 | 63.8 | 📊 实测 |
| `aten::bmm` | 40 | 444.0 | 11.1 | 📊 实测 |
| `ampere_sgemm_128x128_nn` | 20 | 270.9 | 13.5 | 📊 实测 |
| `ampere_sgemm_128x128_tn` | 20 | 173.1 | 8.7 | 📊 实测 |
| `_online_softmax_fwd` | 20 | 50.1 | 2.5 | 📊 实测 |

### MLIR + Triton 3-pass — Top Kernels (全流水线)

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `## Call CompiledFxGraph fdxfcqknozw6a63api4jk` | 20 | 902.3 | 45.1 | 📊 实测 |
| `aten::bmm` | 40 | 443.4 | 11.1 | 📊 实测 |
| `ampere_sgemm_128x128_nn` | 20 | 270.8 | 13.5 | 📊 实测 |
| `ampere_sgemm_128x128_tn` | 20 | 172.6 | 8.6 | 📊 实测 |
| `_fused_scale_mask_softmax_fwd_0` | 20 | 50.6 | 2.5 | 📊 实测 |

### MLIR + Triton Online — Top Kernels (全流水线)

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `## Call CompiledFxGraph fed4yvme444pnh72kzwxs` | 20 | 898.9 | 44.9 | 📊 实测 |
| `aten::bmm` | 40 | 443.8 | 11.1 | 📊 实测 |
| `ampere_sgemm_128x128_nn` | 20 | 270.8 | 13.5 | 📊 实测 |
| `ampere_sgemm_128x128_tn` | 20 | 173.0 | 8.7 | 📊 实测 |
| `_online_softmax_fwd_0` | 20 | 50.1 | 2.5 | 📊 实测 |

### compile + MLIR 自编译 — Top Kernels (全流水线)

| Kernel | 调用次数 | 总耗时 (μs) | 平均 (μs) | 数据来源 |
|--------|:--------:|:-----------:|:---------:|----------|
| `## Call CompiledFxGraph fg7ltuooxtu2wcl2pqab4` | 20 | 866.3 | 43.3 | 📊 实测 |
| `aten::bmm` | 40 | 443.6 | 11.1 | 📊 实测 |
| `ampere_sgemm_128x128_nn` | 20 | 270.8 | 13.5 | 📊 实测 |
| `ampere_sgemm_128x128_tn` | 20 | 172.8 | 8.6 | 📊 实测 |
| `_mlir_compiled_fused_softmax_kernel_0` | 20 | 50.6 | 2.5 | 📊 实测 |

> **说明**:
> - **六个版本都是独立实现**，各自完成完整 Attention 计算 (QK^T → softmax → ·V)
> - **MLIR 融合 (compile)**: torch.compile 包裹原始 FullAttention，编译器自动融合可融合 op
> - **Triton 3-pass / Online**: 仅 softmax 部分替换为手写 Triton kernel，matmul 仍用 cublas
> - **MLIR + Triton**: torch.compile 包裹 TritonAttention，观察编译器优化能否在 Triton kernel 之上进一步优化 matmul 等部分
> - 与 Stage 1 全流水线加速比 (Triton 1.12×) 可直接对比，测量范围一致

## 三阶段实验联系

| 阶段 | 内容 | 关键发现 | 数据来源 |
|------|------|----------|----------|
| Stage 1 Profiling | baseline/SDPA/compiled 对比 | attention 中 90 次碎片化 kernel 启动 | 📂 Stage1 实测 |
| Stage 2 Triton | 手写 scale+mask+softmax 融合 | 融合为 1 kernel，加速 1.18× (SDPA), 1.13× (Triton) | 📂 Stage1 实测 |
| Stage 3 MLIR | 编译器 IR 层面融合分析 | 自动识别 36 个可消除操作 | 📊 实测 |

## 生成文件

| 文件 | 说明 |
|------|------|
| `mlir/generated_torch_dialect.mlir` | Torch dialect IR (融合前) — 📊 实测导出 |
| `mlir/generated_torch_fused.mlir` | Torch dialect IR (融合后) — 📊 实测融合 |
| `mlir/generated_linalg_dialect.mlir` | Linalg dialect IR — 📊 实测导出 |
| `reports/mlir_fusion_analysis.md` | 本报告 |
