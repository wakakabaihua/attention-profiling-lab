# Bottleneck Analysis Report

> **Date**: ____
> **Model**: MiniTransformerBlock (1-layer)
> **Config**: hidden_size=___, num_heads=___, seq_len=___, batch_size=___
> **GPU**: ____
> **PyTorch version**: ____

---

## 1. Profiling 发现

### 1.1 Baseline (Manual Attention)

| 指标 | 数值 |
|------|------|
| 总 CUDA kernel 数量 | |
| 总 CUDA 时间 | |
| 小 kernel (<50μs) 数量 | |
| 小 kernel 时间占比 | |
| Memcpy 事件数 | |
| 平均 kernel 间隔 | |

**Top-5 耗时 kernel:**

| Kernel | 调用次数 | 总时间 | 平均时间 |
|--------|---------|--------|---------|
| | | | |

### 1.2 SDPA / FlashAttention

| 指标 | 数值 |
|------|------|
| 总 CUDA kernel 数量 | |
| 总 CUDA 时间 | |

### 1.3 torch.compile

| 指标 | 数值 |
|------|------|
| 总 CUDA kernel 数量 | |
| 总 CUDA 时间 | |

---

## 2. 瓶颈分类

### 2.1 Kernel Launch Overhead

- [ ] timeline 中 kernel 间存在明显空洞
- [ ] CPU dispatch 耗时高于 GPU 计算

### 2.2 Kernel 粒度过小

- [ ] 存在大量 <50μs 的 elementwise kernel
- [ ] softmax / layernorm / add 被拆分为多个 kernel

### 2.3 Host ↔ Device Memory Copy

- [ ] 出现 HtoD / DtoH memcpy 事件
- [ ] 存在不必要的数据搬运

### 2.4 Attention 碎片化

- [ ] scale / mask / softmax / dropout 分别为独立 kernel
- [ ] 可以合并为单个 fused kernel

### 2.5 Stream 同步

- [ ] 出现 cudaStreamSynchronize
- [ ] 存在不必要的同步等待

---

## 3. 编译优化假设

### 假设 1: Attention Sub-op Fusion

**当前状态:**
- attention 内部有 ___ 个独立 kernel
- 包括: scale, mask, softmax, dropout, matmul

**优化方案:**
- 通过 MLIR pass / Triton kernel 融合为 1–2 个 kernel

**预期收益:**
- kernel 数量减少: ___
- launch overhead 减少: ___
- 端到端 latency 降低: ___%

### 假设 2: Elementwise Fusion (LayerNorm + Add)

**当前状态:**
- LayerNorm 和残差 add 为独立 kernel

**优化方案:**
- 融合 add + LayerNorm 为单个 kernel

**预期收益:**
- kernel 数量减少: ___
- memory read/write 减少: ___

---

## 4. 验证计划

- [ ] 使用 SDPA 验证 attention fusion 收益
- [ ] 使用 torch.compile 验证 Inductor fusion 效果
- [ ] 设计 MLIR attention fusion pass
- [ ] 编写 Triton fused kernel 对比

---

## 5. 结论

_(填写 profiling 后的总结性发现和下一步行动)_
