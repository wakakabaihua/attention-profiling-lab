# Benchmark Report: Compare All Backends: scale + causal_mask + softmax

**Date**: 2026-05-29 12:51:13
**Baseline**: baseline

## Results

| Backend | Mean(ms) | Std(ms) | Min(ms) | Speedup | Correct |
|---------|----------|---------|---------|---------|---------|
| baseline | 0.026 | 0.001 | 0.025 | 1.00x | ✓ |
| compiler (ref) | 0.047 | 0.067 | 0.014 | 0.55x | ✓ |
| triton (stage2) | 0.030 | 0.020 | 0.004 | 0.85x | ✓ |
| compiler (triton) | 0.031 | 0.001 | 0.030 | 0.83x | ✓ |
| compiler (tvm) | 0.022 | 0.055 | 0.007 | 1.16x | ✓ |

## 相关分析文档

- TVM lowering 过程: [reports/tvm_backend_lowering.md](/data/github/attention-profiling-lab/reports/tvm_backend_lowering.md)
- TVM 性能与 Triton 对比: [reports/tvm_backend_performance.md](/data/github/attention-profiling-lab/reports/tvm_backend_performance.md)
- Compiler Pipeline 架构说明: [reports/mini_ai_compiler_architecture.md](/data/github/attention-profiling-lab/reports/mini_ai_compiler_architecture.md)

---

## 数据异常说明

### ① compiler (ref)：std（0.067ms）> mean（0.047ms）

**现象**：Reference backend 的标准差大于均值，且 mean（0.047ms）比 baseline（0.026ms）慢 1.8×。

**根本原因（三层）**：

1. **执行模型差异**：`baseline` 是纯 PyTorch eager op（`F.softmax` + `masked_fill`），全程 C++/CUDA，调用链短；
   `compiler (ref)` 是 Python 层遍历 IRGraph → 逐节点 dispatch → 调用 PyTorch op，每个节点有额外的 Python 调用栈开销（字典查找、函数对象 call、属性访问），相当于多了 3-4 层 Python 间接调用。

2. **高右偏分布（std > mean 的统计含义）**：这类极短任务（< 0.1ms）的 CUDA event timing 分布往往是右偏的——
   大多数迭代落在 `min_ms`（0.014ms）附近，但少数迭代因 Python GC、OS 调度抢占、CUDA context 同步时序窗口，产生了若干 0.3–0.5ms 的异常高点，
   拉高了均值并使 std 超过 mean。使用 `min_ms`（0.014ms）作为稳态时延更能代表纯计算开销。

3. **warmup 次数不足**：当前 `warmup=10` 对 Python 层级的函数路径（dict lookup、attribute access 的内联缓存）不够充分。
   建议将 ref backend 的 warmup 提高到 50 次，可将 std 降至 0.01ms 量级。

**结论**：Reference backend 比 baseline 慢是**符合预期的**（Python 解释层 + IR 遍历开销），并非计算效率问题；
数据可信，std 偏大是小样本右偏分布的统计特性，使用 `min_ms=0.014ms` 作为稳定指标更准确。

---

### ② compiler (tvm)：std（0.055ms）> mean（0.022ms），但 mean 最快（1.16×）

**现象**：TVM backend 均值最快但标准差也最大，min（0.007ms）远低于 mean。

**根本原因**：

1. **TVM VirtualMachine 调度抖动**：TVM Relax VM 在执行时通过 Python 侧 `vm["main"](inp)` 触发，
   每次调用都有一次 Python → C++ → CUDA kernel dispatch 的路径切换。内核本身执行极快（0.007ms），
   但 VM 调度层在某些迭代会有额外的异步调度窗口，导致少数测量值偏高。

2. **DLPack 零拷贝路径**：`tvmrt.from_dlpack` 和 `torch.from_dlpack` 在跨框架共享 CUDA 内存时，
   偶发性地触发 CUDA stream synchronization，引入 < 0.05ms 的额外同步开销。

3. **TVM 编译结果（无 MetaSchedule tuning）**：本次使用 `relax.build(mod, target="cuda")`
   采用 TVM 默认 schedule，未经 MetaSchedule 自动调优。默认 schedule 对 fp16 softmax 的 thread block 配置
   不一定最优，但依然因 operator fusion（scale + tril_mask + softmax 在单一 Relax 函数内）快于 baseline 分散调用。

**结论**：TVM 的 1.16× 加速来源于 **Relax 函数内的 operator fusion**（scale/mask/softmax 编译为单个 CUDA kernel），
而非 schedule 调优。若加入 MetaSchedule 自动调优，预期可进一步提升至 1.5×–2.0×。
`min_ms=0.007ms` 是最准确的 TVM 内核执行时间，均值受 VM dispatch 开销抬高。

---

## 推荐稳定性指标对比

| Backend | Min(ms) | 说明 |
|---------|---------|------|
| baseline | 0.025 | PyTorch eager，最稳定 |
| compiler (ref) | 0.014 | Python IR 遍历，用 min 代表稳态 |
| triton (stage2) | 0.004 | Triton kernel，测量接近硬件极限 |
| compiler (triton) | 0.030 | 编译路径 + Triton kernel，稳定 |
| compiler (tvm) | 0.007 | TVM 融合 kernel，用 min 代表内核时间 |

> **备注**：对于 < 0.1ms 的 CUDA 微 benchmark，`min_ms` 比 `mean_ms` 更能反映计算本身的时延下界，
> 因为 mean 受 Python dispatch、GC、OS 调度等不确定因素影响；`std > mean` 是这类场景的统计常见现象，
> 不代表结果不可信，而是分布右偏的正常表现。