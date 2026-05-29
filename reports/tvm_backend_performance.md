# TVM Backend 分析（二）：性能、调度路径与 Triton 对比

**日期**: 2026-05-29  
**对应代码**: `compiler/backends/tvm_backend.py`, `tvm_integration/relax_importer.py`, `benchmarks/compare_all_backends.py`  
**对比对象**: baseline / compiler(ref) / triton(stage2) / compiler(triton) / compiler(tvm)

---

## 1. Benchmark 结果回顾

来自跨 backend benchmark 报告：

| Backend | Mean(ms) | Std(ms) | Min(ms) | Speedup vs baseline |
|---------|----------|---------|---------|---------------------|
| baseline | 0.026 | 0.001 | 0.025 | 1.00x |
| compiler (ref) | 0.047 | 0.067 | 0.014 | 0.55x |
| triton (stage2) | 0.030 | 0.020 | 0.004 | 0.85x |
| compiler (triton) | 0.031 | 0.001 | 0.030 | 0.83x |
| compiler (tvm) | 0.022 | 0.055 | 0.007 | 1.16x |

如果只看 `mean_ms`，TVM backend 是当前最快路径；如果看 `min_ms`，TVM kernel 的稳态执行时间为 0.007ms，也明显优于 baseline 的 0.025ms。

但这组数据还需要进一步拆解，否则“1.16x 为什么成立”讲不清楚。

---

## 2. 当前 TVM backend 实际用了什么优化

### 2.1 已经使用的优化

当前实现真正用到的优化只有两类：

1. **子图融合后的统一 lowering**  
   内部 IR 中的 `FUSED_SCALE_MASK_SOFTMAX` 节点被整体 lower 成一个 Relax 函数，而不是三个独立子 op 分散在 Python / PyTorch 中执行。

2. **TVM 默认编译链的 operator fusion / codegen**  
   `relax.build(mod, target="cuda")` 会把同一个 Relax 函数中的 elementwise + reduction op 交给 TVM 默认 lowering 与 codegen 流程处理，最终生成 CUDA 可执行体。

### 2.2 当前没有使用的优化

这几点必须明确说清楚，因为它们决定了这份结果的“含金量边界”：

1. **没有 MetaSchedule autotuning**  
   代码中没有 `tune_tir()`、没有 MetaSchedule database、没有 tuned schedule replay。

2. **没有手写 TensorIR schedule**  
   当前仓库里不存在自定义 `tir.Schedule` 脚本对 softmax 做 split / bind / cache_read / cooperative fetching。

3. **没有 warp-level online softmax**  
   Triton Stage 2 的 online softmax 是显式面向 kernel 设计的；TVM 当前路径只是把 scale + mask + softmax 编进一个 Relax 函数，并不等价于 FlashAttention/online-softmax 的算法级重写。

4. **没有显式 shared memory / block tiling 策略说明**  
   因为当前走的是 TVM 默认 build 路径，调度细节是由 TVM 内部默认规则决定的，不是本项目显式控制的 schedule。

**结论**：当前 TVM backend 的 1.16x 加速，本质上是“融合成功 + 默认编译链可执行”，不是“TVM 调度已经优化到位”。

---

## 3. 1.16x 加速到底来自哪里

### 3.1 与 baseline 的关键差异

baseline 的实现路径是：

```python
scores.masked_fill(mask, -inf) * scale
→ F.softmax(...)
```

在 PyTorch eager 模式下，这条路径的问题是：

1. `masked_fill`
2. `mul`
3. `softmax`

虽然每个 op 本身已经是高效 CUDA kernel，但它们仍然是**多个独立的框架级调用点**。对这么小的 workload（B=1, H=12, T=128），Python 调度、ATen dispatch、kernel launch overhead 占比会被放大。

### 3.2 TVM 的收益来源

TVM 路径把 `scale + mask + softmax` 统一成一个 Relax 函数提交给 compiler：

```python
scaled = multiply(scores, scale_const)
masked = where(tril_mask, scaled, neg_inf)
output = nn.softmax(masked, axis=3)
```

这里的收益主要来自三点：

1. **调用边界减少**  
   从多个框架 op 调用，变成一次 VM 调用进入已编译模块。

2. **中间表示统一**  
   在 Relax 函数内部，TVM 可以把多个张量 op 作为一个整体分析，而不是像 eager 模式那样逐个 op 交给运行时。

3. **编译期常量固化**  
   `scale_factor=0.125`、`is_causal=True`、`axis=3` 已经在 build 前固定，可减少一部分运行时判断。

因此，当前的 1.16x 更准确地说是：

> 从“框架运行时逐 op 调度”切换到“compiler 管理的统一 lowered function”后，减少了调度与边界开销。

这是一种**编译路径收益**，不是**精细 schedule 收益**。

---

## 4. 为什么 TVM 只比 compiler(triton) 快一点

这是面试里一定会被追问的问题，因为当前数据里：

- `compiler (tvm)` = 1.16x
- `compiler (triton)` = 0.83x

差距不算巨大，而且和预期中的“TVM 或 Triton 显著拉开 baseline”不完全一致。

### 4.1 原因一：测试 workload 太小

当前 benchmark 是：

- B = 1
- H = 12
- T = 128
- D = 64

这属于很小的 attention 子问题。此时总计算量很低，容易出现：

- kernel 启动开销占主导
- Python dispatch 抖动占主导
- 框架间 bridge 开销被放大

在这种 regime 下，backend 之间真正的 kernel 差异会被大量“非计算因素”淹没。

### 4.2 原因二：compiler(triton) 走的是“通用后端接入”，不是 Stage 2 极致手写路径

`compiler (triton)` 不是直接 benchmark Stage 2 的裸 kernel，而是：

```text
IRGraph → Fusion Pass → TritonKernelSpec → TritonBackend.execute()
```

相比直接调用 `triton_fused_scale_mask_softmax`，它多了：

- IRGraph 遍历
- spec 查找
- backend dispatch
- value_map 管理

这会把很小 workload 下的调度成本放大。也就是说，`compiler (triton)` 测到的是“编译管线集成后的 Triton backend”，不是“理论上的 Triton kernel 上限”。

### 4.3 原因三：TVM 的默认 build 可能恰好适合这个小规模 shape

TVM 当前对固定 shape `(1, 12, 128, 128)` 做静态 build，某些默认 lowering 路径对这种小尺寸归约较友好，因此在这个特殊 shape 上跑出了略优于 compiler(triton) 的均值。

但这不能直接推导为“TVM 普遍强于 Triton”。

**更严谨的表述**：

> 在当前固定 shape、默认 schedule、无 autotuning 的条件下，TVM backend 在 mean latency 上略优于 compiler(triton)；但由于 workload 很小且存在明显 timing jitter，这个差距更多说明“当前接入路径的系统开销差异”，而不是普遍意义上的 kernel 绝对性能排序。

---

## 5. 从 Nsight / profiling 视角，这份 TVM 分析还缺什么

你前面提到“缺少 TVM 专项分析报告”，核心就在这里：现在只能解释到 compiler/runtime 层，还没真正深入到 kernel 级别。

如果要把 TVM 这部分从“能跑”升级成“研究过优化路径”，至少还差三层证据：

### 5.1 TVM 生成 kernel 的 launch 配置

需要补充：

- block size / thread 数
- grid 配置
- register usage
- shared memory usage
- occupancy

这部分可通过：

- 导出生成的 CUDA code / PTX
- 使用 Nsight Compute (`ncu`) 分析 TVM kernel

### 5.2 kernel 时间分解

需要知道：

- `tril + where + softmax` 是否真的被 fuse 到单个 kernel
- 如果没有完全融合，拆成了几个 kernel
- 每个 kernel 的时间占比是多少

### 5.3 与 Triton 的微观差异

需要回答：

- Triton 是否更充分利用了 SRAM/shared memory
- TVM 默认 schedule 是否做了足够的 reduce 优化
- Triton online softmax 的算法优势，为什么在这里没完全体现

**目前项目状态**：这些 profiling 级证据还没有，因此报告应明确写成“当前结论建立在 end-to-end latency 与 lowering 分析上，尚未下钻到 kernel 微架构层”。

---

## 6. 面试里如何准确表述 TVM 结果

建议用下面这套口径，既不夸大，也不会显得做得太浅。

### 版本 A：简洁表述

> 我把 attention 子图中的 `scale + causal_mask + softmax` 融合成一个内部 IR 节点，再 lower 到 TVM Relax IR，最后用 `relax.build(target="cuda")` 生成 GPU 可执行体。当前在固定 shape `(1,12,128,128)` 上，相比 baseline 拿到了 1.16x 的平均加速。但这份收益主要来自编译路径统一和 operator fusion，还没有加入 MetaSchedule autotuning 或手写 TensorIR schedule，所以 TVM 这部分目前更偏“lowering 路径打通”，还不是完整的 schedule 优化研究。

### 版本 B：展开表述

> 当前 TVM backend 的价值主要有两点。第一，我验证了内部 IR 可以稳定 lower 到 Relax，并通过 DLPack 与 PyTorch 做零拷贝数据互通，说明 compiler pipeline 的 backend 抽象是成立的。第二，我拿到了一组比 baseline 更快的结果，说明即使不做 MetaSchedule，单靠融合后的统一编译路径也能减少小算子的调度边界开销。它的局限也很明确：还没分析 TVM 生成 kernel 的 launch 配置，也没做 tuned schedule，所以我不会把这 1.16x 解读成“TVM 已经被优化到位”，而是解读成“TVM backend 已经具备研究优化路径的基础”。

---

## 7. 下一步补强建议

按收益和展示价值排序：

1. **补 kernel 级 profiling**  
   用 Nsight Compute 采样 TVM 生成 kernel 的 occupancy、memory throughput、register pressure。

2. **补导出的 CUDA / TIR**  
   把 Relax build 后的下游 TIR 或 CUDA code 抽出来，明确当前到底是怎样的 schedule。

3. **做一次 MetaSchedule tuning 实验**  
   形成 “default schedule vs tuned schedule” 对照表，这是 TVM 专项报告里最能打的部分。

4. **扩大 shape sweep**  
   例如测试 `T=128/256/512`，区分小 workload 和大 workload 下的 TVM/Triton 差异。

---

## 8. 当前 TVM 部分的完成度判断

| 项目 | 当前状态 | 评价 |
|------|----------|------|
| IR → Relax lowering | 已完成 | 可以讲清楚 |
| TVM backend 接入执行 | 已完成 | 可以演示 |
| 正确性验证 | 已完成 | 数据可信 |
| 默认 build 性能结果 | 已完成 | 可作为阶段性结论 |
| schedule 策略解释 | 未完成 | 目前只能说明“默认 schedule” |
| MetaSchedule / tuning | 未完成 | 核心补强项 |
| Nsight kernel profiling | 未完成 | 缺关键性能证据 |
| TVM vs Triton 深度对比 | 部分完成 | 目前只有 end-to-end 解释 |

**结论**：

当前 TVM backend 已经从“只是加了个 backend”提升到“lowering 路径和性能边界可以清楚说明”，但距离“完整的 TVM 优化研究”还差 schedule 与 profiling 两层证据。

---

相关文档：
- [tvm_backend_lowering.md](/data/github/attention-profiling-lab/reports/tvm_backend_lowering.md)
- [compiler_pipeline_benchmark_20260529_125113.md](/data/github/attention-profiling-lab/reports/compiler_pipeline_benchmark_20260529_125113.md)
- [mini_ai_compiler_architecture.md](/data/github/attention-profiling-lab/reports/mini_ai_compiler_architecture.md)
