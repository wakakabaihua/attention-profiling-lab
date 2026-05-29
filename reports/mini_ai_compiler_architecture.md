# Mini AI Compiler Pipeline 架构说明

**日期**: 2026-05-29  
**覆盖范围**: `compiler/`, `tvm_integration/`, `benchmarks/compare_all_backends.py`

---

## 1. 设计目标

这个项目的目标不是做一个泛化 AI compiler，而是围绕 attention 子图构建一条最小但完整的编译链路，使下面几件事能同时成立：

1. 能从前端图中识别 attention 相关 pattern。
2. 能用统一内部 IR 表示这些 pattern。
3. 能通过 pass 做 pattern fusion。
4. 能 lower 到多个 backend（reference / Triton / MLIR / TVM）。
5. 能用 benchmark 和 correctness test 验证这条链路是成立的。

也就是说，这套架构的重点是“链路完整、边界清晰、可验证”，而不是“支持所有模型”。

---

## 2. 总体架构

```text
PyTorch Module / FX Graph
        │
        ▼
Frontend Import
compiler/frontend/fx_importer.py
        │
        ▼
Internal IR
compiler/ir/{ops.py, graph.py, printer.py}
        │
        ▼
Pass Pipeline
compiler/passes/
  - canonicalize.py
  - pattern_match.py
  - fusion.py
  - validation.py
        │
        ▼
Lowering Layer
compiler/lowering/
  - to_triton.py
  - to_mlir.py
  - pipeline.py
        │
        ├──────────────▶ Triton Backend
        │                compiler/backends/triton_backend.py
        │
        ├──────────────▶ Reference Backend
        │                compiler/backends/reference_backend.py
        │
        ├──────────────▶ MLIR Backend
        │                compiler/backends/mlir_backend.py
        │
        └──────────────▶ TVM Backend
                         compiler/backends/tvm_backend.py
                                │
                                ▼
                         tvm_integration/relax_importer.py
```

这个结构把“图表示、图变换、代码生成、执行”四层拆开，避免 benchmark 脚本里同时混合 pattern match、kernel 调用和 profiling 逻辑。

---

## 3. 模块边界说明

### 3.1 Frontend：把 PyTorch 图转成内部 IR

对应文件：
- `compiler/frontend/fx_importer.py`
- `compiler/frontend/graph_utils.py`

职责：

1. 读取 `torch.fx` / `torch.export` 产生的图。
2. 将 FX node 映射成内部 `OpType`。
3. 抽取后续 pass 所需属性，如：
   - `scale_factor`
   - `is_causal`
   - `dim`
4. 保留节点名与 shape 信息，方便后续调试和对照。

这里的关键设计是：Frontend 只负责“翻译”，不负责优化。

例如在 `fx_importer.py` 中，attention 相关 op 被分类为：

- `aten.mul` → `OpType.SCALE`
- `aten.masked_fill` / `aten.where` → `OpType.MASK`
- `aten.softmax` → `OpType.SOFTMAX`
- `aten.matmul` / `aten.bmm` → `OpType.MATMUL`

这使后续 pass 可以完全基于内部 IR 工作，而不依赖 PyTorch FX 的细节表示。

---

### 3.2 Internal IR：统一表示 attention 子图

对应文件：
- `compiler/ir/ops.py`
- `compiler/ir/graph.py`
- `compiler/ir/printer.py`

内部 IR 的核心结构有三个：

1. `OpType`  
   描述算子种类，目前只覆盖 attention 实验需要的最小集合。

2. `IRNode`  
   表示单个计算节点，包含：
   - `name`
   - `op_type`
   - `inputs`
   - `attrs`
   - `output_shape`
   - `meta`

3. `IRGraph`  
   表示 DAG，提供：
   - `add_node()`
   - `get_node()`
   - `get_users()`
   - `topological_sort()`

设计原则是“足够小，但足够稳定”：

- 用字符串名字表示 def-use 关系，结构简单，便于做 pass。
- 支持动态 shape（`-1`）但当前主要跑静态 shape。
- 用 `attrs` 携带编译常量，避免每个 op 都定义复杂 class hierarchy。

这套 IR 不是工业级大 IR，但非常适合 attention 子图的 pattern 驱动优化实验。

---

### 3.3 Pass 层：在内部 IR 上做语义保留的图变换

对应文件：
- `compiler/passes/canonicalize.py`
- `compiler/passes/pattern_match.py`
- `compiler/passes/fusion.py`
- `compiler/passes/validation.py`

Pass 层按职责拆成四部分：

#### Canonicalize

目标：把属性和图结构整理到规范形式。

做的事情包括：
- 把 `scale_factor` 统一转成 `float`
- 把 `softmax_dim` 统一转成 `int`
- 删除死节点
- 重新按拓扑顺序组织节点

这样可以避免 pattern match 因属性类型不一致而失败。

#### Pattern Match

目标：识别 attention 相关线性链。

当前重点支持：
- `SCALE -> MASK -> SOFTMAX`
- `MATMUL -> SCALE -> MASK -> SOFTMAX`

这个层只“识别”，不修改图。

#### Fusion

目标：把已识别的 pattern 替换成融合节点。

例如：

```text
scores -> scale_0 -> mask_0 -> softmax_0 -> output
```

会被替换成：

```text
scores -> fused_sms_0 -> output
```

并把以下属性继承到融合节点：
- `scale_factor`
- `is_causal`
- `softmax_dim`
- `mask_value`

#### Validation

目标：在每个关键阶段检查图是否合法。

验证内容包括：
- 输入边数量是否符合 `OpSpec`
- 引用的节点是否存在
- 图是否无环
- pass 后结构是否自洽

Pass 层的关键边界是：

> 只变换 IR，不关心 backend 如何执行。

---

### 3.4 Lowering 层：把融合后的 IR 变成 backend 能消费的表示

对应文件：
- `compiler/lowering/pipeline.py`
- `compiler/lowering/to_triton.py`
- `compiler/lowering/to_mlir.py`
- `tvm_integration/relax_importer.py`

Lowering 层的职责不是执行，而是把内部 IR 翻译成更接近执行系统的中间表示。

#### Triton Lowering

`to_triton.py` 把 `FUSED_SCALE_MASK_SOFTMAX` 节点转换为 `TritonKernelSpec`：

- `scale_factor`
- `is_causal`
- `softmax_dim`
- `mask_value`

这是一个很轻量的 lowering，基本上是“属性抽取 + 参数封装”。

#### MLIR Lowering

`to_mlir.py` 提供两条路径：

1. `lower_to_mlir_text()`  
   生成文本形式，主要用于 dump 和调试。

2. 更偏工业化的 module 路径  
   把内部图映射成真实 MLIR IR 结构。

#### TVM Lowering

`tvm_integration/relax_importer.py` 把融合节点 lower 成 Relax 函数：

- `multiply`
- `tril`
- `where`
- `nn.softmax`

这一步是内部 IR 与 TVM 编译器之间最明确的接口边界。

---

### 3.5 Backend 层：执行 lowered artifact

对应文件：
- `compiler/backends/reference_backend.py`
- `compiler/backends/triton_backend.py`
- `compiler/backends/mlir_backend.py`
- `compiler/backends/tvm_backend.py`

Backend 层的统一输入是 `CompilationArtifact`，统一输出是 `torch.Tensor`。

#### Reference Backend

作用：
- 用 PyTorch eager 逐节点执行 IRGraph。
- `FUSED_SCALE_MASK_SOFTMAX` 节点在这里被展开成：
  - scale
  - causal mask
  - softmax
- 它是 correctness 基准，而不是性能基准。

#### Triton Backend

作用：
- 从 `artifact.triton_specs` 中取出 `TritonKernelSpec`
- 调用 Stage 2 的 Triton fused kernel
- 非融合节点回退到 ReferenceBackend

它复用了已有 Triton kernel，因此能把“编译器识别出来的融合节点”接到“手写高性能 kernel”上。

#### TVM Backend

作用：
- 对融合节点调用 `lower_to_relax()`
- 用 `relax.build(target="cuda")` 编译
- 通过 `relax.VirtualMachine` 执行
- 用 DLPack 与 PyTorch 共享 CUDA 张量

TVM backend 的价值是验证：这套 IR 不只能够到 Triton，也能通到另一个 compiler stack。

---

### 3.6 Runtime / Benchmark：验证系统行为

对应文件：
- `compiler/runtime/benchmark_runner.py`
- `benchmarks/compare_all_backends.py`

这里分两层：

1. `BenchmarkRunner`  
   聚焦 compiler 内部 backend 对比。

2. `compare_all_backends.py`  
   聚焦 Stage 1–4 跨阶段对比：
   - baseline
   - compiler(ref)
   - triton(stage2)
   - compiler(triton)
   - compiler(tvm)

这样设计的好处是：

- compiler 系统内部 benchmark 不依赖外部 profiling 脚本
- 端到端对比也不会把 compiler 内部逻辑写死在 runtime 模块中

---

## 4. 为什么这套架构是合理的

### 4.1 关注点分离

这套架构的最大优点是把不同层的问题拆开：

- Frontend 解决“看懂图”
- IR 解决“怎么表示图”
- Pass 解决“怎么变换图”
- Lowering 解决“怎么翻译到目标系统”
- Backend 解决“怎么执行”
- Runtime 解决“怎么评估”

这比把 pattern match、kernel 调用、benchmark 全塞进一个脚本更容易扩展，也更适合面试时讲清楚模块边界。

### 4.2 可替换性强

因为中间通过 `IRGraph` 和 `CompilationArtifact` 解耦：

- 以后可以接新的 frontend
- 可以增加新的 pass
- 可以新增 backend（例如 CUTLASS / Inductor 实验）
- benchmark 层几乎不用改

### 4.3 适合 attention 主题

项目当前的研究主线是 attention 优化，而不是通用编译器。因此用一个“小而清晰”的 IR 比引入大而全的系统更合适。

这也解释了为什么当前 IR 只覆盖：

- MATMUL
- SCALE
- MASK
- SOFTMAX
- FUSED_SCALE_MASK_SOFTMAX

因为这些已经足够支撑本项目的核心问题：

> attention 中哪些结构值得融合，以及融合后如何 lower 到不同 backend。

---

## 5. 测试证据

当前仓库里已经有一组比较完整的测试文件：

- `compiler/tests/test_ir.py`  
  覆盖 IRShape / IRNode / IRGraph 基础行为。

- `compiler/tests/test_pattern_match.py`  
  覆盖 pattern 识别、负样本、双 pattern 等。

- `compiler/tests/test_fusion.py`  
  覆盖融合节点生成、属性继承、原节点删除、图重连、数值正确性。

- `compiler/tests/test_pipeline.py`  
  覆盖从 IRGraph / nn.Module 出发的端到端编译流程。

- `compiler/tests/test_tvm_backend.py`  
  覆盖 Relax lowering、TVM backend 执行与 fallback。

因此如果面试官问“你怎么证明这不是一次性脚本”，可以明确回答：

> 这套 pipeline 每一层都有独立测试入口，IR、pattern match、fusion、pipeline、TVM backend 都有单元测试，而不是只靠 benchmark 结果证明它能工作。

---

## 6. 当前架构的边界与不足

这份架构说明也需要把边界讲清楚，否则容易被理解成“已经接近通用 compiler”。

### 当前已解决

1. attention 子图的最小 IR 抽象
2. pattern-driven fusion
3. 多 backend lowering 与执行
4. correctness + benchmark + report 这三层验证

### 当前未解决

1. 通用图优化框架
2. 动态 shape 编译缓存策略
3. 完整内存规划
4. 真实 cost model 驱动的 pass 决策
5. MetaSchedule / autotuning 集成
6. kernel 级 profile 自动汇总

所以更准确的定位是：

> 一个围绕 attention 子图构建的 mini compiler pipeline 原型，用来验证图优化、lowering 边界与 backend 可替换性。

---

## 7. 对外展示时的总结说法

建议用下面这段作为项目架构总结：

> 我把 attention 优化问题拆成了五层：Frontend 把 FX 图映射到统一 IR；Pass 层在 IR 上做 canonicalize、pattern match 和 fusion；Lowering 层把融合节点分别翻译到 Triton、MLIR、TVM；Backend 层执行这些 lowered artifact；Runtime 层负责 correctness 和 benchmark。这样做的好处是，每一层职责都清楚，既能解释为什么某个优化成立，也能把不同 backend 放在同一个 compiler 抽象下做对比，而不只是零散地写几份 profiling 脚本。

---

相关文档：
- [mini_ai_compiler_pipeline_design.md](/data/github/attention-profiling-lab/reports/mini_ai_compiler_pipeline_design.md)
- [tvm_backend_lowering.md](/data/github/attention-profiling-lab/reports/tvm_backend_lowering.md)
- [tvm_backend_performance.md](/data/github/attention-profiling-lab/reports/tvm_backend_performance.md)
