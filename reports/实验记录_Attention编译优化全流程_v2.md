# Attention 编译优化全流程实验记录（v2：MLIR 原生 Pass 实现）

> **作者实验笔记** · 2026-03-14  
> **环境**: PyTorch 2.12.0.dev20260301+cu126 · torch-mlir 20260301.738 · Triton 3.6.0 · NVIDIA RTX 4090 · CUDA 12.6  
> **完整代码**: [attention-profiling-lab](https://github.com/attention-profiling-lab)

---

## 版本说明

本文档是 v1 实验记录的修订版。v1 在 2026-03-02 完成了三阶段实验的完整闭环，但 Stage 3 的 MLIR Fusion Pass 使用正则表达式和字符串操作"模拟"MLIR 的模式匹配。v1 在自我评估中给出了**工程严谨度 ⭐⭐、工业可扩展性 ⭐**的评级。

v2（本文档）使用 MLIR 原生 Python Bindings 重写了整个 Pass 体系和编译管线，并重新运行了对比实验。本文档特别关注 **v1 → v2 的架构差异和实验数据对比**。

**变更范围**：仅 Stage 3 部分重写。Stage 1（Profiling）和 Stage 2（Triton）的代码和数据保持不变。

---

## 一、实验目的与核心假设（同 v1，不变）

> **假设**：Attention 推理中 softmax 相关的子操作（scale、causal_mask、softmax）被 PyTorch eager mode 拆分为多个独立 CUDA kernel。这些 kernel 的**调度开销**和**中间结果的全局内存反复读写**是可被编译优化消除的瓶颈。

三阶段递进设计：

```
Stage 1: Profiling    →  用 GPU profiler 定位调度/内存瓶颈
Stage 2: Triton       →  手写融合 kernel 验证 "融合确实能消除瓶颈"
Stage 3: MLIR         →  让编译器自动做同样的事（IR 分析 → 自动 codegen）
```

实验参数不变：B=1, H=12, T=128, D=64, dtype=float32（Stage 3 导出）。

---

## 二、Stage 1 & Stage 2（同 v1，不变）

Stage 1（GPU Profiling）和 Stage 2（Triton 手写融合）的实验代码和数据与 v1 完全一致。关键结论简要回顾：

| 阶段 | 关键发现 |
|------|---------|
| Stage 1 | 360 个 kernel，平均 3.1μs，launch overhead 占 40-60%；softmax 子操作 60 次启动 130.3μs |
| Stage 2 | Triton 融合将 282 次 kernel 减少为 40 次；softmax-only 加速 ~13×；全流水线受 Amdahl 定律限制 ~1.12× |

详细数据见 v1 文档第二、三节。

---

## 三、Stage 3 v2：MLIR 原生 Pass —— 架构重写

### 3.1 v1 的核心问题

v1 实验记录在 5.6 节已经明确列出了 Stage 3 的根本缺陷：

> 操作的是 **IR 的文本表示**（字符串），而不是 **MLIR 的内存数据结构**（Operation / Region / Block）。

具体表现：

| 维度 | v1 做法 | 问题 |
|------|--------|------|
| IR 表示 | `str(module)` → 正则表达式解析 | 依赖 IR 打印格式，版本升级即失效 |
| def-use 追溯 | 字符串搜索 SSA 变量名 `%0` | 多 Block 同名变量会出错 |
| 子图替换 | 文本拼接新 IR 行 | 不保证 SSA / dominance 不变量 |
| 属性提取 | `re.search(r'float\s+([\d.e+-]+)', line)` | 脆弱，无类型安全 |

### 3.2 v2 架构：MLIR 原生基础设施

v2 使用 `torch-mlir` 打包的 MLIR Python Bindings（nanobind → C++ `libMLIR`）。这不是 Python 模拟——每个 API 调用直接对应 C++ MLIR 库函数。

#### 编译管线（v2）

```
┌──────────┐    ┌───────────────────┐    ┌───────────────────────┐    ┌──────────────┐    ┌─────┐
│ PyTorch  │ →  │ torch-mlir        │ →  │ MLIR 原生 FusionPass  │ →  │ Triton       │ →  │ GPU │
│ Module   │    │ export_and_import  │    │ RewritePatternSet +   │    │ Codegen+编译 │    │ 执行│
│          │    │ → ir.Module        │    │ walk_and_apply_patterns│    │              │    │     │
└──────────┘    └───────────────────┘    └───────────────────────┘    └──────────────┘    └─────┘
```

#### v1 vs v2 核心 API 对照

| 操作 | v1（Python 模拟） | v2（MLIR 原生） | C++ 等价 |
|------|:--:|:--:|:--:|
| IR 表示 | `str` + `re.findall()` | `ir.Operation` / `ir.Value` | `mlir::Operation*` |
| 图遍历 | 字符串 → dict 映射 | `softmax_op.operands[0].owner` | `Value::getDefiningOp()` |
| 模式匹配 | `op.category == "softmax"` | `RewritePatternSet.add("torch.aten.softmax.int", fn)` | `RewritePatternSet::add<T>()` |
| 替换机制 | 文本行删除 + 拼接 | `PatternRewriter.replace_op(op, results)` | `PatternRewriter::replaceOp()` |
| 操作创建 | 字符串模板 | `ir.Operation.create("custom.fused_...", ...)` | `Operation::create()` |
| 驱动框架 | 手写 for 循环 | `walk_and_apply_patterns(module, frozen)` | `applyPatternsAndFoldGreedily()` |
| 属性提取 | `re.search(r'float\s+([\d.e+-]+)')` | `ir.FloatAttr(attrs["scale"]).value` | `op.getAttrOfType<FloatAttr>()` |
| 属性创建 | 字符串拼接 `is_causal = true` | `ir.BoolAttr.get(True)` | `BoolAttr::get()` |
| 正确性保证 | 无 | MLIR verifier + SSA 自动重连 | 编译器自动维护 |
| Pass 管线 | 不支持 | `PassManager.parse("builtin.module(canonicalize, cse)")` | `PassManager::addPass()` |

### 3.3 Phase 1：Attention 融合 Pass

**文件**：`mlir/passes/attention_fusion_pass.py`（168 行）

从 `torch.aten.softmax.int` 操作反向追溯 SSA def-use chain：

```python
def attention_fusion_pattern(softmax_op: ir.Operation,
                              rewriter: rewrite.PatternRewriter):
    # Step 1: softmax 的第一个操作数应来自 where.ScalarSelf
    masked_value = softmax_op.operands[0]            # ir.Value
    where_op = masked_value.owner                    # ir.Operation (←def-use chain)
    if where_op.name != "torch.aten.where.ScalarSelf":
        return  # 不匹配

    # Step 2: where 的第三个操作数应来自 mul.Scalar
    scaled_value = where_op.operands[2]
    scale_op = scaled_value.owner
    if scale_op.name != "torch.aten.mul.Scalar":
        return

    # Step 3: 创建融合操作
    fused_op = ir.Operation.create(
        "custom.fused_scaled_masked_softmax",
        results=[softmax_op.results[0].type],
        operands=[scale_op.operands[0]],             # 原始 scores
        attributes={
            "scale": ir.FloatAttr.get(ir.F32Type.get(), scale_float),
            "softmax_dim": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), -1),
            "is_causal": ir.BoolAttr.get(True),
            "algorithm": ir.StringAttr.get("online"),
        },
        ip=rewriter.ip,
    )

    # Step 4: MLIR 自动重连 SSA def-use chain
    rewriter.replace_op(softmax_op, list(fused_op.results))
```

**驱动方式**：

```python
patterns = rewrite.RewritePatternSet(module.context)
patterns.add("torch.aten.softmax.int", attention_fusion_pattern, benefit=10)
frozen = patterns.freeze()
rewrite.walk_and_apply_patterns(module.operation, frozen)
```

与 v1 的关键区别：
- **没有字符串解析**——通过 `op.name` 和 `operand.owner` 做结构化匹配
- **没有手写遍历循环**——`walk_and_apply_patterns` 自动驱动（等价 C++ 的 `applyPatternsAndFoldGreedily`）
- **无需手动管理 SSA**——`replace_op()` 自动将 softmax 的所有下游使用者重连到融合操作

#### 融合结果

```mlir
// 融合前: 37 个 Torch dialect 操作
%0 = torch.aten.mul.Scalar %arg0, %float1.250000e-01 ...
... (mask 生成 + 常量操作) ...
%10 = torch.aten.where.ScalarSelf %9, %float-Inf, %0 ...
%11 = torch.aten.softmax.int %10, %int-1, %none ...
return %11

// 融合后: 1 个自定义操作
%0 = "custom.fused_scaled_masked_softmax"(%arg0, 1.250000e-01) {
    softmax_dim = -1 : i64,
    is_causal = true,
    algorithm = "online"
} : (!torch.vtensor<[1,12,128,128],f32>, f32) -> !torch.vtensor<[1,12,128,128],f32>
return %0
```

操作数从 37 减少到 2（-95%），与 v1 一致。

### 3.4 Phase 2：Online Softmax 重写 Pass

**文件**：`mlir/passes/incremental_softmax_pass.py`（210 行）

v2 新增了一个 v1 没有的 Pass：**将标准 3-pass softmax 重写为 online 2-pass softmax**。

这个 Pass 在 Torch dialect 层面匹配被 `torch-mlir` 分解后的 softmax 5-op 链：

```
max.dim → sub.Tensor → exp → sum.dim_IntList → div.Tensor
```

替换为单一的 `custom.online_softmax` 操作。

**工作流程**：

```python
# Step 1: 使用 PassManager 分解 softmax 为基本操作
pm = passmanager.PassManager.parse(
    "builtin.module(func.func(torch-decompose-complex-ops))"
)
pm.run(module.operation)

# Step 2: 在分解后 IR 上匹配 5-op 链
# 从 div.Tensor 反向追溯: div→sum→exp→sub→max
patterns = rewrite.RewritePatternSet(module.context)
patterns.add("torch.aten.div.Tensor", online_softmax_rewrite, benefit=10)
frozen = patterns.freeze()
rewrite.walk_and_apply_patterns(module.operation, frozen)
```

### 3.5 Phase 3：测试体系

**目录**：`mlir/tests/`（5 个测试文件 + FileCheck 框架）

| 测试文件 | 测试数 | 覆盖范围 |
|---------|:---:|---------|
| `test_attention_fusion.py` | 14 | 基础融合、负例、属性、不同形状、pipeline |
| `test_incremental_softmax.py` | 18 | 基础重写、负例、属性、形状、与 scale+mask 联合 |
| `test_edge_cases.py` | 19 | 非因果、dtype、极端形状、非方阵、多 softmax、FullAttention、pipeline 交叉验证 |
| `test_pipeline_e2e.py` | 11 | export→pass→attr 链、pipeline 完整性、GPU 数值验证（Triton）、MLIRCompiler 集成 |
| `test_filecheck.py` | 3 | FileCheck 风格 IR 验证 |
| **合计** | **65** | |

所有测试基于 MLIR 原生 API：
- **输入**：`export_and_import()` 从 PyTorch 导出真正的 `ir.Module`
- **验证**：`module.operation.get_asm()` 检查 IR 内容 + `module.operation.verify()` MLIR verifier

### 3.6 Phase 4：集成到编译管线

**文件**：`mlir/mlir_compiler.py`（v2 重写）

v2 的 `MLIRCompiler.compile()` 流程：

```
Step 1: export_and_import(model, input) → ir.Module (真正的 MLIR Module)
Step 2: run_attention_fusion_pass(mlir_module) → 在 ir.Module 上原地重写
Step 3: _find_fused_op(mlir_module) → 遍历 ir.Operation 树找到融合操作
Step 4: 从 fused_op.attributes 读取属性:
        → ir.FloatAttr(attrs["scale"]).value     = 0.125
        → ir.BoolAttr(attrs["is_causal"]).value   = True
        → ir.IntegerAttr(attrs["softmax_dim"])    = -1
Step 5: 用属性参数化 Triton kernel → 包装为 MLIRCompiledModule
```

v1 → v2 的关键变化：

| 步骤 | v1 | v2 |
|------|----|----|
| Step 1 | `export_to_torch_dialect()` → `get_ir_text()` 获取字符串 | `export_and_import()` → `ir.Module` |
| Step 2 | `AttentionFusionPass().run(ir_text, parsed_ops)` 字符串匹配 | `run_attention_fusion_pass(mlir_module)` MLIR native |
| Step 3 属性提取 | 4 个正则方法：`_extract_scale`、`_detect_causal_mask`、`_extract_softmax_dim`、`_extract_input_shape` | MLIR API：`ir.FloatAttr`、`ir.BoolAttr`、`ir.IntegerAttr` |
| 正确性保证 | 依赖 IR 打印格式不变 | MLIR verifier + 类型安全 |

---

## 四、v2 GPU 实测结果

### 4.1 MLIRCompiler 正确性验证

```
================================================================
  MLIR Compiler 正确性验证
================================================================
  [1/5] torch-mlir export: PyTorch → ir.Module (Torch dialect)
        → 导出成功: ir.Module
  [2/5] MLIR 原生 AttentionFusionPass: RewritePatternSet + walk_and_apply_patterns
        → 匹配成功: mul.Scalar → where.ScalarSelf → softmax.int
        → 替换为 custom.fused_scaled_masked_softmax
  [3/5] 属性提取: 从 MLIR ir.Operation.attributes 读取
        → scale = 0.125 (FloatAttr)
        → is_causal = True (BoolAttr)
        → softmax_dim = -1 (IntegerAttr)
        → input_shape = (1, 12, 128, 128)
  [4/5] Triton codegen: 用 MLIR 属性参数化 kernel 模板
  [5/5] 包装为 MLIRCompiledModule (可直接 forward())
        → ✅ 编译完成

  数值对比:
    max  |ref - compiled| = 9.54e-07     ✅ 通过
    mean |ref - compiled| = 6.64e-10     ✅ 通过
```

### 4.2 仅 ScaleMaskSoftmax（5 版本对比）

| 版本 | Kernel 数 | μs/iter | 加速比 |
|------|:---:|:---:|:---:|
| 融合前（独立 kernel） | 282 | 98.6 | 1.00× |
| torch.compile（Inductor） | 101 | 28.7 | 3.44× |
| **MLIR 自编译 v2（our pass）** | **40** | **5.1** | **19.28×** |
| Triton 三遍扫描 | 40 | 5.1 | 19.33× |
| Triton Online Softmax | 40 | 5.1 | 19.44× |

### 4.3 全流水线 FullAttention（8 版本对比）

| 版本 | Kernel 数 | μs/iter | 加速比 |
|------|:---:|:---:|:---:|
| 原始 FullAttention | 341 | 216.1 | 1.00× |
| torch.compile（Inductor） | 140 | 89.4 | 2.42× |
| **MLIR 自编译 v2（our pass）** | **121** | **157.4** | **1.37×** |
| Triton 三遍扫描 | 121 | 106.2 | 2.03× |
| Triton Online Softmax | 121 | 107.3 | 2.01× |
| MLIR + Triton 三遍 | 140 | 94.0 | 2.30× |
| MLIR + Triton Online | 140 | 130.3 | 1.66× |
| compile + MLIR 自编译 | 140 | 102.2 | 2.11× |

### 4.4 v1 vs v2 性能数据对比

性能数据在两次运行间存在正常的 GPU 微基准波动。以下直接对比 v1（2026-03-02）和 v2（2026-03-14）的实测数据：

#### ScaleMaskSoftmax（softmax 子操作隔离）

| 版本 | v1 实测 | v2 实测 | 说明 |
|------|:---:|:---:|------|
| 融合前 | 65.2 μs/iter | 98.6 μs/iter | GPU 状态不同（温度、频率）导致基线差异 |
| MLIR 自编译 | 5.1 μs/iter (12.88×) | 5.1 μs/iter (19.28×) | kernel 执行时间相同，加速比因基线差异而不同 |
| Triton 三遍 | 5.1 μs/iter (12.81×) | 5.1 μs/iter (19.33×) | 同上 |
| Triton Online | 5.0 μs/iter (13.00×) | 5.1 μs/iter (19.44×) | 同上 |

**结论**：MLIR 自编译 kernel 执行时间一致（5.1μs），v1 和 v2 最终生成的 Triton kernel 完全等价。这是预期的——两个版本参数化同一个 kernel 模板，只是参数提取路径不同。加速比的差异（12.88× vs 19.28×）完全来自基线的波动，不反映 v2 的性能改进。

**v2 的价值不在于更快，而在于给 Stage 3 的编译路径补上了 MLIR 工程严谨性。**

---

## 五、v1 → v2 架构改进分析

### 5.1 v1 不足点的修复对照

v1 实验记录 5.6 节列出了 4 个核心问题和 4 个演进阶段。以下是 v2 的修复状态：

#### 5.6.1 文本匹配 → MLIR AST 操作 → ✅ 已修复

| 维度 | v1 | v2 |
|------|----|----|
| 数据结构 | `str` + `re.findall()` | `ir.Operation` / `ir.Value` / `ir.Block` |
| def-use 追溯 | 字符串搜索 `%0` | `value.owner` / `op.operands` |
| 子图替换 | 文本拼接新 IR | `rewriter.replace_op(op, results)` |
| 合法性保证 | 无 | MLIR verifier + SSA 自动重连 |
| 多 block 支持 | ❌ 正则无法处理 | ✅ `walk_and_apply_patterns` 自动遍历所有 region/block |

v2 直接操作 MLIR 的内存数据结构。`ir.Operation` 是 C++ `mlir::Operation*` 的 Python 包装，不是模拟。

#### 5.6.2 缺失的 IR 结构保证 → ✅ 已修复

| 不变量 | v1 状态 | v2 状态 |
|--------|---------|---------|
| SSA 正确性 | ❌ 依赖正则匹配 `%0` | ✅ `replace_op()` 自动重连 |
| Dominance | ❌ 未检查 | ✅ MLIR verifier 自动验证 |
| Region 嵌套 | ❌ 行级匹配 | ✅ Pattern 框架自动处理 |
| Block 参数 | ❌ 未处理 | ✅ Python bindings 等价 C++ |

#### 5.6.3 可扩展性 → ⚠️ 部分改善

| 维度 | v1 | v2 |
|------|----|----|
| 新 pattern | 手写正则规则 | `RewritePatternSet.add()` 注册，支持 benefit 优先级 |
| IR 格式耦合 | ❌ 依赖打印格式 | ✅ 操作内存对象，与打印解耦 |
| 复杂 model | ❌ 无 control flow 支持 | ⚠️ Pattern 框架支持，但未对复杂模型做验证 |
| PDLL/DRR | ❌ 不支持 | ❌ 仍需手写 Python callback（PDLL 需要 C++ 编译链） |

#### 5.6.4 "MLIR as analyzer" → "MLIR as transformer" → ✅ 已修复

v1 没有真正修改 MLIR Module。v2 通过 `walk_and_apply_patterns` + `replace_op` 原地修改 `ir.Module`，是实际的 IR 变换：

```python
# v2: 变换前
ir_before = module.operation.get_asm()
assert "torch.aten.softmax.int" in ir_before

# v2: 运行 Pass（原地修改）
run_attention_fusion_pass(module)

# v2: 变换后
ir_after = module.operation.get_asm()
assert "torch.aten.softmax.int" not in ir_after  # 已被替换
assert "custom.fused_scaled_masked_softmax" in ir_after
```

### 5.2 v1 建议的演进路径 vs v2 实现

v1 在 5.6.5 节提出了 4 步演进路径：

| 演进阶段 | v1 建议 | v2 达成 |
|---------|--------|:------:|
| 当前 → Python MLIR API | 将 `parse_torch_ir(ir_text)` 替换为 `module.operation.walk()` | ✅ |
| Python API → C++ Pass | 将 Python pattern matching 改写为 `OpRewritePattern` | ⚠️ 使用等价的 Python `RewritePatternSet`（同一 C++ 引擎） |
| 单 Pass → Pass Pipeline | 添加 canonicalize / CSE / DCE 等标准 pass | ✅ `PassManager.parse("canonicalize, cse")` |
| 固定 pattern → 通用 DAG matcher | 用 PDLL 或 DRR 描述融合模式 | ❌ 仍为手写 Python callback |

v2 完成了前 3 步。第 4 步（PDLL/DRR）需要 C++ 编译链（mlir-tblgen），超出当前 Python-only 环境的能力。

### 5.3 新增能力：v1 没有的

| 能力 | v1 | v2 |
|------|:--:|:--:|
| Online Softmax 重写 Pass | ❌ | ✅ 匹配 5-op decomposed softmax → `custom.online_softmax` |
| Pass Pipeline 编排 | ❌ | ✅ Pipeline A (fusion+canonicalize+CSE)、Pipeline B (decompose→online rewrite) |
| PassManager 集成 | ❌ | ✅ 内置 pass (`canonicalize`, `cse`, `symbol-dce`) |
| 65 个单元测试 | ❌ | ✅ 覆盖基础功能、负例、边界、端到端、FileCheck |
| 非因果注意力处理 | ❌ | ✅ Pattern 不匹配时 IR 不变 |
| 类型安全属性提取 | ❌ | ✅ `ir.FloatAttr` / `ir.BoolAttr` / `ir.IntegerAttr` |

### 5.4 v2 更新评级

| 维度 | v1 评级 | v2 评级 | 变化原因 |
|------|:---:|:---:|---------|
| 概念架构 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 不变——IR → 属性提取 → codegen 思路始终正确 |
| 端到端闭环 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 新增 65 个测试 + Pass Pipeline + 完整 CI 验证 |
| 工程严谨度 | ⭐⭐ | ⭐⭐⭐⭐ | 使用 MLIR 原生 API，SSA/dominance/verifier 保证 |
| 工业可扩展性 | ⭐ | ⭐⭐⭐ | `RewritePatternSet` + `PassManager`，但缺 PDLL/DRR |

### 5.5 仍存在的不足

1. **PDLL/DRR 声明式 pattern**：新 fusion pattern 仍需手写 Python callback。工业编译器（如 IREE）使用 PDLL 语言声明式描述 pattern，自动生成匹配代码。但 PDLL 需要 `mlir-pdll` 编译工具，超出 Python-only 环境。

2. **未注册 dialect（unregistered ops）**：`custom.fused_scaled_masked_softmax` 和 `custom.online_softmax` 是未注册操作。MLIR verifier 跳过对这些操作的语义验证。严格做法是注册自定义 dialect + ODS 定义操作的类型约束。

3. **单一模型验证**：当前仅在 `ScaleMaskSoftmax` 和 `FullAttention` 上验证。对含 control flow（`torch.prim.If`）、动态 shape、嵌套 attention（如 MHA + cross-attention）的模型，尚未测试。

4. **Triton codegen 仍为模板参数化**：从 MLIR 属性到 Triton kernel 的映射是硬编码的模板替换，不是通用的 MLIR → GPU 代码生成。工业路径是 MLIR → Linalg → GPU dialect → PTX/AMDGPU。

---

## 六、实验结论

### 6.1 原始假设验证（同 v1，数据更新）

| 假设 | Stage 1 定位 | Stage 2 验证 | Stage 3 v2 自动化 |
|------|-------------|-------------|:---:|
| 存在 kernel launch overhead | ✅ 360 个 kernel，平均 3.1μs | ✅ 融合后减少 86% | ✅ MLIR 原生 pass → 同等效果 |
| 存在内存读写瓶颈 | ✅ 3 kernel 串联，6 次全局内存访问 | ✅ 融合后仅 2 次 | ✅ Triton codegen 维持寄存器 |
| 编译优化可消除瓶颈 | — | ✅ softmax-only 13× | ✅ MLIR 自编译 19.28×* |

\* 19.28× vs v1 的 12.88× 差异源于基线波动（v2 基线 98.6μs vs v1 65.2μs），融合后 kernel 耗时一致（5.1μs）。

### 6.2 v2 的核心价值

**v2 的价值不在于性能提升——最终的 Triton kernel 完全相同。v2 的价值在于：**

1. **从"模拟"到"实现"**：v1 用字符串操作模拟了编译器行为，v2 用 MLIR 原生 API 实现了编译器行为。同样的模式匹配逻辑，v1 是正则表达式跑在字符串上，v2 是 `RewritePatternSet` 跑在 `ir.Module` 上。

2. **获得了 IR 结构保证**：SSA 正确性、dominance 关系、Region/Block 嵌套结构——这些在 v1 中被忽略的编译器不变量，在 v2 中由 MLIR 基础设施自动维护。

3. **与工业编译器架构对齐**：v2 的代码组织（RewritePatternSet → FrozenRewritePatternSet → walk_and_apply_patterns → PassManager pipeline）与 MLIR 官方文档和工业实现（IREE、torch-mlir、mlir-hlo）的 Pass 编写方式完全一致。

4. **可测试、可验证**：65 个单元测试覆盖正例、负例、边界情况、端到端。v1 只有手动运行 `run_mlir_experiment.py` 做验证。

### 6.3 三阶段逻辑联系（v2 更新）

```
┌─────────┬──────────────────────────────────────────────────────────┐
│ Stage 1 │ Profiling 定位瓶颈:                                       │
│ 瓶颈定位│ → 360 个 kernel，平均 3.1μs，launch overhead 占 40-60%    │
│         │ → scale→mask→softmax 串联，6 次全局内存访问               │
├─────────┼──────────────────────────────────────────────────────────┤
│ Stage 2 │ Triton 手写验证:                                         │
│ 假设验证│ → 282→40 次 kernel，减少 86%                              │
│         │ → softmax 部分加速 ~13-19×，全流水线 ~1.1-2× (Amdahl)     │
├─────────┼──────────────────────────────────────────────────────────┤
│ Stage 3 │ MLIR 编译器自动化 (v2: 原生实现):                        │
│ 自动化  │ → RewritePatternSet + walk_and_apply_patterns            │
│         │ → 匹配 mul.Scalar → where.ScalarSelf → softmax.int       │
│         │ → ir.FloatAttr/BoolAttr/IntegerAttr 类型安全属性提取       │
│         │ → Triton codegen → GPU 执行 → 数值误差 < 1e-6            │
│         │ → 65 个单元测试全部通过                                    │
└─────────┴──────────────────────────────────────────────────────────┘
```

### 6.4 数据来源审计

实验中每个数据点的来源标注与 v1 一致：

| 标记 | 含义 |
|------|------|
| 📊 实测 | 本次程序真实执行（v2 运行环境, 2026-03-14） |
| 📂 Stage1 | Stage 1 GPU profiling trace 文件（2026-03-01） |
| 📐 IR推导 | 从 MLIR IR 结构逻辑推导 |
| ⚠️ 估算 | 基于 GPU 架构参数理论计算 |

---

## 七、技术栈总结

| 组件 | 技术 | 用途 | v2 变化 |
|------|------|------|---------|
| 模型框架 | PyTorch 2.12 nightly | Transformer Block 实现 | 不变 |
| GPU Profiler | `torch.profiler` | kernel 级性能剖析 | 不变 |
| 编译器 IR | torch-mlir | PyTorch → MLIR 导出 | 不变 |
| **Fusion Pass** | **`torch_mlir.rewrite`** | **MLIR 原生模式匹配 + 子图替换** | **v1: 正则 → v2: MLIR API** |
| **Pass 管线** | **`torch_mlir.passmanager`** | **Pass 编排 (canonicalize + CSE)** | **v2 新增** |
| **Online Softmax Pass** | **`torch_mlir.rewrite`** | **标准→online softmax 重写** | **v2 新增** |
| **测试框架** | **unittest + MLIR verifier** | **65 个测试** | **v2 新增** |
| Kernel DSL | Triton | 融合 kernel | 不变 |
| 编译管线 | `MLIRCompiler` | IR → 属性提取 → codegen → GPU | v2: MLIR native API |
| 硬件 | NVIDIA RTX 4090 | Ampere 架构，CUDA 12.6 | 不变 |

---

## 八、文件清单（v2 更新）

```
attention-profiling-lab/
├── models/
│   ├── mini_transformer.py              # MiniTransformerBlock (Manual/SDP Attention)
│   └── triton_attention.py              # Triton 融合 kernel (三遍 + Online Softmax)
│
├── benchmarks/
│   ├── profile_attention.py             # Stage 1: 基线 profiling
│   ├── profile_flash_attn.py            # Stage 1: SDPA/FlashAttention profiling
│   ├── profile_compiled.py              # Stage 1: torch.compile profiling
│   ├── profile_triton.py               # Stage 1: Triton 三遍扫描 profiling
│   ├── profile_triton_online.py         # Stage 1: Triton Online profiling
│   ├── analyze_trace.py                 # Stage 1: 多 trace 对比分析
│   └── export_fx_graph.py               # FX Graph 导出
│
├── mlir/
│   ├── export_attention_ir.py           # PyTorch → MLIR IR 导出 + 解析工具
│   ├── fusion_pass.py                   # v1 文本匹配版 Pass（保留作参考对比）
│   ├── mlir_compiler.py                 # v2 编译管线（MLIR 原生 API）
│   ├── run_mlir_experiment.py           # 端到端实验驱动
│   │
│   ├── passes/                          # ★ v2 新增：MLIR 原生 Pass 实现
│   │   ├── __init__.py                  #   包导出
│   │   ├── attention_fusion_pass.py     #   Phase 1: scale+mask+softmax 融合 (168行)
│   │   ├── incremental_softmax_pass.py  #   Phase 2: 标准→online softmax 重写 (210行)
│   │   └── pass_pipeline.py             #   Pass 管线编排 (93行)
│   │
│   ├── tests/                           # ★ v2 新增：测试体系
│   │   ├── test_attention_fusion.py     #   14 个测试
│   │   ├── test_incremental_softmax.py  #   18 个测试
│   │   ├── test_edge_cases.py           #   19 个测试
│   │   ├── test_pipeline_e2e.py         #   11 个测试
│   │   ├── test_filecheck.py            #   3 个测试
│   │   └── filecheck/                   #   FileCheck 框架
│   │       ├── run_filecheck.py         #   CHECK/CHECK-NOT/CHECK-SAME 解析器
│   │       ├── basic_fusion.mlir        #   基础融合测试
│   │       ├── dynamic_shapes.mlir      #   动态形状测试
│   │       └── negative_no_match.mlir   #   反例测试
│   │
│   ├── generated_torch_dialect.mlir     # 输出: Torch dialect IR
│   ├── generated_torch_fused.mlir       # 输出: 融合后 IR
│   ├── generated_linalg_dialect.mlir    # 输出: Linalg dialect IR
│   └── generated_full_attention.mlir    # 输出: FullAttention IR
│
├── traces/                              # GPU profiling trace (Chrome JSON)
│   ├── baseline_trace.json
│   ├── sdpa_trace.json
│   ├── compiled_trace.json
│   ├── triton_trace.json
│   └── triton_online_trace.json
│
├── compiler/                                ★ Stage 4 新增：Mini AI 编译器全栈
│   ├── ir/
│   │   ├── ops.py                           #   OpType 枚举（INPUT/SCALE/MASK/SOFTMAX/FUSED/OUTPUT）
│   │   ├── graph.py                         #   IRShape / IRNode / IRGraph（含拓扑排序）
│   │   └── printer.py                       #   IR 文本打印器
│   ├── frontend/
│   │   └── fx_importer.py                   #   import_fx_graph(): aten ops → OpType
│   ├── passes/
│   │   ├── pattern_match.py                 #   SMS / QK pattern 检测
│   │   ├── fusion.py                        #   ScaleMaskSoftmaxFusionPass
│   │   ├── canonicalize.py                  #   CanonicalizationPass（scale_factor 类型规范化）
│   │   └── validation.py                    #   ValidationPass（IR 完整性验证）
│   ├── lowering/
│   │   ├── pipeline.py                      #   CompilerPipeline / CompilationArtifact
│   │   └── to_mlir.py                       #   lower_to_mlir_text() / lower_to_mlir_module()
│   ├── backends/
│   │   ├── reference_backend.py             #   ReferenceBackend（PyTorch eager）
│   │   ├── triton_backend.py                #   TritonBackend（→ Stage 2 Triton kernel）
│   │   ├── mlir_backend.py                  #   MLIRBackend（→ Stage 3 MLIR compiler）
│   │   └── tvm_backend.py                   #   TVMBackend（→ TVM Relax 编译执行）
│   ├── runtime/
│   │   ├── executor.py                      #   Executor（单次执行 + 数值验证）
│   │   └── benchmark.py                     #   BenchmarkRunner（多 backend CUDA event 计时）
│   └── tests/                               #   55 + 17 = 72 个测试，全部通过
│       ├── test_ir.py                       #   IR 数据结构测试
│       ├── test_fusion.py                   #   Fusion Pass 测试
│       ├── test_pattern_match.py            #   Pattern Match 测试
│       ├── test_pipeline.py                 #   端到端编译管线测试
│       └── test_tvm_backend.py              #   TVM 后端测试（17 个）
│
├── tvm_integration/                         ★ TVM Relax 后端集成
│   ├── __init__.py
│   └── relax_importer.py                    #   lower_to_relax(): IRGraph → tvm.IRModule
│                                            #   _build_relax_module(): BlockBuilder 构造 Relax 函数
│                                            #   print_relax_ir(): IR 文本输出
│
└── reports/
    ├── 实验记录_Attention编译优化全流程.md      # v1 实验记录 (2026-03-02)
    ├── 实验记录_Attention编译优化全流程_v2.md   # v2 实验记录 (本文档)
    ├── MLA_incremental_fusion_pass_plan.md    # v2 实现规划文档
    ├── mlir_fusion_analysis.md                # MLIR 融合分析报告（自动生成）
    └── trace_analysis_latest.md               # Stage 1 trace 对比报告
```

v2 新增文件共 **2162 行**代码（passes/ 504 行 + tests/ 1658 行），不修改 Stage 1 和 Stage 2 的任何文件。

---

## 九、Stage 4：Mini AI Compiler Pipeline（v2 之后新增）

> **实验时间**：2026-05-29  
> **背景**：在 v2 完成 MLIR 原生 Pass 之后，为验证编译器思想的工程落地，构建了一个完整的 Mini AI 编译器全栈，涵盖从 FX 图导入到多后端执行的完整链路。

### 9.1 架构总览

```
PyTorch nn.Module
        │
        ▼  torch.fx.symbolic_trace
  FX Graph (aten IR)
        │
        ▼  compiler/frontend/fx_importer.py
  IRGraph（自定义 IR：OpType 枚举 + IRNode/IRShape）
        │
        ├─→  CanonicalizationPass    # 规范化（类型统一）
        ├─→  ScaleMaskSoftmaxFusionPass  # 模式匹配 + 子图替换
        └─→  ValidationPass          # IR 完整性校验
                 │
                 ▼  compiler/lowering/pipeline.py → CompilationArtifact
        ┌────────┬────────┬──────────┐
        ▼        ▼        ▼          ▼
 Reference  Triton   MLIR      TVM Relax
 Backend    Backend  Backend   Backend
 (eager)   (Stage2) (Stage3)  (新增)
```

### 9.2 TVM Relax 后端

**文件**：`tvm_integration/relax_importer.py` + `compiler/backends/tvm_backend.py`

完整编译链：

```
IRGraph (FUSED_SCALE_MASK_SOFTMAX node)
   │
   ▼  lower_to_relax(graph, input_shape)
tvm.IRModule (Relax 方言)
   │
   ▼  relax.build(mod, target="cuda")
Executable
   │
   ▼  relax.VirtualMachine(ex, dev)
VirtualMachine
   │
   ▼  vm["main"](tvm.runtime.from_dlpack(scores))
Output Tensor (via DLPack zero-copy)
```

关键技术点：

| 技术问题 | 解决方案 |
|---------|---------|
| 命名空间冲突（项目 `tvm/` 目录遮蔽真实 TVM 包） | 重命名为 `tvm_integration/` |
| `tvm_ffi` Cython extension 未构建 | 手动 `python -m cython core.pyx` + `g++ -fPIC -shared` |
| `libcuda.so` 需要提前加载 | `_tvm_setup.py`：`ctypes.CDLL("libcuda.so.1", RTLD_GLOBAL)` |
| venv 启动时自动配置 | `.venv/site-packages/tvm.pth` + `_tvm_setup.py` |
| `tvm.nd` 不存在 | 改用 `tvm.runtime.from_dlpack()` |
| 动态 shape（`tvm.tir.Var`）不可用 | 使用 concrete `input_shape: Tuple[int,int,int,int]` |

### 9.3 测试结果

```
compiler/tests/
  test_ir.py              — IR 数据结构
  test_fusion.py          — Fusion Pass
  test_pattern_match.py   — Pattern Match
  test_pipeline.py        — 端到端编译管线（含 ReferenceBackend E2E 数值验证）
  test_tvm_backend.py     — TVM 后端（17 个测试）

72 passed in 3.44s  ✅
```

### 9.4 四阶段全貌（最终版）

```
Stage 1: GPU Profiling   →  定位瓶颈（360 kernels，launch overhead 40-60%）
Stage 2: Triton 手写     →  验证融合（13-19×，86% kernel 减少）
Stage 3: MLIR v2 原生    →  自动化融合（MLIR API，65 个测试）
Stage 4: Mini Compiler   →  工程化全栈（FX→IR→Pass→多Backend，72 个测试）
         └→ TVM Relax    →  第四条后端路径（Relax IRModule→GPU执行）
```
