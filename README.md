# Attention Profiling Lab

> 通过 GPU Profiling 定位 Transformer 注意力机制中"非算子本身"的性能瓶颈，并验证编译优化（kernel fusion）能否消除这些瓶颈

## 🎯 项目目标

本项目的核心目标是：**通过 GPU Profiling 定位 Transformer Attention 推理中"非算子本身"的性能瓶颈——即调度开销和内存访问模式问题——并验证编译优化能否消除这些瓶颈。**

### 核心假设

> **假设**：Attention 推理中 softmax 相关的子操作（scale、causal_mask、softmax）被 PyTorch eager mode 拆分为多个独立 CUDA kernel。这些 kernel 的**调度开销（kernel launch overhead）**和**中间结果的全局内存反复读写**是可被编译优化消除的瓶颈。

### 三阶段验证路径

```
Stage 1: Profiling    →  用 GPU profiler 定位调度/内存瓶颈
Stage 2: Triton       →  手写融合 kernel 验证 "融合确实能消除瓶颈"
Stage 3: MLIR         →  让编译器自动做同样的事（IR 分析 → 自动 codegen）
```

## 📦 项目结构

```
attention-profiling-lab/
│
├── models/
│   ├── mini_transformer.py       # 可配置的最小 Transformer Block
│   ├── triton_attention.py       # Triton 融合 attention kernel
│   └── __init__.py
│
├── benchmarks/
│   ├── profile_attention.py      # 基线：手写 unfused attention profiling
│   ├── profile_flash_attn.py     # 对比：PyTorch SDPA (FlashAttention)
│   ├── profile_compiled.py       # 对比：torch.compile (Inductor)
│   ├── profile_triton.py         # 对比：Triton 融合 kernel
│   ├── export_fx_graph.py        # FX Graph 导出与分析
│   ├── analyze_trace.py          # 多 trace 对比分析工具
│   └── __init__.py
│
├── mlir/                         # MLIR 融合 Pass 实验
│   ├── export_attention_ir.py    # Attention MLIR IR 导出与解析工具
│   ├── fusion_pass.py            # Attention Fusion Pass (Python 实现)
│   ├── mlir_compiler.py          # MLIR → Triton → GPU 编译管线
│   ├── run_mlir_experiment.py    # MLIR 融合 Pass 端到端实验
│   ├── generated_*.mlir          # 生成的 MLIR IR 文件
│   └── README.md
│
├── traces/                       # profiling 输出 (chrome trace JSON)
├── reports/                      # 性能分析报告
│   └── 实验记录_Attention编译优化全流程.md  # 完整实验记录
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔬 核心实验结果

### 定位到的瓶颈（非算子本身）

| 瓶颈类型 | 量化指标 | 优化后 |
|---------|---------|--------|
| Kernel Launch Overhead | 282 次启动 × 3μs ≈ 846μs | 40 次启动 × 3μs ≈ 120μs（减少 86%）|
| 中间结果内存读写 | 6 次 × 32KB = 192KB/iter | 2 次 × 32KB = 64KB/iter（减少 66%）|

### 验证结论

```
如果 [scale + mask + softmax 融合为单个 kernel]
能通过 [消除 kernel launch overhead + 减少内存读写] 
性能可能提升 [13× 仅 softmax 部分] / [1.12× 全流水线]
```

### 五路对比结果（RTX 4090 实测）

| 版本 | Kernel 启动次数 | 总耗时 (ms) | 加速比 |
|------|:---:|:---:|:---:|
| Baseline（手写 unfused） | 360 | 1.12 | 1.00× |
| torch.compile（Inductor） | 280 | 1.01 | 1.12× |
| SDPA（FlashAttention） | 220 | 0.96 | 1.18× |
| Triton（三遍融合） | 280 | 1.00 | 1.12× |
| Triton（Online Softmax） | 280 | 1.00 | 1.12× |

### 仅 ScaleMaskSoftmax 对比（隔离 softmax 子操作）

| 版本 | Kernel 数 | μs/iter | 加速比 |
|------|:---:|:---:|:---:|
| 融合前（独立 kernel） | 282 | 65.2 | 1.00× |
| torch.compile（Inductor） | 101 | 36.1 | 1.81× |
| **MLIR 自编译（our pass）** | **40** | **5.1** | **12.88×** |
| Triton 三遍扫描 | 40 | 5.1 | 12.81× |
| Triton Online Softmax | 40 | 5.0 | 13.00× |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch nightly 2.12+ (with CUDA support)
- torch-mlir nightly (dev-wheels)
- Triton 3.6.0+
- NVIDIA GPU (compute capability ≥ 7.0 推荐)

### 安装

```bash
git clone https://github.com/<your-username>/attention-profiling-lab.git
cd attention-profiling-lab
python -m venv .venv && source .venv/bin/activate

# PyTorch nightly (CUDA 12.6)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126

# torch-mlir nightly
pip install --pre torch-mlir -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels --no-deps

# 其余依赖
pip install -r requirements.txt
```

### 运行 Profiling

```bash
# 第一步：基线 profiling（手写 unfused attention）
python benchmarks/profile_attention.py

# 第二步：SDPA / FlashAttention profiling
python benchmarks/profile_flash_attn.py

# 第三步：torch.compile profiling
python benchmarks/profile_compiled.py

# 第四步：Triton 融合 kernel profiling
python benchmarks/profile_triton.py

# 对比分析（自动发现所有 trace）
python benchmarks/analyze_trace.py

# FX Graph 导出与分析
python benchmarks/export_fx_graph.py --export_dot

# MLIR 融合 Pass 实验
python mlir/run_mlir_experiment.py
```

### 自定义参数

```bash
python benchmarks/profile_attention.py \
    --seq_len 256 \
    --hidden_size 1024 \
    --num_heads 16 \
    --batch_size 4 \
    --warmup 20 \
    --repeat 50
```

### 查看 Trace

生成的 `traces/*.json` 文件可以用以下方式打开：

- **Chrome**: `chrome://tracing` → Load
- **Perfetto**: [perfetto.dev](https://perfetto.dev/) → Open trace file
- **Nsight Systems**: 导入 `.json` 或直接使用 `nsys profile`

## 🔬 实验设计

### Stage 1：GPU Profiling —— 定位调度与内存瓶颈

| 实验 | 模型 | 说明 |
|------|------|------|
| Baseline | 手写 ManualAttention | 每个 sub-op 独立 kernel，暴露所有瓶颈 |
| SDPA | PyTorch SDPA | FlashAttention 后端，展示 fusion 收益 |
| Compiled | torch.compile | Inductor 编译优化效果 |

**重点观察指标**：
1. **Kernel launch overhead** — 360 个 kernel，平均 3.1μs，launch overhead 占 40-60%
2. **中间结果内存读写** — scale→mask→softmax 串联，6 次全局内存访问
3. **小 kernel 占比** — 100% kernel < 50μs

### Stage 2：Triton 手写融合 —— 验证瓶颈消除

| 实验 | 模型 / 工具 | 说明 |
|------|------|------|
| Triton 融合 | 手写 Triton kernel | 融合 scale + mask + softmax 为 1 个 kernel |
| Online Softmax | Triton Online kernel | 两遍扫描版本，减少 33% 全局内存加载 |
| FX Graph 导出 | torch.export | 导出计算图，识别可融合子图模式 |

**验证结果**：
- Kernel 启动次数：282 → 40（减少 86%）
- 内存读写：6 次 → 2 次（减少 66%）
- softmax 部分加速：**13×**

### Stage 3：MLIR 编译器分析 —— 自动化融合

| 实验 | 工具 | 说明 |
|------|------|------|
| MLIR 导出 | torch-mlir | PyTorch Attention → Torch dialect / Linalg dialect |
| 融合 Pass | Python MLIR Pass | 模式匹配 scale→mask→softmax，替换为融合操作 |
| Triton Codegen | MLIRCompiler | 从 IR 自动提取属性，生成等价 Triton kernel |

**自动化结果**：MLIR 自编译 12.88× ≈ 手写 Triton 12.81×

## 📊 核心结论

### 1. 原始假设验证

| 假设 | Stage 1 定位 | Stage 2 验证 | Stage 3 自动化 |
|------|-------------|-------------|---------------|
| 存在 kernel launch overhead | ✅ 360 个 kernel，launch overhead 占 40-60% | ✅ 融合后减少 86%，加速 13× | ✅ MLIR 自编译达到同等效果 |
| 存在内存读写瓶颈 | ✅ 3 个 kernel 串联，6 次全局内存访问 | ✅ 融合后仅 2 次，减少 66% | ✅ Triton codegen 保持中间结果在寄存器 |

**结论：原始假设完全成立。**

### 2. Amdahl 定律的实际体现

| 测量范围 | 可优化部分占比 | 理论加速上限 | 实测加速 |
|---------|:---:|:---:|:---:|
| 仅 ScaleMaskSoftmax | 100% | ∞ | 13× |
| 完整 Attention | 11.6% | 1.13× | 1.12× |

→ 即使编译优化把某部分加速到极致，整体收益仍受限于该部分的占比。

### 3. 编译优化的价值

- **自动化**：MLIR 路径的价值不在于跑得更快，而在于将人工分析过程自动化为编译器 pass
- **收益本质**：13× 加速比并非算法优化，而是消除调度开销 + 内存访问开销
- **替代关系**：编译器融合与手写 kernel 是替代关系，非叠加关系

## 🛠 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 模型框架 | PyTorch 2.12 nightly | Transformer Block 实现 |
| GPU Profiler | `torch.profiler` | kernel 级性能剖析 |
| 编译器 IR | torch-mlir | PyTorch → MLIR 导出 |
| Fusion Pass | 自建 Python Pass | MLIR IR 上的 pattern matching |
| Kernel DSL | Triton 3.6.0 | scale+mask+softmax 融合 kernel |
| 编译管线 | 自建 `MLIRCompiler` | IR 分析 → Triton codegen → GPU |
| 硬件 | NVIDIA RTX 4090 | Ampere 架构，CUDA 12.6 |

## 📝 详细实验记录

完整的实验记录（包括三阶段设计、实测数据、结论分析）请参阅：

📄 [实验记录_Attention编译优化全流程.md](reports/实验记录_Attention编译优化全流程.md)

## 📝 License

MIT
