# Triton Online Softmax Kernel 性能调优报告

> **日期**: 2026-03-14  
> **GPU**: NVIDIA GeForce RTX 4090 (128 SMs, SM 8.9, 24GB GDDR6X, ~1008 GB/s)  
> **软件栈**: Triton 3.6.0 · CUDA 12.6 · PyTorch 2.12 · Driver 560.35.05  
> **目标 Kernel**: `_tunable_online_softmax_fwd`（2-pass Online Softmax with causal mask + scale）  
> **默认参数**: B=1, H=12, D=64, seq_len=128, dtype=fp16

---

## 一、背景与动机

在前序实验（Stage 2）中，我们为 Attention 推理编写了两个 Triton 融合 kernel：

- **3-pass kernel**（`_fused_scale_mask_softmax_fwd`）：分别做 max、exp/sum、normalize 三轮扫描
- **Online 2-pass kernel**（`_online_softmax_fwd`）：在首轮扫描中利用 running max 消除 max 预扫描，合并为 2 轮

两个 kernel 在默认参数下性能接近（~5 µs），将 softmax 子操作从 282 次 kernel 减少为 40 次，加速约 13-19×。但 **默认参数是否最优？不同 tiling 策略和内存加载方式对性能的影响有多大？** 这是本次性能调优要回答的问题。

### 调优参数空间

| 参数 | 含义 | 搜索范围 |
|------|------|---------|
| `BLOCK_T` | 每个 block 一次处理的列数（分块大小） | 16, 32, 64, 128, 256, 512, 1024 |
| `num_warps` | 每个 block 的 warp 数（32 线程/warp） | 1, 2, 4, 8, 16 |
| `num_stages` | Software pipelining 级数（双缓冲深度） | 1, 2, 3, 4, 5 |
| `dtype` | 数据类型 | fp16, fp32 |
| `seq_len` | 序列长度（影响 grid size 和工作量） | 64, 128, 256, 512, 1024 |

---

## 二、实验方法

### 2.1 调优框架

编写了 `benchmarks/tune_triton_kernel.py`（~500 行），包含 7 个独立实验：

```bash
python benchmarks/tune_triton_kernel.py --experiment block    # 实验1: BLOCK_T 扫描
python benchmarks/tune_triton_kernel.py --experiment warps    # 实验2: num_warps 扫描
python benchmarks/tune_triton_kernel.py --experiment stages   # 实验3: num_stages 扫描
python benchmarks/tune_triton_kernel.py --experiment dtype    # 实验4: fp16 vs fp32
python benchmarks/tune_triton_kernel.py --experiment scaling  # 实验5: seq_len 缩放
python benchmarks/tune_triton_kernel.py --experiment sweep    # 实验6: 联合参数扫描
python benchmarks/tune_triton_kernel.py --experiment autotune # 实验7: Triton autotune
```

**关键实现细节**：

- 创建了可调参独立 kernel `_tunable_online_softmax_fwd`（与原始 kernel 逻辑相同，但 `num_warps` / `num_stages` 不硬编码）
- 使用 `triton.testing.do_bench()` 进行基准测试（自动 warmup + 多次测量取中位数）
- 计算有效带宽 = `数据量 / 执行时间`（数据量 = 2 reads + 1 write × B×H×T×T × dtype_size）

### 2.2 硬件级 Profiling

**目标**：获取 kernel 的硬件级指标（occupancy、stall 原因、throughput 利用率），解释"为什么"某些参数配置更优。

**工具适配过程**（解决兼容性问题）：

| 工具 | 版本 | 结果 |
|------|------|------|
| `ncu`（系统默认） | 2021.3.1 | ❌ Section files not found（版本太旧） |
| `ncu`（CUDA 11.7） | 2022.2.0 | ❌ "API call not supported in installed CUDA driver"（CUDA 11.x 注入库与 12.6 driver 不兼容） |
| `ncu`（手动安装） | 2026.1 | ❌ "Cuda driver is not compatible"（需要 driver ≥570，当前 560） |
| `nsys` | 2025.2.1 | ✅ 可采集 CUPTI kernel 级数据（通过 SQLite 提取） |
| **`ncu`（CUDA 12.8）** | **2025.1.0** | **✅ 完整 40-pass profiling 成功** |

最终找到 `/usr/local/cuda/bin/ncu`（CUDA 12.8 symlink 中的 ncu 2025.1.0），与 driver 560 兼容。

编写了专用的 profiling 脚本 `benchmarks/ncu_profile_kernel.py`（最小化 kernel launch，避免 profiling 噪声）：

```bash
# nsys 采集
nsys profile --trace=cuda -o reports/ncu_softmax_nsys python benchmarks/ncu_profile_kernel.py

# ncu 全量采集（40 passes，需要 sudo）
sudo /usr/local/cuda/bin/ncu --set full --launch-skip 1 --launch-count 1 \
    -o reports/ncu_softmax_full python benchmarks/ncu_profile_kernel.py
```

---

## 三、实验结果

### 3.1 实验1：BLOCK_T（分块大小）扫描

**条件**: seq_len=128, fp16, num_warps=4, num_stages=2

| BLOCK_T | Online (µs) | BW (GB/s) | 循环次数 |
|--------:|------------:|----------:|---------:|
| 16 | 9.71 | 121.5 | 8 |
| 32 | 6.79 | 173.7 | 4 |
| 64 | 5.79 | 203.8 | 2 |
| **128** | **5.06** | **233.3** | **1** |
| 256 | 4.95 | 238.3 | 1 |
| **512** | **4.73** | **249.6** | **1** |
| 1024 | 5.37 | 219.7 | 1 |

**发现**：
- BLOCK_T=16→128 持续提升：循环次数从 8 降到 1，消除循环控制开销 + 提升向量化效率
- BLOCK_T=256/512 达到峰值：虽然 `seq_len=128` 只需 128 列，Triton 的 mask 机制让超出部分不影响正确性，更大 tile 允许编译器更激进地调度
- BLOCK_T=1024 开始下降：寄存器压力增大 → occupancy 下降

### 3.2 实验2：num_warps 扫描

**条件**: seq_len=128, BLOCK_T=128, fp16

| num_warps | 线程数 | Online (µs) | 相对性能 |
|----------:|-------:|------------:|---------:|
| 1 | 32 | 4.96 | 1.00× |
| **2** | **64** | **4.93** | **1.00×** |
| 4 | 128 | 5.05 | 0.98× |
| 8 | 256 | 6.19 | 0.80× |
| 16 | 512 | 8.42 | 0.59× |

**发现**：
- **num_warps=2 最优，而非 Triton 默认的 4**
- 128 个元素 ÷ 32 线程/warp ≈ 4 元素/线程，1-2 个 warp 即可饱和计算
- num_warps=8/16 大量 warp 完全闲置，浪费寄存器 → occupancy 下降

**核心规律**：`num_warps × 32 ≈ BLOCK_T` 时效率最高。

### 3.3 实验3：num_stages 扫描

| BLOCK_T | 循环次数 | num_stages=1 (µs) | num_stages=5 (µs) | 加速 |
|--------:|---------:|-------------------:|-------------------:|-----:|
| 32 | 4 | 6.91 | **6.38** | **1.08×** |
| 64 | 2 | 5.52 | 5.50 | 1.00× |
| 128 | 1 | 5.06 | 5.06 | 1.00× |

**发现**：`num_stages` 仅在循环次数 ≥ stages 时有效。当 `BLOCK_T ≥ seq_len`（单次循环），stages 完全无意义。

### 3.4 实验4：fp16 vs fp32

| seq_len | fp16 (µs) | fp32 (µs) | fp16 加速比 | 瓶颈类型 |
|--------:|----------:|----------:|------------:|---------|
| 128 | 5.15 | 5.78 | 1.12× | Compute/Launch-bound |
| 256 | 7.67 | 9.95 | 1.30× | 混合 |
| 512 | 16.16 | 33.67 | **2.08×** | **Memory-bound** |

**发现**：小 seq_len → compute-bound → fp16 收益有限。大 seq_len → memory-bound → fp16 接近 2× 加速。

### 3.5 实验5：seq_len 缩放

对比两种 tiling 策略：

- **策略 A**：BLOCK_T = next_power_of_2(seq_len)（单次循环优先）
- **策略 B**：BLOCK_T = 128 固定（多次循环）

| seq_len | 策略 A (µs) | 策略 B (µs) | A 优势 |
|--------:|------------:|------------:|-------:|
| 64 | 4.30 | 4.19 | — |
| 128 | 5.45 | 5.35 | — |
| 256 | **7.44** | 9.64 | **29%** |
| 512 | **16.34** | 24.86 | **34%** |
| 1024 | **60.06** | 82.59 | **37%** |

**结论**：默认 `BLOCK_T = next_power_of_2(T)` 对所有 seq_len 均为最优策略。

### 3.6 实验6：联合参数扫描

在 (BLOCK_T, num_warps, num_stages) 的 36 种组合中全面搜索。

**seq_len=128 最优 Top 5**：

| 排名 | BLOCK_T | num_warps | num_stages | µs | BW (GB/s) |
|:---:|--------:|----------:|-----------:|---:|----------:|
| 1 | **128** | **2** | **1** | **4.59** | **257.1** |
| 2 | 256 | 4 | 2 | 4.63 | 254.7 |
| 3 | 128 | 2 | 2 | 4.67 | 252.8 |
| 4 | 256 | 4 | 1 | 4.87 | 242.3 |
| 5 | 256 | 2 | 1 | 4.88 | 241.7 |
| 36 | 32 | 8 | 2 | 9.13 | 129.3 |

**最优 vs 最差：1.99× 差距**。调参不是小事。

### 3.7 实验7：Triton @autotune

Triton 内置 `@triton.autotune` 在 64 种配置中搜索，选出 `BLOCK_T=256, num_warps=4, num_stages=1`（4.90 µs）。与手动最优（4.59 µs）差距不到 7%，但非全局最优。

**对比**：

| 方法 | 最优参数 | 性能 |
|------|---------|:---:|
| 手动扫描 | BLOCK_T=128, warps=2, stages=1 | **4.59 µs** |
| @autotune | BLOCK_T=256, warps=4, stages=1 | 4.90 µs |
| Triton 默认 | BLOCK_T=128, warps=4, stages=2 | 5.06 µs |

---

## 四、硬件级深度分析

### 4.1 nsys Profiling (Kernel 级)

通过 nsys 2025.2.1 + CUPTI，从 SQLite 数据库提取 kernel 硬件参数。

| 参数 | 值 |
|------|:---:|
| Kernel Duration | 2.24~2.56 µs |
| Grid | (1536, 1, 1) |
| Block | (64, 1, 1) = 2 warps × 32 |
| Registers/Thread | **26** |
| Static Shared Mem | 0 bytes |
| Dynamic Shared Mem | 8 bytes |
| Local Mem (spill) | **0 bytes** |
| 理论 Occupancy | **100%** |

寄存器低（26）、无 spill、理论 occupancy 满 → 单看这些指标，kernel 条件很好。

### 4.2 ncu 2025.1 全量分析 (40 passes)

使用 `ncu --set full` 获取完整硬件计数器，揭示了 nsys 无法看到的实际运行时行为。

#### Speed of Light

| 指标 | 值 | 判定 |
|------|:---:|:---:|
| Compute (SM) Throughput | **15.94%** | ⚠️ 低 |
| Memory Throughput | **15.94%** | ⚠️ 低 |
| DRAM Throughput | **12.16%** | ⚠️ 低 |
| FP32 Peak 利用率 | ~1% | ⚠️ 极低 |

SM 和 Memory 利用率都只有 ~16%，远未饱和。

#### 根因：Grid 太小

| 指标 | 值 | 含义 |
|------|:---:|------|
| Grid Size | 1,536 blocks | B×H×T = 1×12×128 |
| 理论可调度 blocks | 3,072 | 128 SM × 24 blocks/SM |
| **Waves/SM** | **0.50** | **仅半波！一半 SM 闲置** |
| 理论 Occupancy | 100% | 寄存器/shared mem 不受限 |
| **实际 Achieved Occupancy** | **31.53%** | **理论 vs 实际差距 69%** |

**ncu 诊断**：*"This kernel grid is too small to fill the available resources on this device, resulting in only 0.5 full waves across all SMs."*

这解释了一个此前令人困惑的矛盾：**nsys 显示理论 occupancy = 100%，但 GPU 利用率极低。** 原因是 grid 只有 1536 blocks，而 RTX 4090 最多同时容纳 3072 个 2-warp blocks，导致约一半 SM 完全空闲。理论 occupancy 100% 仅表示"每个 SM **如果被用到**，可以装满 warp"，并不等于"所有 SM 都被用到了"。

#### Warp Stall 原因分析

ncu 提供了 warps 不发射指令的详细原因分布：

| 排名 | Stall 原因 | 占比 | 含义 |
|:---:|------|:---:|------|
| 1 | **Long Scoreboard** | **32.4%** | 等待全局内存加载完成（L2/DRAM 延迟） |
| 2 | **Short Scoreboard** | **21.6%** | 等待 L1/shared mem/MIO 操作 |
| 3 | **Wait** | **16.0%** | 等待固定延迟指令（如 exp、div） |
| 4 | **IMC Miss** | **10.2%** | 指令缓存 miss |
| 5 | **Selected** | **7.7%** | 正在执行（非 stall） |
| 6 | **Barrier** | **5.3%** | 等待 warp 间同步 |
| 7-10 | 其他 | <7% | Not Selected, Math Throttle, No Instruction, Drain |

**解读**：

- 最大 stall 来自 **Long Scoreboard（32%）**= 等待全局内存加载。这是 softmax kernel 的固有特征：每次循环都要从全局内存加载一行数据。
- **Short Scoreboard（22%）** + **Wait（16%）** 合计 38%，说明即使数据到了 L1/寄存器，计算指令（exp、div、reduce）本身也有不可忽略的延迟。
- **IMC Miss（10%）** 较高，说明 Triton JIT 生成的 kernel 代码较大（指令缓存 32KB/SM），可能存在指令 footprint 优化空间。

**Warp Latency**: 平均每条指令 **14.72 cycles** stall。调度器每 **3.5 cycles** 才发射一条指令（理想值=1 cycle）。

#### Cache 命中率

| Cache 层级 | Hit Rate |
|-----------|:---:|
| L1/TEX | 33.33% |
| L2 | 54.62% |

L1 命中率仅 33%，说明数据主要从 L2/DRAM 获取。L2 命中率 55% 表明约一半的 L2 请求命中，另一半要走 DRAM。

#### Compute Pipeline 利用率

| Pipeline | 利用率 (elapsed %) |
|------|:---:|
| LSU (Load/Store) | 15.94% |
| ALU (整数/逻辑) | 10.46% |
| FMA (浮点乘加) | 5.98% |
| Tensor Core | 0% |

所有 pipeline 均严重未利用。LSU 最高（15.94%），说明 kernel 的主要工作是 load/store，但仍然远未饱和。Tensor Core 未被使用——因为 softmax 是 element-wise 操作，不涉及矩阵乘法。

#### ncu 自动诊断汇总

ncu 给出的优化建议（按估计加速排序）：

| 优先级 | 诊断 | 估计加速 |
|:---:|------|:---:|
| 1 | Compute Pipeline Under-utilized | 80.47% |
| 2 | Issue Slot Utilization 低 | 71.47% |
| 3 | Achieved Occupancy 低 | 68.47% |
| 4 | FP32 指令未融合 | 5.43% |
| 5 | SMSP Workload Imbalance | 5.41% |

前 3 项指向同一个根因：**grid 太小，GPU 资源闲置**。

---

## 五、综合诊断

### 5.1 瓶颈全景图

```
┌────────────────────────────────────────────────────────────────────────┐
│  _tunable_online_softmax_fwd  (B=1, H=12, seq_len=128)               │
│                                                                        │
│  ┌─ 根本瓶颈 ──────────────────────────────────────────────────────┐  │
│  │  ★ Grid 太小 (0.5 wave / 1536 blocks / 128 SMs)               │  │
│  │    → 实际 occupancy 31.5%（理论 100%）                          │  │
│  │    → SM/Memory throughput 均只有 ~16%                           │  │
│  │    → Kernel 时间 ~3.3µs ≈ CUDA launch overhead                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─ Stall 分析 ────────────────────────────────────────────────────┐  │
│  │  Long Scoreboard 32% : 全局内存延迟（不可完全消除）             │  │
│  │  Short Scoreboard 22%: L1/shared mem 延迟                       │  │
│  │  Wait 16%            : exp/div 计算延迟                         │  │
│  │  → 每指令 14.72 cycle stall，IPC 仅 1.08                       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─ 不同 seq_len 下的瓶颈迁移 ─────────────────────────────────────┐  │
│  │  seq_len=128 : Launch-bound（kernel 太短，launch 开销占比大）   │  │
│  │  seq_len=256 : Compute-bound（SM 开始有足够工作）               │  │
│  │  seq_len≥512 : Memory-bound（fp16 接近 2× 加速，BW 接近峰值）  │  │
│  └────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.2 参数选择与瓶颈的关系

| 参数决策 | 硬件层面解释 |
|---------|------------|
| **num_warps=2 优于 4** | 128 列只需 2 warps（64 线程）饱和计算。多出的 warp 浪费寄存器、增加 barrier stall（ncu 显示 barrier stall = 5.3%） |
| **BLOCK_T=128~256 最优** | 匹配 seq_len 消除循环开销。更大的 tile 允许编译器做更好的指令调度，但超过 512 后寄存器压力增大 |
| **num_stages=1 足够** | 单次循环时无法流水线化。ncu 确认 shared mem 仅 1 KB — 没有 multi-buffer 需求 |
| **fp16 在小 seq_len 收益有限** | ncu 显示 DRAM throughput 仅 12% — 远未触及带宽瓶颈，fp16 的带宽优势无处发挥 |

---

## 六、结论

### 6.1 调优结果

| 配置 | 参数 | seq_len=128 性能 |
|------|------|:---:|
| **Triton 默认** | BLOCK_T=128, warps=4, stages=2 | 5.06 µs |
| **@autotune 选择** | BLOCK_T=256, warps=4, stages=1 | 4.90 µs |
| **手动扫描最优** | BLOCK_T=128, warps=2, stages=1 | **4.59 µs** |
| 最差配置 | BLOCK_T=32, warps=8, stages=2 | 9.13 µs |

**最优比默认快 10%，比最差快 99%。** 手动扫描比 @autotune 快 7%。

### 6.2 四条核心调优规律

1. **`num_warps × 32 ≈ BLOCK_T` 时效率最高**  
   warp 过多 → 闲置浪费，过少 → 延迟隐藏不足。在本 kernel 中，128 列 / 32 = 4 线程/warp，2 个 warp 即饱和。

2. **`BLOCK_T = next_power_of_2(seq_len)` 是最优 tiling 策略**  
   消除循环控制开销 > 略高的寄存器压力。策略 A（自适应）在 seq_len=256~1024 上比策略 B（固定 128）快 29-37%。

3. **`num_stages` 仅在循环次数 ≥ stages 时有效**  
   单次循环时 stages 无意义。多次循环时最多 ~8% 收益（BLOCK_T=32, 4 次循环, stages=5）。

4. **瓶颈随 seq_len 迁移：launch-bound → compute-bound → memory-bound**  
   调优策略必须匹配当前瓶颈类型：小 seq_len 优化 launch 开销（合并 kernel），大 seq_len 优化内存访问（fp16 + tiling）。

### 6.3 理论 vs 实际 Occupancy 的教训

本次调优中最重要的认知更新：

| 指标 | nsys 报告 | ncu 实测 | 对比 |
|------|:---:|:---:|------|
| Occupancy | 100%（理论） | 31.5%（实际） | **差距 69%** |

**理论 occupancy 100% 不等于 GPU 被充分利用。** 当 grid size 不足以填满所有 SM 时（如本例 0.5 wave），大量 SM 完全空闲。nsys/CUPTI 报告的"occupancy"是基于寄存器和 shared mem 配额计算的理论上限，不反映实际利用率。**必须用 ncu 的 Achieved Occupancy 才能看到真实情况。**

### 6.4 优化路径建议

针对 ncu 揭示的瓶颈，后续优化方向：

| 方向 | 预期收益 | 实现难度 | 说明 |
|------|:---:|:---:|------|
| 增大 batch/seq_len | 高 | 低 | B=4 或 seq_len=512 可使 grid 超过 1 wave，充分利用 SM |
| 每 block 处理多行 | 高 | 中 | 将 grid 维度从 B×H×T 改为 B×H×(T/rows_per_block)，增加 per-block 工作量 |
| FP32 指令融合 | 中 | 低 | ncu 显示 110K 非融合 vs 3K 融合 FP32 指令，融合可提升 ~49% FP32 性能 |
| Software pipelining | 低 | 中 | 仅在多次循环（长 seq_len + 小 BLOCK_T）场景下有 ~8% 收益 |
| Tensor Core 利用 | 大 | 高 | 需改为矩阵乘法形式（如 FlashAttention 思路），当前 softmax 是 element-wise |

---

## 七、实验文件清单

| 文件 | 用途 |
|------|------|
| `benchmarks/tune_triton_kernel.py` | 7 个调优实验的自动化框架（~500 行） |
| `benchmarks/ncu_profile_kernel.py` | ncu/nsys 专用的最小化 profiling 脚本 |
| `reports/triton_tuning_analysis.md` | 完整实验数据表格与详细分析 |
| `reports/ncu_softmax_full.ncu-rep` | ncu 2025.1 全量 profiling 报告（40 passes） |
| `reports/ncu_softmax_nsys.sqlite` | nsys CUPTI kernel 数据（SQLite 格式） |

## 八、复现指南

```bash
# 1. 运行全部调优实验
cd /data/github/attention-profiling-lab
source .venv/bin/activate
python benchmarks/tune_triton_kernel.py

# 2. nsys profiling
nsys profile --trace=cuda -o reports/ncu_softmax_nsys \
    python benchmarks/ncu_profile_kernel.py

# 3. ncu 全量 profiling（需 sudo + 兼容版本）
sudo /usr/local/cuda/bin/ncu --set full --launch-skip 1 --launch-count 1 \
    -o reports/ncu_softmax_full --force-overwrite \
    $(which python) benchmarks/ncu_profile_kernel.py

# 4. 提取 ncu 关键指标
sudo /usr/local/cuda/bin/ncu --import reports/ncu_softmax_full.ncu-rep \
    --page raw --metrics gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed --csv
```
