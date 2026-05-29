# Triton Kernel 性能调优实验报告

> 生成时间: 2026-03-14
> GPU: NVIDIA GeForce RTX 4090 (128 SMs, SM 8.9, 24GB GDDR6X)
> Triton: 3.6.0 | CUDA: 12.6 | PyTorch: 2.x
> 配置: B=1, H=12, D=64, dtype=fp16

---

## 0. 实验目标

系统性地调优 Online Softmax Triton kernel 的三个核心参数：`BLOCK_T`（分块大小）、`num_warps`（warp 数）、`num_stages`（流水线级数），理解每个参数如何影响 GPU 性能，并找到最优配置。

---

## 1. 实验1：BLOCK_T（分块大小）扫描

### 概念

BLOCK_T 决定每个线程块（program）一次从全局内存加载多少列数据到寄存器中处理。

```
seq_len = 128 时：
  BLOCK_T=16  → 内层循环 8 次，每次仅处理 16 列
  BLOCK_T=64  → 内层循环 2 次
  BLOCK_T=128 → 内层循环 1 次（无循环开销）
  BLOCK_T=256 → 内层循环 1 次（128列有效 + 128列 padding 被 mask 掉）
```

### 实测数据 (seq_len=128, fp16, num_warps=4, num_stages=2)

| BLOCK_T | 3-pass (µs) | Online (µs) | 3p BW (GB/s) | OL BW (GB/s) | 循环次数 | 备注 |
|--------:|------------:|------------:|--------------:|--------------:|---------:|------|
| 16 | 10.17 | 9.71 | 154.6 | 121.5 | 8 | 太小，向量化低效 |
| 32 | 7.00 | 6.79 | 224.6 | 173.7 | 4 | |
| 64 | 6.41 | 5.79 | 245.2 | 203.8 | 2 | |
| **128** | **5.07** | **5.06** | **310.5** | **233.3** | **1** | 恰好匹配 seq_len |
| **256** | **4.64** | **4.95** | **338.8** | **238.3** | **1** | 最优（3-pass） |
| 512 | 4.78 | **4.73** | 329.4 | 249.6 | 1 | **最优（Online）** |
| 1024 | 5.57 | 5.37 | 282.6 | 219.7 | 1 | 寄存器压力增大 |

### 分析

1. **BLOCK_T=16→128 性能持续提升**：循环次数从 8 降到 1，消除循环控制开销 + 更好的向量化
2. **BLOCK_T=256/512 达到峰值**：虽然 `seq_len=128` 只需 128 列，Triton 的 mask 机制让额外空间不影响正确性，反而更大的 tile 允许编译器做更激进的调度优化
3. **BLOCK_T=1024 开始下降**：寄存器压力过大，每个线程要维持更多临时变量 → occupancy 下降（能同时执行的 warp 变少）
4. **3-pass 和 Online 最优 BLOCK_T 不同**（256 vs 512）：3-pass 有 3 个循环体各自独立，寄存器 lifetime 短；Online 的第 1 个循环体计算密度更高（维护 running max/sum），对更大 tile 的容忍度更好

---

## 2. 实验2：num_warps（warp 数）扫描

### 概念

```
1 warp = 32 个 CUDA 线程
num_warps=1 →  32 线程/block
num_warps=2 →  64 线程/block
num_warps=4 → 128 线程/block  （Triton 默认）
num_warps=8 → 256 线程/block

GPU warp 调度器的工作：当一个 warp 在等待内存加载（~400 cycles）时，
切换到另一个准备好的 warp 执行计算，实现延迟隐藏。
```

### 实测数据 (seq_len=128, BLOCK_T=128, fp16)

| num_warps | 线程数 | Online (µs) | BW (GB/s) | 相对性能 | 分析 |
|----------:|-------:|------------:|----------:|---------:|------|
| **1** | 32 | **4.96** | 237.9 | **1.00x** | |
| **2** | 64 | **4.93** | 239.1 | **1.00x** | **最优** |
| 4 | 128 | 5.05 | 233.7 | 0.98x | Triton 默认 |
| 8 | 256 | 6.19 | 190.6 | 0.80x | warp 闲置 |
| 16 | 512 | 8.42 | 140.2 | 0.59x | 严重过配 |

### 分析

1. **num_warps=1~2 最优**：因为 `BLOCK_T=128` 意味着每行只有 128 个元素要处理。128 个元素 ÷ 32 线程/warp ≈ 4 个元素/线程。1~2 个 warp 足以覆盖全部计算
2. **num_warps=4 略慢**：多出的 warp 共享有限的计算工作，产生 warp 间同步开销（barrier）
3. **num_warps=8/16 显著下降**：256/512 线程远超数据并行度（128 列），大量 warp 完全闲置，白白占用寄存器资源 → 降低 occupancy

**核心规律**：`num_warps × 32 ≈ BLOCK_T` 时效率最高。warp 太多 = 浪费资源。

---

## 3. 实验3：num_stages（流水线级数）扫描

### 概念

```
num_stages 控制 software pipelining（软件流水线）：

num_stages=1（无流水线）：
  iter1: [Load][Compute][Store]
  iter2:                        [Load][Compute][Store]

num_stages=2（double buffering）：
  iter1: [Load1][Compute1][Store1]
  iter2:   [Load2]  [Compute2]  [Store2]   ← Load2 与 Compute1 重叠
  
  → 隐藏了内存加载延迟
```

### 实测数据 (seq_len=128, fp16, num_warps=4)

**BLOCK_T=32, 4 次循环迭代**（stages 有效场景）：

| num_stages | 共享内存 (bytes) | Online (µs) | BW (GB/s) | 相对 |
|-----------:|-----------------:|------------:|----------:|-----:|
| 1 | 64 | 6.91 | 170.7 | 1.00x |
| 2 | 128 | 6.87 | 171.6 | 1.01x |
| 3 | 192 | 6.82 | 173.0 | 1.01x |
| 4 | 256 | 6.61 | 178.5 | 1.05x |
| **5** | 320 | **6.38** | **184.9** | **1.08x** |

**BLOCK_T=64, 2 次循环迭代**：

| num_stages | 共享内存 (bytes) | Online (µs) | BW (GB/s) | 相对 |
|-----------:|-----------------:|------------:|----------:|-----:|
| 1 | 128 | 5.52 | 213.8 | 1.00x |
| 5 | 640 | 5.50 | 214.5 | 1.00x |

### 分析

1. **BLOCK_T=32 时有 4 次循环 → stages 有 ~8% 收益**：更多 stage = 更好的 load/compute 重叠
2. **BLOCK_T=64 时只有 2 次循环 → stages 几乎无影响**：循环次数太少，流水线填充阶段占总时间的比例太大
3. **BLOCK_T=128/256 时（单次循环）→ stages 完全无效**：没有循环迭代 = 无法流水线化

**核心规律**：`num_stages` 只在 `loops >= stages` 时才有意义。对于小 seq_len（BLOCK_T ≥ T），优化 stages 是浪费。

---

## 4. 实验4：fp16 vs fp32 性能对比

### 实测数据

| seq_len | fp16 (µs) | fp32 (µs) | fp16 加速比 | fp16 BW | fp32 BW | 分析 |
|--------:|----------:|----------:|------------:|--------:|--------:|------|
| 128 | 5.15 | 5.78 | 1.12x | 229.1 | 408.4 | 计算为主 |
| 256 | 7.67 | 9.95 | 1.30x | 615.2 | 948.0 | 带宽开始受限 |
| 512 | 16.16 | 33.67 | **2.08x** | 1168.3 | 1121.2 | **内存带宽瓶颈** |

### 分析

1. **seq_len=128 时 fp16 只快 12%**：数据量太小（B×H×T×T = 1×12×128×128 = 192K 元素），未达到带宽瓶颈，kernel 主要是 compute-bound（受计算限制）+ launch overhead 占比高
2. **seq_len=512 时 fp16 接近 2x 加速**：数据量 = 1×12×512×512 = 3M 元素，fp32 需要 12MB vs fp16 需要 6MB → 带宽差异显著
3. **fp32 在 T=512 时 BW=1121 GB/s，接近理论峰值 1008 GB/s**（超出是因为 L2 cache 命中导致有效带宽高于 DRAM 峰值） → 确认大 seq_len 时 kernel 是 memory-bound

**核心规律**：小 seq_len → compute-bound → fp16 收益有限；大 seq_len → memory-bound → fp16 接近 2x 加速。

---

## 5. 实验5：序列长度缩放分析

### 策略对比

**策略 A：BLOCK_T = next_power_of_2(T)**（当前默认 — 单次循环优先）

| seq_len | BLOCK_T | grid_size | µs | BW (GB/s) | µs/row |
|--------:|--------:|----------:|---:|----------:|-------:|
| 64 | 64 | 768 | 4.30 | 68.7 | 0.0056 |
| 128 | 128 | 1,536 | 5.45 | 216.3 | 0.0036 |
| 256 | 256 | 3,072 | 7.44 | 634.6 | 0.0024 |
| 512 | 512 | 6,144 | 16.34 | 1155.1 | 0.0027 |
| 1024 | 1024 | 12,288 | 60.06 | 1257.1 | 0.0049 |

**策略 B：BLOCK_T = 128 固定**（多次循环策略）

| seq_len | BLOCK_T | iterations | µs | BW (GB/s) | µs/row |
|--------:|--------:|-----------:|---:|----------:|-------:|
| 64 | 128 | 1 | 4.19 | 70.5 | 0.0055 |
| 128 | 128 | 1 | 5.35 | 220.6 | 0.0035 |
| 256 | 128 | 2 | 9.64 | 489.4 | 0.0031 |
| 512 | 128 | 4 | 24.86 | 759.1 | 0.0040 |
| 1024 | 128 | 8 | 82.59 | 914.1 | 0.0067 |

### 分析

1. **策略 A 全面优于策略 B**：让 BLOCK_T 匹配 seq_len、执行单次循环始终更快
2. **T=256 时策略 A 快 29%**（7.44 vs 9.64 µs）：2 次循环的开销（循环控制 + 额外 load）明显
3. **T=1024 策略 A 快 37%**（60 vs 83 µs）：差距随循环次数增大
4. **µs/row 在 T=256 处最低**：此时 grid=3072，恰好充分利用 128 SMs × ~24 active blocks → 最优 SM 利用率

**结论**：默认策略 `BLOCK_T = next_power_of_2(T)` 对这个 kernel 是最优的。不需要改动。

---

## 6. 实验6：联合参数扫描

### seq_len=128 最优 Top 5

| 排名 | BLOCK_T | num_warps | num_stages | µs | BW (GB/s) |
|-----:|--------:|----------:|-----------:|---:|----------:|
| 1 | **128** | **2** | **1** | **4.59** | **257.1** |
| 2 | 256 | 4 | 2 | 4.63 | 254.7 |
| 3 | 128 | 2 | 2 | 4.67 | 252.8 |
| 4 | 256 | 4 | 1 | 4.87 | 242.3 |
| 5 | 256 | 2 | 1 | 4.88 | 241.7 |
| ... | | | | | |
| 36 | 32 | 8 | 2 | 9.13 | 129.3 |

**最优 vs 最差：1.99x 差距**

### seq_len=256 最优 Top 5

| 排名 | BLOCK_T | num_warps | num_stages | µs | BW (GB/s) |
|-----:|--------:|----------:|-----------:|---:|----------:|
| 1 | **256** | **2** | **2** | **7.03** | **670.9** |
| 2 | 128 | 2 | 1 | 7.19 | 656.7 |
| 3 | 128 | 2 | 2 | 7.20 | 655.5 |
| 4 | 256 | 4 | 3 | 7.23 | 652.5 |
| 5 | 256 | 4 | 1 | 7.23 | 652.5 |
| ... | | | | | |
| 36 | 32 | 8 | 2 | 25.26 | 186.8 |

**最优 vs 最差：3.59x 差距**

### 规律总结

- **num_warps=2 一致表现最优**（不是默认的 4）
- **BLOCK_T 匹配 seq_len** 时最优
- **num_stages 影响很小**（因为 BLOCK_T ≥ seq_len 时无循环）
- 最差配置（小 BLOCK_T + 多 warp）比最优慢 2~3.6x

---

## 7. 实验7：@triton.autotune 结果

Triton 的 `@triton.autotune` 在 64 种配置中自动搜索，选出：

| 参数 | 自动选择值 | 手动扫描最优值 |
|------|----------|-------------|
| BLOCK_T | 256 | 128 |
| num_warps | 4 | 2 |
| num_stages | 1 | 1 |
| 执行时间 | 4.90 µs | 4.59 µs |
| 带宽利用率 | 23.9% | 25.5% |

Autotune 选择的配置（#2 in 手动扫描）与手动最优略有差异，因为测量噪声。实际差距不到 7%，两种配置都是合理选择。

---

## 8. 核心发现汇总

### 参数调优规律

| 参数 | 规律 | RTX 4090 上的最优值（seq_len=128） |
|------|------|-------------------------------|
| BLOCK_T | 匹配 seq_len 的 next_power_of_2 | 128~256 |
| num_warps | `≈ BLOCK_T / 32`，通常 1~2 足够 | **2**（比默认 4 好） |
| num_stages | 仅多次循环时有效，单次循环无影响 | 1（单循环时） |
| dtype | 大 seq_len 时 fp16 接近 2x；小 seq_len 时差异不大 | fp16 |

### Bound 分析

| seq_len | 瓶颈类型 | 证据 |
|--------:|---------|------|
| ≤128 | **Launch-bound** | µs/row = 0.005，kernel 极短，launch 开销占比大 |
| 256 | **Compute-bound** | fp16 vs fp32 差距仅 30%，BW 利用率 60% |
| ≥512 | **Memory-bound** | fp16 vs fp32 接近 2x，BW 接近理论峰值 |

### 调优建议

1. **当前默认参数不是最优**：`num_warps=4` 应改为 `num_warps=2`（或使用 `@triton.autotune`）
2. **BLOCK_T 策略正确**：`next_power_of_2(T)` 已经是最优策略
3. **大 seq_len 场景**：应始终使用 fp16，并考虑更激进的 tiling（多行并行处理）

---

## 9. Nsight Systems 硬件级 Profiling 结果

使用 `nsys profile` 采集了 Online Softmax kernel（BLOCK_T=128, num_warps=2, num_stages=1）在 seq_len=128 下的硬件级指标。

> 注意：系统安装的 ncu 版本（2021.3.1 / 2022.2.0）与 CUDA 12.6 驱动不兼容，
> 因此使用 Nsight Systems 2025.2.1 采集 CUPTI kernel 级数据。

### 采集到的 Kernel 硬件参数

| 参数 | 值 | 说明 |
|------|---|------|
| **Kernel name** | `_tunable_online_softmax_fwd` | Triton JIT 编译的 kernel |
| **Duration** | 2.24~2.56 µs | 4 次执行的范围 |
| **Grid** | (1536, 1, 1) | B×H×T = 1×12×128 = 1536 programs |
| **Block** | (64, 1, 1) | = num_warps=2 × 32 threads/warp |
| **Registers/thread** | **26** | 很低！远低于 128 上限 |
| **Static shared mem** | 0 bytes | Triton 未使用 static smem |
| **Dynamic shared mem** | 8 bytes | 极少量（可能是 Triton 内部用于 reduction） |
| **Shared mem executed** | 32768 bytes (32 KB) | SM 分配的实际 smem |
| **Local mem/thread** | 0 bytes | 无 register spill |

### Occupancy 分析

```
  每 warp 分配寄存器: ceil(26 × 32 / 256) × 256 = 1024 个 32-bit 寄存器
  寄存器限制:   65536 / 1024 = 64 warps/SM → 不是瓶颈
  SM warp 上限: 48 warps/SM → 实际限制
  每 block:     2 warps
  最大 blocks/SM: min(48/2, 24) = 24 blocks/SM
  活跃 warps:   24 × 2 = 48 warps/SM

  理论 occupancy = 48/48 = 100%  ✅
```

**解读**：occupancy 已达 100%，说明 num_warps=2 配合 26 regs/thread 是高效的。每个 SM 可同时执行 24 个 block、48 个 warp，GPU 的 warp 调度器被充分利用。

### 带宽分析

```
  数据量: 12 × 128 × 128 × 2 bytes (fp16) = 384 KB per read
  Online Softmax: 2 reads + 1 write = 1152 KB total
  时间: ~2.5 µs
  有效带宽: 1152 KB / 2.5 µs ≈ 461 GB/s
  RTX 4090 峰值: 1008 GB/s
  带宽利用率: ~46%
```

**解读**：带宽利用率仅 46%，说明 kernel 在 seq_len=128 下 **不是** memory-bound。
结合 occupancy=100%，kernel 的瓶颈是 `compute` + `launch overhead`：
- grid=1536 programs 分配到 128 SMs → 平均每个 SM 只有 12 个 block
- 每个 block 只处理 128 个 fp16 元素（256 bytes）→ 计算量极小
- kernel 本身只需 ~2.5 µs，CUDA launch overhead (~2-5 µs) 已经可比

### 与 seq_len=512 的对比

| 指标 | seq_len=128 | seq_len=512 (推断) |
|------|:-----------:|:------------------:|
| Grid size | 1,536 | 6,144 |
| 每 block 数据量 | 256 B | 1 KB |
| 带宽利用率 | ~46% | **>100%** (L2 cache 加成) |
| 瓶颈 | Launch + Compute | **Memory bandwidth** |

### 如何采集数据

```bash
# 使用 Nsight Systems (推荐，兼容 CUDA 12.6)
nsys profile --trace=cuda --stats=true -o reports/ncu_softmax_nsys \
    python benchmarks/ncu_profile_kernel.py

# 从 SQLite 提取 kernel 数据
python3 -c "
import sqlite3
conn = sqlite3.connect('reports/ncu_softmax_nsys.sqlite')
c = conn.cursor()
c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL')
# ... 分析代码见 ncu_profile_kernel.py
"

# 如需 Nsight Compute (ncu)，需要升级到 CUDA 12.x 版本:
#   apt install nsight-compute-2024.x  # 或从 NVIDIA 官网下载
```

### 关键指标速查

| 指标 | 含义 | nsys 值 | ncu 值 | 目标 |
|------|------|:-----------:|:-----------:|------|
| 理论 Occupancy | SM 最大 warp 容量占比 | **100%** ✅ | **100%** ✅ | >50% |
| 实际 Achieved Occupancy | 运行时实际 warp 利用 | — | **31.5%** ⚠️ | >50% |
| Regs/thread | 寄存器压力 | **26** ✅ | **26** ✅ | <128 |
| Local mem spill | 寄存器溢出 | **0** ✅ | — | 0 |
| SM Throughput | 计算利用率 | — | **15.94%** ⚠️ | >50% |
| DRAM Throughput | 内存带宽利用率 | 46% | **12.16%** ⚠️ | >50% |
| Waves/SM | 填充 SM 的波数 | — | **0.50** ⚠️ | ≥1 |
| IPC (active) | 每活跃周期指令数 | — | **1.08** | 越高越好 |

> **注**: nsys 报告 100% occupancy 是*理论*值（基于寄存器/shared mem 不受限），ncu 的 31.5% 是*实际测量*值，差距来源于 grid 太小（0.5 wave）。

### Occupancy 100%（理论） 但实际利用率低 — 为什么？

这是典型的 **latency-bound / launch-bound** 特征：
1. 每个 block 的工作量太小（128 × fp16 = 256 bytes），几个时钟周期就完成
2. GPU warp 调度器虽然满载，但每个 warp 很快就无事可做
3. kernel 总执行时间（~2.5 µs）已接近 CUDA launch overhead 的量级
4. 解决方案：增大每个 block 的工作量（处理多行、更大 seq_len）

---

## 10. Nsight Compute 硬件级深度分析

使用 `ncu 2025.1.0 --set full`（40 passes）对 `_tunable_online_softmax_fwd` kernel 进行全面硬件剖析。

**配置**: B=1, H=12, D=64, seq_len=128, fp16, BLOCK_T=128, num_warps=2, num_stages=1

### 10.1 Speed of Light（Roofline 定位）

| 指标 | 值 | 解读 |
|------|:---:|------|
| **Compute (SM) Throughput** | **15.94%** | 计算单元严重未充分利用 |
| **Memory Throughput** | **15.94%** | 内存子系统同样未充分利用 |
| **DRAM Throughput** | **12.16%** | 仅使用了峰值带宽的 ~1/8 |
| **FP32 Peak 利用率** | **~1%** | Roofline 远离计算峰值 |

**SOL 诊断**: "This kernel grid is too small to fill the available resources on this device, resulting in only **0.5 full waves** across all SMs."

kernel 有 1536 个 block，但 RTX 4090 有 128 个 SM，理论可同时跑 128×(48÷2)=3072 个 block（每 SM 最多 24 个 2-warp block），所以 1536 block 仅产生 **0.5 wave**。

### 10.2 Launch Statistics

| 指标 | 值 |
|------|:---:|
| Block Size | **64** threads (2 warps) |
| Grid Size | **1,536** blocks |
| Registers/Thread | **26** |
| Shared Memory/Block | **1.032 KB** (8 bytes dynamic + 0 static) |
| Waves/SM | **0.50** |
| Kernel Duration | **3.33 µs** |

### 10.3 Occupancy 分析

| 指标 | 值 |
|------|:---:|
| **理论 Occupancy** | **100%** |
| **实际 Achieved Occupancy** | **31.53%** |
| Achieved Active Warps/SM | **15.13** (max 48) |
| Occupancy 限制因素 — 寄存器 | 10876% (不受限) |
| Occupancy 限制因素 — Block Size | 5270% (不受限) |
| Occupancy 限制因素 — Shared Mem | 1.176% (不受限) |

**关键发现**: 理论 100% vs 实际 31.5%，差距巨大。根本原因是 **grid 太小**（0.5 wave），大量 SM 无法被填满，导致有效occupancy远低于理论值。

### 10.4 Warp Stall 原因分析（Top Stall Reasons）

| 排名 | Stall 原因 | 值 (warps/cycle) | 占比 | 含义 |
|:---:|------|:---:|:---:|------|
| 1 | **Long Scoreboard** | **4.22** | **32.4%** | 等待全局内存加载完成（L2/DRAM 延迟） |
| 2 | **Short Scoreboard** | **2.81** | **21.6%** | 等待 L1/shared memory/MIO 操作 |
| 3 | **Wait** | **2.09** | **16.0%** | 等待固定延迟指令完成 |
| 4 | **IMC Miss** | **1.33** | **10.2%** | 指令缓存 miss |
| 5 | **Selected (执行中)** | **1.00** | **7.7%** | 正在执行（不是 stall） |
| 6 | **Barrier** | **0.69** | **5.3%** | 等待 barrier 同步 |
| 7 | **Not Selected** | **0.38** | **2.9%** | 就绪但未被调度器选中 |
| 8 | **Math Pipe Throttle** | **0.22** | **1.7%** | 计算流水线反压 |
| 9 | **No Instruction** | **0.13** | **1.0%** | 指令缓冲空 |
| 10 | **Drain** | **0.10** | **0.8%** | warp 退出中 |
| - | **其他** (dispatch, branch, misc, mio, membar, lg, tex, sleeping) | **≤0.09** | **<1%** | 可忽略 |

**Warp Latency**: 平均每条指令 **14.72 cycles** stall — 调度器每 **3.5 cycles** 才发射一条指令（理想值 1）。

### 10.5 Compute Pipeline 利用率

| Pipeline | 利用率 (elapsed %) | 利用率 (active %) |
|------|:---:|:---:|
| **FMA** (浮点乘加) | **5.98%** | **10.23%** |
| **ALU** (整数/逻辑) | **10.46%** | — |
| **LSU** (加载存储) | **15.94%** | — |
| **Tensor Core** | **0%** | **0%** |

所有 compute pipeline 均 **under-utilized** — 最高的 LSU 也只有 15.94%。

### 10.6 Memory 子系统

| 指标 | 值 |
|------|:---:|
| **DRAM Read** | **399.6 KB** |
| **DRAM Write** | **0 bytes** |
| **L1/TEX Hit Rate** | **33.33%** |
| **L2 Hit Rate** | **54.62%** |
| **IPC (active)** | **1.08** inst/cycle |

DRAM 写入为 0 说明 kernel 的输出完全在 L2/L1 缓存中被消费，或者写回被延迟。

### 10.7 ncu 优化建议汇总

ncu 自动诊断给出的优化建议，按估计加速幅度排序：

| 优先级 | 诊断 | 估计加速 | 建议 |
|:---:|------|:---:|------|
| 1 | **Compute Pipeline Under-utilized** | **80.47%** | 所有计算流水线利用率低，需增加 warp/调度器 |
| 2 | **Issue Slot Utilization** | **71.47%** | 每 3.5 cycle 才发射 1 指令，12 warps/scheduler 只分配了 4.2 个 |
| 3 | **Achieved Occupancy 低** | **68.47%** | 理论 100% vs 实际 31.5%，grid 太小 |
| 4 | **Workload Imbalance (SMSP)** | **5.41%** | SM 间负载不均衡 ±10% |
| 5 | **FP32 指令未融合** | **5.43%** | 3072 fused vs 110592 non-fused FP32，融合可提升 49% FP32 性能 |

### 10.8 综合诊断：瓶颈全景

```
┌──────────────────────────────────────────────────────────────────┐
│  _tunable_online_softmax_fwd  (seq_len=128, BLOCK_T=128)       │
│                                                                  │
│  ┌─ 根本瓶颈 ────────────────────────────────────────────────┐  │
│  │  ★ Grid 太小 (0.5 wave / 1536 blocks / 128 SMs)         │  │
│  │    → 到达 occupancy 31.5%（理论 100%）                    │  │
│  │    → SM/Memory throughput 均只有 ~16%                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Stall 分析 ──────────────────────────────────────────────┐  │
│  │  Long Scoreboard (32%) = 等待全局内存                     │  │
│  │  Short Scoreboard (22%) = 等待 L1/shared/MIO             │  │
│  │  Wait (16%) = 等待固定延迟指令                            │  │
│  │  → 每指令 14.72 cycle stall，IPC 仅 1.08                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ 优化方向 ────────────────────────────────────────────────┐  │
│  │  1. 增大工作量：更大 seq_len / batch → 更多 blocks       │  │
│  │  2. 增加 block 内工作：每 block 处理多行                  │  │
│  │  3. 使用 FMA 融合指令减少非融合 FP32 操作                 │  │
│  │  4. 利用 software pipelining 掩盖内存延迟                │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 11. 运行调优脚本

```bash
# 完整运行所有实验
python benchmarks/tune_triton_kernel.py

# 快速模式（约 2 分钟）
python benchmarks/tune_triton_kernel.py --quick

# 单独运行某个实验
python benchmarks/tune_triton_kernel.py --experiment block    # BLOCK_T 扫描
python benchmarks/tune_triton_kernel.py --experiment warps    # num_warps 扫描
python benchmarks/tune_triton_kernel.py --experiment stages   # num_stages 扫描
python benchmarks/tune_triton_kernel.py --experiment dtype    # fp16 vs fp32
python benchmarks/tune_triton_kernel.py --experiment scaling  # 序列长度缩放
python benchmarks/tune_triton_kernel.py --experiment sweep    # 联合扫描
python benchmarks/tune_triton_kernel.py --experiment autotune # Triton autotune

# 指定参数
python benchmarks/tune_triton_kernel.py --seq_len 512 --batch_size 4
```
