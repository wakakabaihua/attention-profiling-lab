# TVM Backend 分析（一）：Lowering 过程与 Relax IR 生成

**日期**: 2026-05-29  
**对应代码**: `tvm_integration/relax_importer.py`, `compiler/backends/tvm_backend.py`  
**实验配置**: B=1, H=12, T=128, D=64, dtype=fp16, CUDA target

---

## 1. 整体 Lowering 路径

TVM backend 在 Mini AI Compiler Pipeline 中的 lowering 路径如下：

```
IRGraph (内部 IR)
    └─ FUSED_SCALE_MASK_SOFTMAX 节点
         │  attrs: {scale_factor, is_causal, softmax_dim}
         ▼
lower_to_relax(graph, input_shape=(1,12,128,128))
         │  tvm_integration/relax_importer.py
         ▼
tvm.IRModule (Relax 函数 `main`)
         │  relax.build(mod, target="cuda")
         ▼
tvm.relax.Executable (CUDA binary)
         │  relax.VirtualMachine(ex, dev)
         ▼
VM["main"](inp_tvm) → out_tvm
         │  via DLPack (zero-copy CUDA 内存共享)
         ▼
torch.from_dlpack(out_tvm) → torch.Tensor
```

与 Triton backend 的 lowering 路径对比：

| 阶段 | Triton Backend | TVM Backend |
|------|---------------|-------------|
| 属性提取 | `lower_to_triton_specs()` → `TritonKernelSpec` | `lower_to_relax()` → `tvm.IRModule` |
| 中间表示 | Python dataclass（scale/is_causal/dim） | Relax SSA 函数（typed IR） |
| 编译 | Triton JIT（运行时编译） | `relax.build()` → cubin |
| 调用 | Python function call | TVM VirtualMachine |
| 内存 | PyTorch CUDA Tensor 直接传入 | DLPack zero-copy（无拷贝） |

---

## 2. IRGraph → Relax IRModule：节点映射细节

`lower_to_relax()` 在 `tvm_integration/relax_importer.py` 中只处理一种融合节点：`FUSED_SCALE_MASK_SOFTMAX`。属性提取逻辑：

```python
fused_node = _find_fused_node(graph)   # 找 FUSED_SCALE_MASK_SOFTMAX
scale      = float(fused_node.attrs.get("scale_factor", 1.0))   # 0.125
is_causal  = bool(fused_node.attrs.get("is_causal", True))       # True
softmax_dim = int(fused_node.attrs.get("softmax_dim", -1))       # -1 → axis=3
```

生成的 Relax 函数结构（对应 `_build_relax_module`）：

```python
# 等价的 Relax IR 伪代码（TVMScript 风格）
@R.function
def main(scores: R.Tensor([1, 12, 128, 128], "float16")):
    with R.dataflow():
        # Step 1: Scale
        scale_const = R.const(0.125, "float16")
        scaled = R.multiply(scores, scale_const)          # aten.mul → relax.op.multiply

        # Step 2: Causal Mask（is_causal=True）
        ones    = R.ones([1, 12, 128, 128], "float16")
        tril_m  = R.tril(ones)                            # 下三角 = 1，上三角 = 0
        bool_m  = R.astype(tril_m, "bool")
        neg_inf = R.broadcast_to(R.const(-inf, "float16"), [1, 12, 128, 128])
        masked  = R.where(bool_m, scaled, neg_inf)        # aten.masked_fill → relax.op.where

        # Step 3: Softmax
        output  = R.nn.softmax(masked, axis=3)            # dim=-1 → axis=3
        R.output(output)
    return output
```

**关键设计决策**：
- `relax.op.tril` + `relax.op.where` 替代 PyTorch 的 `masked_fill`，是因为 Relax 没有直接的 `masked_fill_scalar` op，需要用 `tril`（生成遮罩矩阵）→ `where`（条件选择）组合实现等价语义。
- 所有 op 在同一 Relax 函数（`main`）内，这是 TVM 能做 operator fusion 的前提：`relax.build()` 编译时会把整个函数内的 op 视为可融合单元。

---

## 3. Relax IR 与内部 IR 的边界

```
┌─────────────────────┐     lower_to_relax()     ┌──────────────────────┐
│  内部 IRGraph        │ ─────────────────────▶  │  tvm.IRModule        │
│                     │                          │                      │
│  IRNode:            │   属性提取 + 构建 BB      │  Relax Function:     │
│  - SCALE (0.125)    │   BlockBuilder API        │  - relax.multiply    │
│  - MASK (is_causal) │                          │  - relax.tril        │
│  - SOFTMAX (dim=-1) │                          │  - relax.where       │
│   ↓ (after fusion)  │                          │  - relax.nn.softmax  │
│  FUSED_SMS          │                          │                      │
└─────────────────────┘                          └──────────────────────┘
         ▲                                                  │
  compiler/passes/                                 relax.build(target="cuda")
  fusion.py                                                 │
                                                   tvm.relax.Executable
                                                   (GPU binary)
```

内部 IR 不感知 TVM 的存在；TVM backend 在 `IRGraph` 的 `FUSED_SCALE_MASK_SOFTMAX` 节点处接管，通过 `lower_to_relax()` 完成从内部 IR 到 TVM 世界的跨越。这是两个系统的明确边界。

---

## 4. DLPack 零拷贝内存共享

TVM backend 的执行路径使用 DLPack 协议避免 GPU 内存拷贝：

```python
# PyTorch fp16 CUDA Tensor → TVM NDArray（零拷贝）
inp_tvm = tvmrt.from_dlpack(scores)      # 共享 CUDA 指针，无 memcpy

# 执行 TVM 编译好的 kernel
out_tvm = vm["main"](inp_tvm)

# TVM NDArray → PyTorch Tensor（零拷贝）
return torch.from_dlpack(out_tvm)
```

DLPack 标准定义了跨框架共享 GPU 内存的协议（`DLManagedTensor`），只传递指针 + shape + dtype + strides，不发生数据拷贝。这意味着 TVM kernel 直接读写 PyTorch 分配的 CUDA 内存。

**DLPack 的偶发同步开销**：  
在少数情况下，`from_dlpack` 需要在 CUDA stream 间做隐式同步（当 PyTorch 的 stream 与 TVM dev 的 stream 不一致时），这是 `compiler (tvm)` std 偏大的部分原因之一（详见 benchmark 数据异常说明）。

---

## 5. 编译缓存机制

`TVMBackend` 实现了基于 `(graph_id, input_shape)` 的编译缓存：

```python
cache_key = (id(graph), tuple(scores.shape))  # (1, 12, 128, 128)
if cache_key in self._cache:
    vm, dev = self._cache[cache_key]           # 复用已编译 VM
else:
    mod = lower_to_relax(graph, input_shape=shape)
    ex  = relax.build(mod, target="cuda")      # 首次编译（~100ms）
    vm  = relax.VirtualMachine(ex, dev)
    self._cache[cache_key] = (vm, dev)         # 缓存
```

**重要说明**：`relax.build()` 首次调用耗时约 100–300ms（CUDA JIT 编译），但缓存后每次推理开销仅为 VM dispatch（~0.01ms 量级）。benchmark 中的 `warmup=50`（含 `extra_warmup=40`）确保首次编译开销不污染计时结果。

---

## 6. 当前 Lowering 的局限性

| 局限 | 说明 | 影响 |
|------|------|------|
| 静态 shape | `lower_to_relax()` 需要具体 shape（无动态 shape 支持） | 每个不同 shape 需重编译 |
| 无 MetaSchedule | 使用 TVM 默认 schedule，未做自动调优 | 性能未达理论上界 |
| `tril` 生成遮罩 | 每次执行时动态生成 tril 矩阵（[1,12,128,128] 全 1 矩阵） | 可在编译期预计算 |
| fp16 精度 | Relax IR 使用 fp16，与 PyTorch baseline 一致，但与 MLIR（fp32）不同 | 数值比较需注意 dtype |

---

*下一份报告*: [tvm_backend_performance.md](tvm_backend_performance.md) — TVM kernel 性能分析与 Triton 对比
