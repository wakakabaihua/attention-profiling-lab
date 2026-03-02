# FX Graph 分析报告

**配置**: hidden_size=768, num_heads=12, seq_len=128, batch_size=1

## 节点分类统计

| 类别 | 节点数 | 占比 |
| --- | ---: | ---: |
| Shape（形状变换） | 9 | 32.1% |
| MatMul（矩阵乘法） | 6 | 21.4% |
| Other（其他） | 4 | 14.3% |
| Elementwise（逐元素运算） | 3 | 10.7% |
| LayerNorm | 2 | 7.1% |
| Mask（遮罩） | 2 | 7.1% |
| Softmax | 1 | 3.6% |
| Activation（激活函数） | 1 | 3.6% |
| **总计** | **28** | |

## FX Graph 节点列表

| # | 节点名称 | 算子 | 类别 |
| ---: | --- | --- | --- |
| 9 | `layer_norm` | `layer_norm.default` | LayerNorm |
| 10 | `linear` | `linear.default` | MatMul（矩阵乘法） |
| 11 | `view` | `view.default` | Shape（形状变换） |
| 12 | `unbind` | `unbind.int` | Shape（形状变换） |
| 13 | `getitem` | `getitem` | Other（其他） |
| 14 | `getitem_1` | `getitem` | Other（其他） |
| 15 | `getitem_2` | `getitem` | Other（其他） |
| 16 | `transpose` | `transpose.int` | Shape（形状变换） |
| 17 | `transpose_1` | `transpose.int` | Shape（形状变换） |
| 18 | `transpose_2` | `transpose.int` | Shape（形状变换） |
| 19 | `transpose_3` | `transpose.int` | Shape（形状变换） |
| 20 | `matmul` | `matmul.default` | MatMul（矩阵乘法） |
| 21 | `mul` | `mul.Tensor` | Elementwise（逐元素运算） |
| 22 | `ones` | `ones.default` | Other（其他） |
| 23 | `triu` | `triu.default` | Mask（遮罩） |
| 24 | `masked_fill` | `masked_fill.Scalar` | Mask（遮罩） |
| 25 | `softmax` | `softmax.int` | Softmax |
| 26 | `matmul_1` | `matmul.default` | MatMul（矩阵乘法） |
| 27 | `transpose_4` | `transpose.int` | Shape（形状变换） |
| 28 | `contiguous` | `contiguous.default` | Shape（形状变换） |
| 29 | `view_1` | `view.default` | Shape（形状变换） |
| 30 | `linear_1` | `linear.default` | MatMul（矩阵乘法） |
| 31 | `add` | `add.Tensor` | Elementwise（逐元素运算） |
| 32 | `layer_norm_1` | `layer_norm.default` | LayerNorm |
| 33 | `linear_2` | `linear.default` | MatMul（矩阵乘法） |
| 34 | `gelu` | `gelu.default` | Activation（激活函数） |
| 35 | `linear_3` | `linear.default` | MatMul（矩阵乘法） |
| 36 | `add_1` | `add.Tensor` | Elementwise（逐元素运算） |

## 可融合子图模式

### 模式 1：Attention 子操作融合（scale + mask + softmax）

- **说明**：QK^T → scale → causal_mask → softmax 可融合为单个 fused kernel
- **涉及节点数**：6
- **MLIR 目标**：`custom.fused_scaled_masked_softmax`
- **匹配关键词**：matmul, mul, triu, masked_fill, softmax

### 模式 2：LayerNorm + 残差 Add 融合

- **说明**：LayerNorm 和后续的残差 add 可融合，减少一次全局读写
- **涉及节点数**：4
- **MLIR 目标**：`custom.fused_layer_norm_residual`
- **匹配关键词**：layer_norm, add

### 模式 3：MLP 激活融合（GeLU + bias）

- **说明**：线性层 + GeLU 激活可融合为单个 kernel
- **涉及节点数**：7
- **MLIR 目标**：`custom.fused_linear_gelu`
- **匹配关键词**：gelu, add, linear

### 模式 4：QKV 投影融合

- **说明**：QKV 投影 + reshape + split 可在一个 kernel 中完成
- **涉及节点数**：12
- **MLIR 目标**：`custom.fused_qkv_projection`
- **匹配关键词**：linear, view, unbind, transpose

## FX → MLIR 算子映射

| FX / ATen 算子 | MLIR 方言映射 |
| --- | --- |
| `aten.mm` | `linalg.matmul` |
| `aten.bmm` | `linalg.batch_matmul` |
| `aten.matmul` | `linalg.matmul / linalg.batch_matmul` |
| `aten.linear` | `linalg.matmul + arith.addf（带 bias）` |
| `aten.addmm` | `linalg.matmul + arith.addf` |
| `aten.add` | `arith.addf` |
| `aten.mul` | `arith.mulf` |
| `aten.div` | `arith.divf` |
| `aten.sub` | `arith.subf` |
| `aten.neg` | `arith.negf` |
| `aten.rsqrt` | `math.rsqrt` |
| `aten.exp` | `math.exp` |
| `aten.log` | `math.log` |
| `aten.tanh` | `math.tanh` |
| `aten.sqrt` | `math.sqrt` |
| `aten.gelu` | `linalg.generic { math.erf + arith.mulf }` |
| `aten.relu` | `arith.maximumf(x, 0)` |
| `aten.silu` | `arith.mulf(x, sigmoid(x))` |
| `aten.softmax` | `linalg.generic { math.exp, arith.divf }（归约）` |
| `aten._softmax` | `linalg.generic { math.exp, arith.divf }（归约）` |
| `aten.layer_norm` | `linalg.generic { mean + variance + normalize }` |
| `aten.native_layer_norm` | `linalg.generic { mean + variance + normalize }` |
| `aten.view` | `tensor.collapse_shape / tensor.expand_shape` |
| `aten.reshape` | `tensor.collapse_shape / tensor.expand_shape` |
| `aten.transpose` | `linalg.transpose` |
| `aten.permute` | `linalg.transpose（多维）` |
| `aten.contiguous` | `memref.copy（如需布局转换）` |
| `aten.unbind` | `tensor.extract_slice ×N` |
| `aten.cat` | `tensor.insert_slice` |
| `aten.slice` | `tensor.extract_slice` |
| `aten.select` | `tensor.extract_slice` |
| `aten.masked_fill` | `arith.select + arith.constant` |
| `aten.triu` | `linalg.generic { arith.cmpi + arith.select }` |
| `aten.where` | `arith.select` |
| `aten.clone` | `memref.copy` |
| `aten.copy_` | `memref.copy` |
| `aten.to` | `arith.truncf / arith.extf（dtype 转换）` |
| `aten.dropout` | `（推理时消除 → 恒等映射）` |
| `aten.scaled_dot_product_attention` | `custom.fused_attention（融合 kernel）` |

## 编译优化建议

基于 FX Graph 结构分析，提出以下 MLIR Pass 设计方向：

| Pass 名称 | 输入模式 | 输出 | 预期收益 |
| --- | --- | --- | --- |
| `custom.fused_scaled_masked_softmax` | QK^T → scale → causal_mask → softmax 可融合为单个 fused kernel | 单个 fused kernel | 减少 5 次 kernel launch |
| `custom.fused_layer_norm_residual` | LayerNorm 和后续的残差 add 可融合，减少一次全局读写 | 单个 fused kernel | 减少 3 次 kernel launch |
| `custom.fused_linear_gelu` | 线性层 + GeLU 激活可融合为单个 kernel | 单个 fused kernel | 减少 6 次 kernel launch |
| `custom.fused_qkv_projection` | QKV 投影 + reshape + split 可在一个 kernel 中完成 | 单个 fused kernel | 减少 11 次 kernel launch |
| `canonicalize<view>` | 9 个 view/reshape 节点 | 消除冗余形状变换 | 减少 memory copy |

---

*报告由 `benchmarks/export_fx_graph.py` 自动生成于 2026-03-01 15:25:55*
