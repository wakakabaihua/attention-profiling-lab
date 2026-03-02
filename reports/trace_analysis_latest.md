# Trace 对比分析报告

> **生成时间**: 2026-03-02 11:26:37

> **Trace 目录**: `traces`

> **对比版本**: baseline, compiled, sdpa, triton_online, triton

---

## 1. 总体对比

| 指标 | baseline | compiled | sdpa | triton_online | triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kernel 总时间 (ms) | 1.12 | 1.01 | 0.96 | 1.00 | 1.00 |
| Kernel 启动总次数 | 360 | 280 | 220 | 280 | 280 |
| Kernel 平均时长 (μs) | 3.1 | 3.6 | 4.3 | 3.6 | 3.6 |
| 小 kernel (<50μs) | 360 | 280 | 220 | 280 | 280 |
| 中 kernel (50–500μs) | 0 | 0 | 0 | 0 | 0 |
| 大 kernel (≥500μs) | 0 | 0 | 0 | 0 | 0 |
| 内存拷贝事件数 | 20 | 0 | 0 | 0 | 0 |
| 内存拷贝时间 (ms) | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |

## 2. 加速比（相对于基线）

基线: **baseline**

| 版本 | 加速比 | Kernel 启动次数变化 |
| --- | ---: | ---: |
| compiled | 1.12x | +22% |
| sdpa | 1.18x | +39% |
| triton_online | 1.12x | +22% |
| triton | 1.12x | +22% |

## 3. Top Kernel — baseline

| Kernel | 调用次数 | 总时间 (μs) | 均值 (μs) |
| --- | ---: | ---: | ---: |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 40 | 310.1 | 7.8 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn` | 40 | 229.9 | 5.7 |
| `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::Half, ` | 40 | 113.4 | 2.8 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_t` | 20 | 99.0 | 5.0 |
| `void (anonymous namespace)::softmax_warp_forward<c10::Half, c10::Half, float, 7,` | 20 | 51.8 | 2.6 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_nn` | 20 | 50.0 | 2.5 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c1` | 40 | 45.3 | 1.1 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<a` | 20 | 41.4 | 2.1 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<a` | 20 | 39.1 | 2.0 |
| `void at::native::triu_tril_kernel<bool, int, true, 8, false>(at::cuda::detail::T` | 20 | 37.0 | 1.9 |

## 4. Top Kernel — compiled

| Kernel | 调用次数 | 总时间 (μs) | 均值 (μs) |
| --- | ---: | ---: | ---: |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 40 | 311.6 | 7.8 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn` | 40 | 223.6 | 5.6 |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(a` | 20 | 129.2 | 6.5 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_t` | 20 | 89.6 | 4.5 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_nn` | 20 | 45.3 | 2.3 |
| `triton_per_fused_native_layer_norm_0` | 20 | 37.5 | 1.9 |
| `triton_poi_fused__unsafe_view_gelu_4` | 20 | 34.3 | 1.7 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, __half, __half, float, false, __` | 20 | 32.0 | 1.6 |
| `triton_per_fused__unsafe_view_add_native_layer_norm_3` | 20 | 31.7 | 1.6 |
| `triton_per_fused__softmax_exp_masked_fill_mul_ones_prepare_softmax_online_sub_tr` | 20 | 29.6 | 1.5 |

## 5. Top Kernel — sdpa

| Kernel | 调用次数 | 总时间 (μs) | 均值 (μs) |
| --- | ---: | ---: | ---: |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 40 | 312.2 | 7.8 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn` | 20 | 186.9 | 9.3 |
| `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, fa` | 20 | 136.0 | 6.8 |
| `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::Half, ` | 40 | 112.7 | 2.8 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_t` | 20 | 98.9 | 4.9 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c1` | 40 | 45.2 | 1.1 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, __half, __half, float, false, __` | 20 | 33.5 | 1.7 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl` | 20 | 31.5 | 1.6 |

## 6. Top Kernel — triton_online

| Kernel | 调用次数 | 总时间 (μs) | 均值 (μs) |
| --- | ---: | ---: | ---: |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 40 | 311.2 | 7.8 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn` | 40 | 231.2 | 5.8 |
| `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::Half, ` | 40 | 112.7 | 2.8 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_t` | 20 | 99.5 | 5.0 |
| `_online_softmax_fwd` | 20 | 51.0 | 2.5 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_nn` | 20 | 49.2 | 2.5 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c1` | 40 | 44.9 | 1.1 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<a` | 20 | 39.6 | 2.0 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, __half, __half, float, false, __` | 20 | 33.1 | 1.7 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl` | 20 | 31.6 | 1.6 |

## 7. Top Kernel — triton

| Kernel | 调用次数 | 总时间 (μs) | 均值 (μs) |
| --- | ---: | ---: | ---: |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 40 | 310.8 | 7.8 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn` | 40 | 231.2 | 5.8 |
| `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::Half, ` | 40 | 112.5 | 2.8 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_t` | 20 | 99.8 | 5.0 |
| `_fused_scale_mask_softmax_fwd` | 20 | 51.7 | 2.6 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_nn` | 20 | 49.1 | 2.5 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c1` | 40 | 44.9 | 1.1 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<a` | 20 | 39.5 | 2.0 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, __half, __half, float, false, __` | 20 | 33.1 | 1.7 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl` | 20 | 31.6 | 1.6 |

## 8. 瓶颈分析与优化假设

### 观察到的瓶颈

- 基线共 **360** 次 kernel 启动
- 小 kernel (<50μs) 占比 **100%**（360 / 360）
- 存在 **20** 次 Host↔Device 内存拷贝事件（耗时 0.02 ms）
- **compiled** 减少了 **80** 次 kernel 启动（22% 减少）

### Kernel 功能分类（基于 baseline）

| 类别 | 启动次数 | 总时间 (μs) | 时间占比 |
| --- | ---: | ---: | ---: |
| MatMul（矩阵乘法） | 140 | 722.5 | 64.3% |
| Elementwise（逐元素运算） | 120 | 158.1 | 14.1% |
| LayerNorm | 40 | 113.4 | 10.1% |
| Mask（遮罩） | 40 | 78.4 | 7.0% |
| Softmax | 20 | 51.8 | 4.6% |

### 编译优化假设

> 以下"当前状态"均基于 **baseline** 的实际 profiling 数据。

| 优化方向 | 当前状态（baseline 实测） | 预期效果 |
| --- | --- | --- |
| Attention 子操作融合（scale + mask + softmax） | 共 60 次启动，耗时 130μs（占比 11.6%） | 合并为 1 个 fused kernel，消除 59 次 launch overhead |
| Elementwise 融合（LayerNorm + Add + GeLU） | 共 160 次启动，耗时 272μs（占比 24.2%） | 合并为 2–3 个 fused kernel，减少显存读写 |
| 消除不必要的内存拷贝 | 共 20 次 HtoD/DtoH，耗时 24μs | 通过 buffer 复用消除冗余拷贝 |

---

*报告由 `benchmarks/analyze_trace.py` 自动生成于 2026-03-02 11:26:37*
