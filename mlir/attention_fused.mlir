// ============================================================================
// Attention 计算的 MLIR 表示 —— 融合后（Fused）
// ============================================================================
//
// 本文件展示 attention fusion pass 的目标形态：
// 将 scale + causal_mask + softmax 的 5 个独立 linalg.generic 操作
// 融合为 1 个单一操作（custom.fused_scaled_masked_softmax）。
//
// 这对应 Triton 融合 kernel 在 MLIR 层面的语义表示。
//
// 融合前（attention_unfused.mlir）:
//   kernel 1: scale        → linalg.generic (elementwise mul)
//   kernel 2: causal_mask  → linalg.generic (select)
//   kernel 3: softmax_max  → linalg.generic (reduction max)
//   kernel 4: softmax_exp  → linalg.generic (exp + sum reduction)
//   kernel 5: softmax_norm → linalg.generic (div)
//   共 5 个 kernel launch, 4 次中间 tensor 全局内存读写
//
// 融合后（本文件）:
//   kernel 1: fused_scaled_masked_softmax → 1 个 kernel
//   共 1 个 kernel launch, 0 次中间 tensor 全局内存读写
//
// 收益分析（基于第一阶段 profiling 数据）:
//   - 消除 4 次 kernel launch overhead（每次 ~5μs on RTX 4090）
//   - 消除 4 次中间 tensor 的全局内存写+读（128×128×fp16 × 12 heads × 4 次 ≈ 48KB）
//   - 预计 attention 子操作耗时降低 50-70%
// ============================================================================

module @attention_fused {

  // ========== 融合后的单一操作 ==========
  //
  // 语义等价于:
  //   scaled  = scores * scale
  //   masked  = where(j <= i, scaled, -inf)
  //   max_val = max(masked, dim=-1)
  //   exp_val = exp(masked - max_val)
  //   sum_val = sum(exp_val, dim=-1)
  //   probs   = exp_val / sum_val
  //
  // 但在 single kernel 中完成，中间结果全部在寄存器 / shared memory 中。

  // 方案 A: 使用自定义方言操作（推荐用于实际编译器开发）
  func.func @attention_fused_custom_op(
      %scores: tensor<1x12x128x128xf16>,
      %scale: f16
  ) -> tensor<1x12x128x128xf16> {

    // 自定义融合操作 —— 编译器会将此 lower 到 GPU kernel
    // 属性指定 softmax 维度和是否使用因果遮罩
    %probs = "custom.fused_scaled_masked_softmax"(%scores, %scale) {
      softmax_dim = -1 : i32,
      is_causal = true,
      // 标注：此操作由 attention_fusion_pass 生成
      fusion_source = "attention_fusion_pass_v1"
    } : (tensor<1x12x128x128xf16>, f16) -> tensor<1x12x128x128xf16>

    return %probs : tensor<1x12x128x128xf16>
  }

  // 方案 B: 使用 linalg.generic 表示融合后的语义（用于验证）
  // 展示如何在不引入自定义方言的情况下，用标准 MLIR 表达融合逻辑
  func.func @attention_fused_linalg(
      %scores: tensor<1x12x128x128xf16>,
      %scale: f16
  ) -> tensor<1x12x128x128xf16> {

    // ---- 第一步: 融合 scale + mask + 求 max（online softmax 技术） ----
    // 使用 online softmax (Milakov & Gimelshein, 2018) 的思想，
    // 在单次遍历中同时完成 scale + mask + max + exp_sum

    %neg_inf = arith.constant -65504.0 : f16
    %zero = arith.constant 0.0 : f16

    // 初始化: (max_val, sum_exp) per row
    %init_max = linalg.fill ins(%neg_inf : f16)
                outs(tensor.empty() : tensor<1x12x128x1xf16>)
                -> tensor<1x12x128x1xf16>
    %init_sum = linalg.fill ins(%zero : f16)
                outs(tensor.empty() : tensor<1x12x128x1xf16>)
                -> tensor<1x12x128x1xf16>

    // online softmax pass 1: 融合 scale + mask + max + running sum_exp
    // 关键点: 在归约循环中同时维护 max 和 sum_exp
    // 当 max 更新时，重新缩放已有的 sum_exp
    //
    // 伪代码:
    //   for j in range(T):
    //       x = scores[i,j] * scale
    //       x = -inf if j > i else x     # causal mask
    //       if x > max_val:
    //           sum_exp = sum_exp * exp(old_max - new_max)  # 重新缩放
    //           max_val = x
    //       sum_exp += exp(x - max_val)
    //
    // 注意: 标准 linalg.generic 难以表达 online softmax 的双变量归约，
    // 这正是需要自定义操作或 Triton kernel 的原因。

    // ---- 第二步: 归一化 ----
    // probs[i,j] = exp(scores[i,j] * scale - max[i]) / sum_exp[i]
    // 同样需要 causal mask: j > i 的位置输出 0

    // 此处省略具体实现，因为标准 linalg.generic 表达 online softmax 较为复杂。
    // 实际编译器中推荐使用方案 A（自定义操作）+ 专门的 lowering pass。

    // 最终效果等价于方案 A
    %probs = "custom.fused_scaled_masked_softmax"(%scores, %scale) {
      softmax_dim = -1 : i32,
      is_causal = true,
      fusion_source = "attention_fusion_pass_v1"
    } : (tensor<1x12x128x128xf16>, f16) -> tensor<1x12x128x128xf16>

    return %probs : tensor<1x12x128x128xf16>
  }
}
