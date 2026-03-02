// ============================================================================
// Attention 计算的 MLIR 表示 —— 融合前（Unfused Baseline）
// ============================================================================
//
// 本文件展示手写 ManualAttention 中 attention 子操作（scale + mask + softmax）
// 在 MLIR 层面的表示。每个操作对应一个独立的 linalg.generic / math 操作，
// 会被分别 lower 为独立 CUDA kernel —— 这正是 profiling 中观察到的性能瓶颈。
//
// 对应 Python 代码（ManualAttention.forward 步骤 2-4）:
//   attn_scores = attn_scores * scale           # 步骤 2: scale
//   attn_scores = attn_scores.masked_fill(...)   # 步骤 3: causal mask
//   attn_probs = F.softmax(attn_scores, dim=-1)  # 步骤 4: softmax
//
// 输入: %scores : tensor<1x12x128x128xf16>     (QK^T 的结果)
// 输出: %probs  : tensor<1x12x128x128xf16>     (softmax 概率)
// ============================================================================

// --- 辅助函数类型 ---
// B=1, H=12, T=128

module @attention_unfused {

  // ========== 步骤 2: Scale（逐元素乘法） ==========
  // 对应 CUDA: 独立 elementwise kernel
  // profiling 观察: ~3μs, 启动开销 > 计算开销
  func.func @scale_scores(
      %scores: tensor<1x12x128x128xf16>,
      %scale: f16
  ) -> tensor<1x12x128x128xf16> {

    %init = tensor.empty() : tensor<1x12x128x128xf16>

    %scaled = linalg.generic {
      indexing_maps = [
        affine_map<(b, h, i, j) -> (b, h, i, j)>,   // 输入: scores
        affine_map<(b, h, i, j) -> (b, h, i, j)>     // 输出
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%scores : tensor<1x12x128x128xf16>)
      outs(%init : tensor<1x12x128x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      %result = arith.mulf %in, %scale : f16
      linalg.yield %result : f16
    } -> tensor<1x12x128x128xf16>

    return %scaled : tensor<1x12x128x128xf16>
  }

  // ========== 步骤 3: Causal Mask（条件赋值） ==========
  // 对应 CUDA: 独立 elementwise kernel
  // profiling 观察: ~4μs, 含 triu mask 生成 + masked_fill
  func.func @causal_mask(
      %scaled: tensor<1x12x128x128xf16>
  ) -> tensor<1x12x128x128xf16> {

    %init = tensor.empty() : tensor<1x12x128x128xf16>
    %neg_inf = arith.constant -65504.0 : f16   // fp16 近似 -inf

    %masked = linalg.generic {
      indexing_maps = [
        affine_map<(b, h, i, j) -> (b, h, i, j)>,   // 输入
        affine_map<(b, h, i, j) -> (b, h, i, j)>     // 输出
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%scaled : tensor<1x12x128x128xf16>)
      outs(%init : tensor<1x12x128x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      // 获取索引: i = row, j = col
      %i = linalg.index 2 : index
      %j = linalg.index 3 : index
      // 因果条件: j <= i（下三角）
      %cond = arith.cmpi sle, %j, %i : index
      %result = arith.select %cond, %in, %neg_inf : f16
      linalg.yield %result : f16
    } -> tensor<1x12x128x128xf16>

    return %masked : tensor<1x12x128x128xf16>
  }

  // ========== 步骤 4: Softmax（归约 + 逐元素） ==========
  // 对应 CUDA: 2-3 个 kernel（max归约 + exp+sum归约 + 归一化）
  // profiling 观察: ~5μs 总计, 是最大的融合机会
  //
  // softmax(x)_j = exp(x_j - max(x)) / sum(exp(x_i - max(x)))
  // 需要两次 pass:
  //   pass 1: 沿 dim=-1 求 max 和 sum(exp)
  //   pass 2: 归一化

  // --- pass 1a: 求行最大值 ---
  func.func @softmax_max(
      %input: tensor<1x12x128x128xf16>
  ) -> tensor<1x12x128x1xf16> {

    %neg_inf = arith.constant -65504.0 : f16
    %init = linalg.fill ins(%neg_inf : f16)
            outs(tensor.empty() : tensor<1x12x128x1xf16>)
            -> tensor<1x12x128x1xf16>

    %max_val = linalg.generic {
      indexing_maps = [
        affine_map<(b, h, i, j) -> (b, h, i, j)>,   // 输入
        affine_map<(b, h, i, j) -> (b, h, i, 0)>     // 输出（归约 j 维度）
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%input : tensor<1x12x128x128xf16>)
      outs(%init : tensor<1x12x128x1xf16>) {
    ^bb0(%in: f16, %acc: f16):
      %result = arith.maximumf %in, %acc : f16
      linalg.yield %result : f16
    } -> tensor<1x12x128x1xf16>

    return %max_val : tensor<1x12x128x1xf16>
  }

  // --- pass 1b: exp(x - max) 并求和 ---
  func.func @softmax_exp_sum(
      %input: tensor<1x12x128x128xf16>,
      %max_val: tensor<1x12x128x1xf16>
  ) -> (tensor<1x12x128x128xf16>, tensor<1x12x128x1xf16>) {

    // exp(x - max)
    %init_exp = tensor.empty() : tensor<1x12x128x128xf16>
    %exp_out = linalg.generic {
      indexing_maps = [
        affine_map<(b, h, i, j) -> (b, h, i, j)>,
        affine_map<(b, h, i, j) -> (b, h, i, 0)>,
        affine_map<(b, h, i, j) -> (b, h, i, j)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %max_val : tensor<1x12x128x128xf16>, tensor<1x12x128x1xf16>)
      outs(%init_exp : tensor<1x12x128x128xf16>) {
    ^bb0(%x: f16, %m: f16, %out: f16):
      %diff = arith.subf %x, %m : f16
      %result = math.exp %diff : f16
      linalg.yield %result : f16
    } -> tensor<1x12x128x128xf16>

    // sum(exp)
    %zero = arith.constant 0.0 : f16
    %init_sum = linalg.fill ins(%zero : f16)
                outs(tensor.empty() : tensor<1x12x128x1xf16>)
                -> tensor<1x12x128x1xf16>

    %sum_val = linalg.generic {
      indexing_maps = [
        affine_map<(b, h, i, j) -> (b, h, i, j)>,
        affine_map<(b, h, i, j) -> (b, h, i, 0)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%exp_out : tensor<1x12x128x128xf16>)
      outs(%init_sum : tensor<1x12x128x1xf16>) {
    ^bb0(%in: f16, %acc: f16):
      %result = arith.addf %in, %acc : f16
      linalg.yield %result : f16
    } -> tensor<1x12x128x1xf16>

    return %exp_out, %sum_val : tensor<1x12x128x128xf16>, tensor<1x12x128x1xf16>
  }

  // --- pass 2: 归一化 ---
  func.func @softmax_normalize(
      %exp_out: tensor<1x12x128x128xf16>,
      %sum_val: tensor<1x12x128x1xf16>
  ) -> tensor<1x12x128x128xf16> {

    %init = tensor.empty() : tensor<1x12x128x128xf16>
    %normalized = linalg.generic {
      indexing_maps = [
        affine_map<(b, h, i, j) -> (b, h, i, j)>,
        affine_map<(b, h, i, j) -> (b, h, i, 0)>,
        affine_map<(b, h, i, j) -> (b, h, i, j)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%exp_out, %sum_val : tensor<1x12x128x128xf16>, tensor<1x12x128x1xf16>)
      outs(%init : tensor<1x12x128x128xf16>) {
    ^bb0(%e: f16, %s: f16, %out: f16):
      %result = arith.divf %e, %s : f16
      linalg.yield %result : f16
    } -> tensor<1x12x128x128xf16>

    return %normalized : tensor<1x12x128x128xf16>
  }

  // ========== 完整 unfused 流程 ==========
  func.func @attention_scale_mask_softmax_unfused(
      %scores: tensor<1x12x128x128xf16>,
      %scale: f16
  ) -> tensor<1x12x128x128xf16> {

    // kernel 1: scale
    %scaled = func.call @scale_scores(%scores, %scale)
        : (tensor<1x12x128x128xf16>, f16) -> tensor<1x12x128x128xf16>

    // kernel 2: causal mask
    %masked = func.call @causal_mask(%scaled)
        : (tensor<1x12x128x128xf16>) -> tensor<1x12x128x128xf16>

    // kernel 3: softmax - max
    %max_val = func.call @softmax_max(%masked)
        : (tensor<1x12x128x128xf16>) -> tensor<1x12x128x1xf16>

    // kernel 4: softmax - exp + sum
    %exp_out, %sum_val = func.call @softmax_exp_sum(%masked, %max_val)
        : (tensor<1x12x128x128xf16>, tensor<1x12x128x1xf16>)
        -> (tensor<1x12x128x128xf16>, tensor<1x12x128x1xf16>)

    // kernel 5: softmax - normalize
    %probs = func.call @softmax_normalize(%exp_out, %sum_val)
        : (tensor<1x12x128x128xf16>, tensor<1x12x128x1xf16>)
        -> tensor<1x12x128x128xf16>

    return %probs : tensor<1x12x128x128xf16>
  }
}
