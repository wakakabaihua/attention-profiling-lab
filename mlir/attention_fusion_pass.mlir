// ============================================================================
// Attention Fusion Pass 设计文档（伪 MLIR Pass）
// ============================================================================
//
// Pass 名称: AttentionFusionPass
// 作用: 识别 attention 中 scale + mask + softmax 的子图模式，
//       替换为融合操作 custom.fused_scaled_masked_softmax
//
// ============================================================================

// ==========================================
// 1. 模式匹配规则（Pattern）
// ==========================================
//
// 识别以下子图结构:
//
//   %scaled = linalg.generic { arith.mulf %scores, %scale }          (scale)
//   %masked = linalg.generic { arith.select (j<=i), %scaled, -inf }  (causal mask)
//   %max    = linalg.generic { reduction: arith.maximumf }           (softmax step1)
//   %exp    = linalg.generic { math.exp (%masked - %max) }           (softmax step2)
//   %sum    = linalg.generic { reduction: arith.addf %exp }          (softmax step3)
//   %probs  = linalg.generic { arith.divf %exp, %sum }              (softmax step4)
//
// 匹配条件:
//   (a) %scores 的形状为 (B, H, T, T) —— 注意力分数矩阵
//   (b) scale 为标量常量
//   (c) mask 模式为上三角因果遮罩（通过 linalg.index 检测）
//   (d) softmax 归约维度为 dim=-1（最后一个维度）
//   (e) 6 个操作之间通过 SSA 值直接连接（无旁路使用）

// ==========================================
// 2. 替换规则（Rewrite）
// ==========================================
//
// 将上述 6 个操作替换为:
//
//   %probs = "custom.fused_scaled_masked_softmax"(%scores, %scale) {
//       softmax_dim = -1 : i32,
//       is_causal = true
//   } : (tensor<?x?x?x?xf16>, f16) -> tensor<?x?x?x?xf16>
//
// SSA 重连:
//   - 原本使用 %probs 的所有后续操作（如 PV matmul）自动指向新操作的结果
//   - 中间值 %scaled, %masked, %max, %exp, %sum 变为死代码，由 DCE pass 清理

// ==========================================
// 3. Lowering 路径
// ==========================================
//
// custom.fused_scaled_masked_softmax
//   │
//   ├── GPU Lowering（首选）
//   │   └── 生成 Triton kernel 调用
//   │       └── _fused_scale_mask_softmax_fwd (见 models/triton_attention.py)
//   │
//   ├── CUDA Lowering（备选）
//   │   └── 生成 CUDA C++ kernel
//   │       └── 使用 shared memory 的 online softmax
//   │
//   └── CPU Lowering（测试用）
//       └── 展开回标量循环

// ==========================================
// 4. Pass 伪代码（C++ 风格）
// ==========================================
//
// struct AttentionFusionPattern : public OpRewritePattern<linalg::GenericOp> {
//
//   LogicalResult matchAndRewrite(
//       linalg::GenericOp divOp,        // 匹配最后的 div（normalize）
//       PatternRewriter &rewriter
//   ) const override {
//
//     // ---- Step 1: 反向追溯子图 ----
//     // 从 divOp 开始，验证其输入是否来自 exp_sum 和 exp 操作
//     auto expOp = divOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
//     auto sumOp = divOp.getInputs()[1].getDefiningOp<linalg::GenericOp>();
//     if (!expOp || !sumOp) return failure();
//
//     // 验证 exp 操作的输入来自 masked - max
//     auto maskOp = ...;
//     auto scaleOp = ...;
//
//     // ---- Step 2: 验证模式 ----
//     // 检查 scale 是否为标量乘法
//     if (!isScalarMul(scaleOp)) return failure();
//     // 检查 mask 是否为因果遮罩
//     if (!isCausalMask(maskOp)) return failure();
//     // 检查归约维度
//     if (getSoftmaxDim(sumOp) != -1) return failure();
//
//     // ---- Step 3: 提取参数 ----
//     Value scores = scaleOp.getInputs()[0];    // 原始 scores tensor
//     Value scale  = getScalarConstant(scaleOp); // scale 值
//
//     // ---- Step 4: 创建融合操作 ----
//     auto fusedOp = rewriter.create<custom::FusedScaledMaskedSoftmaxOp>(
//         divOp.getLoc(),
//         divOp.getResultTypes(),
//         scores, scale,
//         /*softmax_dim=*/ -1,
//         /*is_causal=*/ true
//     );
//
//     // ---- Step 5: 替换并清理 ----
//     rewriter.replaceOp(divOp, fusedOp.getResults());
//     // 中间操作会被 DCE 自动删除
//
//     return success();
//   }
// };
//
// struct AttentionFusionPass
//     : public PassWrapper<AttentionFusionPass, OperationPass<ModuleOp>> {
//
//   void runOnOperation() override {
//     RewritePatternSet patterns(&getContext());
//     patterns.add<AttentionFusionPattern>(&getContext());
//     if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
//       signalPassFailure();
//   }
// };

// ==========================================
// 5. 验证计划
// ==========================================
//
// (a) 功能验证:
//     - 对比 unfused 和 fused 的输出，最大误差 < 1e-3 (fp16)
//     - 已通过 models/triton_attention.py 的 verify_correctness() 验证
//
// (b) 性能验证:
//     - baseline (unfused):  traces/baseline_trace.json
//     - triton (fused):      traces/triton_trace.json
//     - 对比指标: kernel 数量、总耗时、launch overhead
//
// (c) 编译正确性:
//     - 确保 pass 不改变非 attention 部分的 IR
//     - 确保 SSA 值的使用者正确重连
//     - 确保动态形状的兼容性

// ==========================================
// 6. 扩展方向
// ==========================================
//
// (a) 支持非因果遮罩（双向 attention）
//     → is_causal = false，去掉 causal mask 判断
//
// (b) 支持 dropout 融合
//     → 在 softmax 后、PV matmul 前加入 dropout
//     → custom.fused_scaled_masked_softmax_dropout
//
// (c) 支持 FlashAttention 风格的 tiling
//     → 将 QK^T 和 PV matmul 也纳入融合
//     → custom.flash_attention（完整融合）
//     → 需要更复杂的 tiling + online softmax
//
// (d) 支持 GQA（Grouped Query Attention）
//     → Q heads 数 ≠ KV heads 数
//     → 需要广播语义的处理
