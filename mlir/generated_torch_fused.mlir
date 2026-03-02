module {
  func.func @main(%arg0: !torch.vtensor<[1,12,128,128],f32>) -> !torch.vtensor<[1,12,128,128],f32> {
    // ===== AttentionFusionPass: 融合 scale + causal_mask + softmax =====
    // 原始: torch.aten.mul.Scalar → torch.aten.where.ScalarSelf → torch.aten.softmax.int
    // 消除 36 个操作 → 1 个融合操作
    %11 = "custom.fused_scaled_masked_softmax"(%arg0, 1.250000e-01) {
        softmax_dim = -1 : i64,
        is_causal = true,
        fusion_source = "attention_fusion_pass_v1"
    } : (!torch.vtensor<[1,12,128,128],f32>, f32) -> !torch.vtensor<[1,12,128,128],f32>
    return %11 : !torch.vtensor<[1,12,128,128],f32>
  }
}
