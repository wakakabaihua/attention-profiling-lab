// RUN: attention_fusion
//
// 反例测试：只有 softmax 没有 scale + mask 前驱，不应匹配融合 pattern
//
// CHECK-NOT: custom.fused_scaled_masked_softmax
// CHECK: torch.aten.softmax.int

module {
  func.func @main(%arg0: !torch.vtensor<[2,4,16,16],f32>) -> !torch.vtensor<[2,4,16,16],f32> {
    %int-1 = torch.constant.int -1
    %none = torch.constant.none
    %0 = torch.aten.softmax.int %arg0, %int-1, %none : !torch.vtensor<[2,4,16,16],f32>, !torch.int, !torch.none -> !torch.vtensor<[2,4,16,16],f32>
    return %0 : !torch.vtensor<[2,4,16,16],f32>
  }
}
