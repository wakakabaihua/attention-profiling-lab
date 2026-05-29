// RUN: online_softmax
//
// 动态形状测试：不同维度的 softmax 应匹配 online softmax 模式
// 使用 2D 输入形状
//
// CHECK: custom.online_softmax
// CHECK-SAME: dim = -1
// CHECK-SAME: algorithm = "online_2pass"
// CHECK-NOT: torch.aten.div.Tensor
// CHECK-NOT: torch.aten.softmax.int

module {
  func.func @main(%arg0: !torch.vtensor<[8,64],f32>) -> !torch.vtensor<[8,64],f32> {
    %int-1 = torch.constant.int -1
    %none = torch.constant.none
    %0 = torch.aten.softmax.int %arg0, %int-1, %none : !torch.vtensor<[8,64],f32>, !torch.int, !torch.none -> !torch.vtensor<[8,64],f32>
    return %0 : !torch.vtensor<[8,64],f32>
  }
}
