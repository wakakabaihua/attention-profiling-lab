// RUN: attention_fusion
//
// 基础融合测试：scale + causal_mask + softmax → fused_scaled_masked_softmax
//
// CHECK: custom.fused_scaled_masked_softmax
// CHECK-SAME: scale =
// CHECK-SAME: is_causal = true
// CHECK-SAME: algorithm = "online"
// CHECK-NOT: torch.aten.softmax.int

module {
  func.func @main(%arg0: !torch.vtensor<[1,2,16,16],f32>) -> !torch.vtensor<[1,2,16,16],f32> {
    %float1.250000e-01 = torch.constant.float 1.250000e-01
    %0 = torch.aten.mul.Scalar %arg0, %float1.250000e-01 : !torch.vtensor<[1,2,16,16],f32>, !torch.float -> !torch.vtensor<[1,2,16,16],f32>
    %int16 = torch.constant.int 16
    %int16_0 = torch.constant.int 16
    %1 = torch.prim.ListConstruct %int16, %int16_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %int11 = torch.constant.int 11
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %2 = torch.aten.ones %1, %int11, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[16,16],i1>
    %int16_1 = torch.constant.int 16
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %cpu_4 = torch.constant.device "cpu"
    %false_5 = torch.constant.bool false
    %3 = torch.aten.arange %int16_1, %none_2, %none_3, %cpu_4, %false_5 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[16],si64>
    %int-2 = torch.constant.int -2
    %4 = torch.aten.unsqueeze %3, %int-2 : !torch.vtensor<[16],si64>, !torch.int -> !torch.vtensor<[1,16],si64>
    %int16_6 = torch.constant.int 16
    %none_7 = torch.constant.none
    %none_8 = torch.constant.none
    %cpu_9 = torch.constant.device "cpu"
    %false_10 = torch.constant.bool false
    %5 = torch.aten.arange %int16_6, %none_7, %none_8, %cpu_9, %false_10 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[16],si64>
    %int-1 = torch.constant.int -1
    %6 = torch.aten.unsqueeze %5, %int-1 : !torch.vtensor<[16],si64>, !torch.int -> !torch.vtensor<[16,1],si64>
    %int1 = torch.constant.int 1
    %7 = torch.aten.sub.Tensor %4, %6, %int1 : !torch.vtensor<[1,16],si64>, !torch.vtensor<[16,1],si64>, !torch.int -> !torch.vtensor<[16,16],si64>
    %int1_11 = torch.constant.int 1
    %8 = torch.aten.ge.Scalar %7, %int1_11 : !torch.vtensor<[16,16],si64>, !torch.int -> !torch.vtensor<[16,16],i1>
    %9 = torch.aten.logical_and %8, %2 : !torch.vtensor<[16,16],i1>, !torch.vtensor<[16,16],i1> -> !torch.vtensor<[16,16],i1>
    %float-Inf = torch.constant.float 0xFFF0000000000000
    %10 = torch.aten.where.ScalarSelf %9, %float-Inf, %0 : !torch.vtensor<[16,16],i1>, !torch.float, !torch.vtensor<[1,2,16,16],f32> -> !torch.vtensor<[1,2,16,16],f32>
    %int-1_12 = torch.constant.int -1
    %none_13 = torch.constant.none
    %11 = torch.aten.softmax.int %10, %int-1_12, %none_13 : !torch.vtensor<[1,2,16,16],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,16,16],f32>
    return %11 : !torch.vtensor<[1,2,16,16],f32>
  }
}
