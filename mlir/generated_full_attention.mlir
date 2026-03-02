module {
  func.func @main(%arg0: !torch.vtensor<[1,12,128,64],f32>, %arg1: !torch.vtensor<[1,12,128,64],f32>, %arg2: !torch.vtensor<[1,12,128,64],f32>) -> !torch.vtensor<[1,12,128,64],f32> {
    %int-2 = torch.constant.int -2
    %int-1 = torch.constant.int -1
    %0 = torch.aten.transpose.int %arg1, %int-2, %int-1 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,64,128],f32>
    %1 = torch.aten.matmul %arg0, %0 : !torch.vtensor<[1,12,128,64],f32>, !torch.vtensor<[1,12,64,128],f32> -> !torch.vtensor<[1,12,128,128],f32>
    %float1.250000e-01 = torch.constant.float 1.250000e-01
    %2 = torch.aten.mul.Scalar %1, %float1.250000e-01 : !torch.vtensor<[1,12,128,128],f32>, !torch.float -> !torch.vtensor<[1,12,128,128],f32>
    %int128 = torch.constant.int 128
    %int128_0 = torch.constant.int 128
    %3 = torch.prim.ListConstruct %int128, %int128_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %int11 = torch.constant.int 11
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %4 = torch.aten.ones %3, %int11, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128,128],i1>
    %int128_1 = torch.constant.int 128
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %cpu_4 = torch.constant.device "cpu"
    %false_5 = torch.constant.bool false
    %5 = torch.aten.arange %int128_1, %none_2, %none_3, %cpu_4, %false_5 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int-2_6 = torch.constant.int -2
    %6 = torch.aten.unsqueeze %5, %int-2_6 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[1,128],si64>
    %int128_7 = torch.constant.int 128
    %none_8 = torch.constant.none
    %none_9 = torch.constant.none
    %cpu_10 = torch.constant.device "cpu"
    %false_11 = torch.constant.bool false
    %7 = torch.aten.arange %int128_7, %none_8, %none_9, %cpu_10, %false_11 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int-1_12 = torch.constant.int -1
    %8 = torch.aten.unsqueeze %7, %int-1_12 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int1 = torch.constant.int 1
    %9 = torch.aten.sub.Tensor %6, %8, %int1 : !torch.vtensor<[1,128],si64>, !torch.vtensor<[128,1],si64>, !torch.int -> !torch.vtensor<[128,128],si64>
    %int1_13 = torch.constant.int 1
    %10 = torch.aten.ge.Scalar %9, %int1_13 : !torch.vtensor<[128,128],si64>, !torch.int -> !torch.vtensor<[128,128],i1>
    %11 = torch.aten.logical_and %10, %4 : !torch.vtensor<[128,128],i1>, !torch.vtensor<[128,128],i1> -> !torch.vtensor<[128,128],i1>
    %float-Inf = torch.constant.float 0xFFF0000000000000
    %12 = torch.aten.where.ScalarSelf %11, %float-Inf, %2 : !torch.vtensor<[128,128],i1>, !torch.float, !torch.vtensor<[1,12,128,128],f32> -> !torch.vtensor<[1,12,128,128],f32>
    %int-1_14 = torch.constant.int -1
    %none_15 = torch.constant.none
    %13 = torch.aten.softmax.int %12, %int-1_14, %none_15 : !torch.vtensor<[1,12,128,128],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,12,128,128],f32>
    %14 = torch.aten.matmul %13, %arg2 : !torch.vtensor<[1,12,128,128],f32>, !torch.vtensor<[1,12,128,64],f32> -> !torch.vtensor<[1,12,128,64],f32>
    return %14 : !torch.vtensor<[1,12,128,64],f32>
  }
}
