#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> ()>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32> {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<0xFF800000> : tensor<f32>
    %cst_2 = arith.constant dense<true> : tensor<128x128xi1>
    %cst_3 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<1x12x128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x12x128x128xf32>) outs(%0 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %21 = arith.mulf %in, %cst_3 : f32
      linalg.yield %21 : f32
    } -> tensor<1x12x128x128xf32>
    %2 = tensor.empty() : tensor<128xi64>
    %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%2 : tensor<128xi64>) {
    ^bb0(%out: i64):
      %21 = linalg.index 0 : index
      %22 = arith.index_cast %21 : index to i64
      linalg.yield %22 : i64
    } -> tensor<128xi64>
    %expanded = tensor.expand_shape %3 [[0, 1]] output_shape [1, 128] : tensor<128xi64> into tensor<1x128xi64>
    %expanded_4 = tensor.expand_shape %3 [[0, 1]] output_shape [128, 1] : tensor<128xi64> into tensor<128x1xi64>
    %4 = tensor.empty() : tensor<128x128xi64>
    %5 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%expanded, %expanded_4 : tensor<1x128xi64>, tensor<128x1xi64>) outs(%4 : tensor<128x128xi64>) {
    ^bb0(%in: i64, %in_6: i64, %out: i64):
      %21 = arith.subi %in, %in_6 : i64
      linalg.yield %21 : i64
    } -> tensor<128x128xi64>
    %6 = tensor.empty() : tensor<128x128xi1>
    %7 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<128x128xi64>) outs(%6 : tensor<128x128xi1>) {
    ^bb0(%in: i64, %out: i1):
      %21 = arith.cmpi sge, %in, %c1_i64 : i64
      linalg.yield %21 : i1
    } -> tensor<128x128xi1>
    %8 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%7, %cst_2 : tensor<128x128xi1>, tensor<128x128xi1>) outs(%6 : tensor<128x128xi1>) {
    ^bb0(%in: i1, %in_6: i1, %out: i1):
      %21 = arith.andi %in, %in_6 : i1
      linalg.yield %21 : i1
    } -> tensor<128x128xi1>
    %9 = linalg.generic {indexing_maps = [#map5, #map6, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %cst_1, %1 : tensor<128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%0 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_6: f32, %in_7: f32, %out: f32):
      %21 = arith.select %in, %in_6, %in_7 : f32
      linalg.yield %21 : f32
    } -> tensor<1x12x128x128xf32>
    %10 = tensor.empty() : tensor<1x12x128xi64>
    %11 = linalg.fill ins(%c0_i64 : i64) outs(%10 : tensor<1x12x128xi64>) -> tensor<1x12x128xi64>
    %12 = tensor.empty() : tensor<1x12x128xf32>
    %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %14:2 = linalg.generic {indexing_maps = [#map, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9 : tensor<1x12x128x128xf32>) outs(%13, %11 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_6: i64):
      %21 = linalg.index 3 : index
      %22 = arith.index_cast %21 : index to i64
      %23 = arith.maximumf %in, %out : f32
      %24 = arith.cmpf ogt, %in, %out : f32
      %25 = arith.select %24, %22, %out_6 : i64
      linalg.yield %23, %25 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_5 = tensor.expand_shape %14#0 [[0], [1], [2, 3]] output_shape [1, 12, 128, 1] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map8, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %expanded_5 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%0 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %21 = arith.subf %in, %in_6 : f32
      linalg.yield %21 : f32
    } -> tensor<1x12x128x128xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<1x12x128x128xf32>) outs(%0 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %21 = math.exp %in : f32
      linalg.yield %21 : f32
    } -> tensor<1x12x128x128xf32>
    %17 = tensor.empty() : tensor<1x12x128x1xf32>
    %18 = linalg.fill ins(%cst_0 : f32) outs(%17 : tensor<1x12x128x1xf32>) -> tensor<1x12x128x1xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%16 : tensor<1x12x128x128xf32>) outs(%18 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %21 = arith.addf %in, %out : f32
      linalg.yield %21 : f32
    } -> tensor<1x12x128x1xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map8, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16, %19 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%0 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %21 = arith.divf %in, %in_6 : f32
      linalg.yield %21 : f32
    } -> tensor<1x12x128x128xf32>
    return %20 : tensor<1x12x128x128xf32>
  }
}
