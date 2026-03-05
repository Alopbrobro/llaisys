#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

/// Dequantize INT8 weight to FP32 output
/// @param out     [out_features, in_features] FP32 output
/// @param weight  [out_features, in_features] INT8 quantized weight
/// @param scale   [out_features]              FP32 per-channel scale
///
/// Formula: out[i][j] = weight[i][j] * scale[i]
void dequantize(tensor_t out, tensor_t weight, tensor_t scale);

} // namespace llaisys::ops
