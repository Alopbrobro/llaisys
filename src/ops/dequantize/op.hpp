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

/// Dequantize INT4 packed weight to FP32 output (per-group symmetric)
/// @param out     [rows, cols]               FP32 output
/// @param weight  [rows, cols/2]             uint8 packed weight (2 x INT4 per byte)
/// @param scale   [rows, num_groups]         FP32 per-group scale
/// @param group_size  Number of elements per group (cols / num_groups)
///
/// Packing: byte = ((val1+8) << 4) | ((val0+8) & 0x0F)
/// Unpack:  val_even = (byte & 0x0F) - 8,  val_odd = (byte >> 4) - 8
/// Formula: out[i][j] = unpack(weight, i, j) * scale[i][j / group_size]
void dequantize_int4(tensor_t out, tensor_t weight, tensor_t scale, int group_size);

} // namespace llaisys::ops
