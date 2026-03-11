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

/// Dequantize AWQ INT4 packed weight (int32, output-packed) to FP32 output
/// @param out       [out_features, in_features]   FP32 output (transposed for linear)
/// @param qweight   [in_features, out_features/8] I32 packed weight (8 x 4-bit per int32)
/// @param qzeros    [num_groups, out_features/8]  I32 packed zero-points
/// @param scales    [num_groups, out_features]     FP32 per-group scale
/// @param group_size  Number of elements per group along in_features
///
/// Formula: out[out_col][in_idx] = (unpack(qweight, in_idx, out_col) - unpack(qzeros, group, out_col)) * scale[group][out_col]
void dequantize_awq_int4(tensor_t out, tensor_t qweight, tensor_t qzeros, tensor_t scales, int group_size);

} // namespace llaisys::ops
