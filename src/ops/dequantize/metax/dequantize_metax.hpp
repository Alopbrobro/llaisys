#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::metax {
void dequantize(tensor_t out, tensor_t weight, tensor_t scale);
void dequantize_int4(tensor_t out, tensor_t weight, tensor_t scale, int group_size);
} // namespace llaisys::ops::metax
