#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void dequantize(tensor_t out, tensor_t weight, tensor_t scale);
} // namespace llaisys::ops::nvidia
