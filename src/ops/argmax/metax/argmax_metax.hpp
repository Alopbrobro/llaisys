#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::metax {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
} // namespace llaisys::ops::metax
