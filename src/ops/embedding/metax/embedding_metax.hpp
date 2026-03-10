#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::metax {
void embedding(tensor_t out, tensor_t index, tensor_t weight);
} // namespace llaisys::ops::metax
