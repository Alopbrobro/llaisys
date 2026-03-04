#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void sample(tensor_t out_idx, tensor_t logits,
            float temperature, int top_k, float top_p, uint64_t seed);
} // namespace llaisys::ops::nvidia
