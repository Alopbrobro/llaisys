#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// Sample a token from logits using temperature, top-k, and top-p.
// out_idx: [1] INT32 tensor to store the sampled token index
// logits:  [1, vocab] or [vocab] F32 tensor of raw logits
void sample(tensor_t out_idx, tensor_t logits,
            float temperature, int top_k, float top_p, uint64_t seed);
} // namespace llaisys::ops
