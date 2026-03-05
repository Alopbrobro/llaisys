#pragma once
#include "../tensor/tensor.hpp"

namespace llaisys {
namespace ops {

// 0. Element-wise Add (C = A + B)
void add(tensor_t c, tensor_t a, tensor_t b);

// 1. Argmax
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);

// 2. Embedding
void embedding(tensor_t out, tensor_t index, tensor_t weight);

// 3. Linear (Y = XW^T + b)
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);

// 4. RMS Normalization
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);

// 5. RoPE (Rotary Positional Embeddings)
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);

// 6. Self Attention (GQA + Causal Mask)
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);

// 7. SwiGLU (Element-wise)
void swiglu(tensor_t out, tensor_t gate, tensor_t up);

// 8. Sample (Temperature + Top-K + Top-P)
void sample(tensor_t out_idx, tensor_t logits,
            float temperature, int top_k, float top_p, uint64_t seed);

// 9. Dequantize (INT8 → FP32, per-channel symmetric)
void dequantize(tensor_t out, tensor_t weight, tensor_t scale);

} // namespace ops
} // namespace llaisys