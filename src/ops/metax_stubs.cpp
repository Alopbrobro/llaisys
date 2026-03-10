// =============================================================
// MetaX (沐曦) ops 桩实现 (Stubs)
// 当 MXMACA SDK 不可用时编译此文件，提供符号定义。
// 所有函数在运行时抛出异常。
// =============================================================
#include <stdexcept>
#include "../tensor/tensor.hpp"

namespace llaisys::ops::metax {

[[noreturn]] static void no_metax() {
    throw std::runtime_error("MetaX (MXMACA) runtime is not available on this platform");
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t n) { no_metax(); }
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) { no_metax(); }
void dequantize(tensor_t out, tensor_t weight, tensor_t scale) { no_metax(); }
void dequantize_int4(tensor_t out, tensor_t weight, tensor_t scale, int group_size) { no_metax(); }
void embedding(tensor_t out, tensor_t index, tensor_t weight) { no_metax(); }
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) { no_metax(); }
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) { no_metax(); }
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) { no_metax(); }
void sample(tensor_t out_idx, tensor_t logits, float temperature, int top_k, float top_p, uint64_t seed) { no_metax(); }
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) { no_metax(); }
void swiglu(tensor_t out, tensor_t gate, tensor_t up) { no_metax(); }

} // namespace llaisys::ops::metax
