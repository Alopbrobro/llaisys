#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/dequantize_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/dequantize_metax.hpp"
#endif

#include <stdexcept>

namespace llaisys::ops {

// CPU fallback: dequantize INT8 → FP32
static void dequantize_cpu(tensor_t out, tensor_t weight, tensor_t scale) {
    float*       out_ptr = reinterpret_cast<float*>(out->data());
    const int8_t* w_ptr  = reinterpret_cast<const int8_t*>(weight->data());
    const float*  s_ptr  = reinterpret_cast<const float*>(scale->data());

    int64_t rows = weight->shape()[0]; // out_features
    int64_t cols = weight->shape()[1]; // in_features

    for (int64_t i = 0; i < rows; ++i) {
        float si = s_ptr[i];
        for (int64_t j = 0; j < cols; ++j) {
            out_ptr[i * cols + j] = static_cast<float>(w_ptr[i * cols + j]) * si;
        }
    }
}

void dequantize(tensor_t out, tensor_t weight, tensor_t scale) {
    if (weight->dtype() != LLAISYS_DTYPE_I8) {
        throw std::runtime_error("dequantize: weight must be INT8");
    }
    if (out->dtype() != LLAISYS_DTYPE_F32) {
        throw std::runtime_error("dequantize: output must be FP32");
    }
    if (scale->dtype() != LLAISYS_DTYPE_F32) {
        throw std::runtime_error("dequantize: scale must be FP32");
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::dequantize(out, weight, scale);
    }
#endif

#ifdef ENABLE_METAX_API
    if (out->deviceType() == LLAISYS_DEVICE_METAX) {
        return metax::dequantize(out, weight, scale);
    }
#endif

    dequantize_cpu(out, weight, scale);
}

// ─────────────────────────────────────────────
// INT4 packed (uint8) → FP32, per-group symmetric
// ─────────────────────────────────────────────

static void dequantize_int4_cpu(tensor_t out, tensor_t weight, tensor_t scale, int group_size) {
    float*        out_ptr = reinterpret_cast<float*>(out->data());
    const uint8_t* w_ptr  = reinterpret_cast<const uint8_t*>(weight->data());
    const float*   s_ptr  = reinterpret_cast<const float*>(scale->data());

    int64_t rows        = out->shape()[0];
    int64_t cols        = out->shape()[1];          // 原始列数 = packed_cols * 2
    int64_t packed_cols = weight->shape()[1];        // cols / 2
    int64_t num_groups  = scale->shape()[1];         // cols / group_size

    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t pc = 0; pc < packed_cols; ++pc) {
            uint8_t byte = w_ptr[i * packed_cols + pc];
            int8_t val_even = (int8_t)(byte & 0x0F) - 8;   // 低 nibble
            int8_t val_odd  = (int8_t)(byte >> 4)   - 8;    // 高 nibble

            int64_t col_even = pc * 2;
            int64_t col_odd  = pc * 2 + 1;

            float s_even = s_ptr[i * num_groups + col_even / group_size];
            float s_odd  = s_ptr[i * num_groups + col_odd  / group_size];

            out_ptr[i * cols + col_even] = (float)val_even * s_even;
            out_ptr[i * cols + col_odd]  = (float)val_odd  * s_odd;
        }
    }
}

void dequantize_int4(tensor_t out, tensor_t weight, tensor_t scale, int group_size) {
    if (weight->dtype() != LLAISYS_DTYPE_U8) {
        throw std::runtime_error("dequantize_int4: weight must be U8 (packed INT4)");
    }
    if (out->dtype() != LLAISYS_DTYPE_F32) {
        throw std::runtime_error("dequantize_int4: output must be FP32");
    }
    if (scale->dtype() != LLAISYS_DTYPE_F32) {
        throw std::runtime_error("dequantize_int4: scale must be FP32");
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::dequantize_int4(out, weight, scale, group_size);
    }
#endif

#ifdef ENABLE_METAX_API
    if (out->deviceType() == LLAISYS_DEVICE_METAX) {
        return metax::dequantize_int4(out, weight, scale, group_size);
    }
#endif

    dequantize_int4_cpu(out, weight, scale, group_size);
}

} // namespace llaisys::ops
