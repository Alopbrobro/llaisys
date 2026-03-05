#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/dequantize_nvidia.cuh"
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

    dequantize_cpu(out, weight, scale);
}

} // namespace llaisys::ops
