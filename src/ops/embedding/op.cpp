#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

#include "../../utils/types.hpp"

namespace llaisys::ops {

template<typename T>
void embedding_cpu_kernel(tensor_t out, tensor_t index, tensor_t weight) {
    const T* src_ptr = reinterpret_cast<const T*>(weight -> data());
    const int64_t* idx = reinterpret_cast<const int64_t*>(index -> data());
    T* out_ptr = reinterpret_cast<T*>(out -> data());
    
    int64_t row_dim = index -> shape()[0];
    int64_t clow_dim = weight -> shape()[1];

    int64_t w_row_stride = weight -> strides()[0];
    int64_t w_clow_stride = weight -> strides()[1];

    int64_t o_row_stride = out -> strides()[0];
    int64_t o_clow_stride = out -> strides()[1];

    int64_t i_clow_stride = index -> strides()[0];

    for(int i = 0;  i < row_dim; i++){
        for(int j = 0; j < clow_dim; j++){
            out_ptr[i * o_row_stride + j * o_clow_stride] = src_ptr[idx[i * i_clow_stride] * w_row_stride + j * w_clow_stride];
        }
    }
}

// Mixed precision: SrcT weight → float output (FP16/BF16 → FP32)
template<typename SrcT>
void embedding_mixed_cpu_kernel(tensor_t out, tensor_t index, tensor_t weight) {
    const SrcT* src_ptr = reinterpret_cast<const SrcT*>(weight -> data());
    const int64_t* idx = reinterpret_cast<const int64_t*>(index -> data());
    float* out_ptr = reinterpret_cast<float*>(out -> data());

    int64_t row_dim = index -> shape()[0];
    int64_t clow_dim = weight -> shape()[1];

    int64_t w_row_stride = weight -> strides()[0];
    int64_t w_clow_stride = weight -> strides()[1];
    int64_t o_row_stride = out -> strides()[0];
    int64_t o_clow_stride = out -> strides()[1];
    int64_t i_clow_stride = index -> strides()[0];

    for(int i = 0; i < row_dim; i++){
        for(int j = 0; j < clow_dim; j++){
            out_ptr[i * o_row_stride + j * o_clow_stride] =
                llaisys::utils::cast<float>(src_ptr[idx[i * i_clow_stride] * w_row_stride + j * w_clow_stride]);
        }
    }
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    auto dtype = weight->dtype();
    auto out_dtype = out->dtype();

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::embedding(out, index, weight);
    }
#endif

    // Mixed precision path: weight is F16/BF16 but output is F32
    if (out_dtype == llaisysDataType_t::LLAISYS_DTYPE_F32 && dtype != llaisysDataType_t::LLAISYS_DTYPE_F32) {
        if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F16) {
            embedding_mixed_cpu_kernel<llaisys::fp16_t>(out, index, weight);
        } else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_BF16) {
            embedding_mixed_cpu_kernel<llaisys::bf16_t>(out, index, weight);
        } else {
            throw std::runtime_error("Embedding: unsupported mixed precision dtype");
        }
        return;
    }

    if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F32) {
        embedding_cpu_kernel<float>(out, index, weight);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F16) { 
        embedding_cpu_kernel<uint16_t>(out, index, weight);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_BF16) { 
        embedding_cpu_kernel<uint16_t>(out, index, weight);
    }
    else {
        throw std::runtime_error("数据类型不支持");
    }
}
} // namespace llaisys::ops
