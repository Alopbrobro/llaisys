#include "op.hpp"

namespace llaisys::ops {
template <typename T>
void argmax_cpu_kernel(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    const T* src_ptr = reinterpret_cast<const T *>( vals -> data());
    int* idx_ptr = reinterpret_cast<int *>(max_idx -> data());
    T* val_ptr = reinterpret_cast<T *>(max_val -> data());

    // Support both 1D [N] and 2D [1, N] inputs.
    // Historically this op assumed 1D and used shape()[0], which breaks for logits shaped [1, vocab].
    size_t numel = 0;
    ptrdiff_t stride = 0;
    ptrdiff_t base = 0;

    if (vals->ndim() == 1) {
        numel = vals->shape()[0];
        stride = vals->strides()[0];
        base = 0;
    } else if (vals->ndim() == 2) {
        // Expect [1, N]; take argmax over the last dimension.
        if (vals->shape()[0] != 1) {
            throw std::runtime_error("argmax: 2D input must have shape [1, N]");
        }
        numel = vals->shape()[1];
        stride = vals->strides()[1];
        base = 0;
    } else {
        throw std::runtime_error("argmax: only supports 1D [N] or 2D [1, N]");
    }

    T maxVal = std::numeric_limits<T>::lowest();
    int maxValIdex = 0;

    for(size_t i = 0; i < numel; ++i){
        size_t offset = base + i * stride;
        T val = src_ptr[offset];
        if(val > maxVal){
            maxVal = val;
            maxValIdex = static_cast<int>(i);
        }
    }
    idx_ptr[0] = maxValIdex;
    val_ptr[0] = maxVal;
}
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    auto dtype = vals->dtype();
    if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F32) {
        argmax_cpu_kernel<float>(max_idx, max_val, vals);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F16) { 
        argmax_cpu_kernel<uint16_t>(max_idx, max_val, vals);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_BF16) { 
        argmax_cpu_kernel<uint16_t>(max_idx, max_val, vals);
    }
    else {
        throw std::runtime_error("数据类型不支持");
    }
}
} // namespace llaisys::ops
