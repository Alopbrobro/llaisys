#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

namespace llaisys::ops {
template<typename T>
void rope_cpu_kernel(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {// out、in:[seqlen, nhead, d]张量是连续的, pos_ids: [seqlen,] dtype是int64
    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());

    size_t seq_len  = in->shape()[0];
    size_t n_head   = in->shape()[1];
    size_t head_dim = in->shape()[2];

    if(head_dim % 2 != 0){
        throw std::runtime_error("head_dim是奇数");
    }
    ptrdiff_t in_s0 = in->strides()[0]; // seqlen 维度的步长
    ptrdiff_t in_s1 = in->strides()[1]; // nhead 维度的步长
    ptrdiff_t in_s2 = in->strides()[2]; // head_dim 维度的步长
    ptrdiff_t out_s0 = out->strides()[0];
    ptrdiff_t out_s1 = out->strides()[1];
    ptrdiff_t out_s2 = out->strides()[2];
    ptrdiff_t pos_s0 = pos_ids->strides()[0];
    size_t head_dim_fre = head_dim / 2;
    for (size_t i = 0; i < seq_len; i++) {
        auto pos_offset = static_cast<ptrdiff_t>(i) * pos_s0;
        float pos = static_cast<float>(pos_ptr[pos_offset]);
        for (size_t j = 0; j < n_head; j++) {
            auto in_base = static_cast<ptrdiff_t>(i) * in_s0 + static_cast<ptrdiff_t>(j) * in_s1;
            auto out_base = static_cast<ptrdiff_t>(i) * out_s0 + static_cast<ptrdiff_t>(j) * out_s1;
            for (size_t l = 0; l < head_dim_fre; l++) {
                size_t idx_a = l;
                size_t idx_b = l + head_dim_fre;

                // Match torch reference: freqs = positions / (theta ** (2*i / head_dim))
                float exponent = 2.0f * static_cast<float>(l) / static_cast<float>(head_dim);
                float angle = pos / std::pow(theta, exponent);
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);

                float val_a = llaisys::utils::cast<float>(in_ptr[in_base + static_cast<ptrdiff_t>(idx_a) * in_s2]);
                float val_b = llaisys::utils::cast<float>(in_ptr[in_base + static_cast<ptrdiff_t>(idx_b) * in_s2]);

                // a' = a * cos - b * sin
                // b' = b * cos + a * sin
                float res_a = val_a * cos_val - val_b * sin_val;
                float res_b = val_b * cos_val + val_a * sin_val;

                out_ptr[out_base + static_cast<ptrdiff_t>(idx_a) * out_s2] = llaisys::utils::cast<T>(res_a);
                out_ptr[out_base + static_cast<ptrdiff_t>(idx_b) * out_s2] = llaisys::utils::cast<T>(res_b);
            }
        }
    }


}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    auto dtype = in->dtype();

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rope(out, in, pos_ids, theta);
    }
#endif

    if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F32) {
        rope_cpu_kernel<float>(out, in, pos_ids, theta);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F16) { 
        rope_cpu_kernel<llaisys::fp16_t>(out, in, pos_ids, theta);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_BF16) { 
        rope_cpu_kernel<llaisys::bf16_t>(out, in, pos_ids, theta);
    }
    else {
        throw std::runtime_error("数据类型不支持");
    }
}
} // namespace llaisys::ops
