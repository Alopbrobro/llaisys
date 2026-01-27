#include "op.hpp"

namespace llaisys::ops {
template<typename T>
void rope_cpu_kernel(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {// out、in:[seqlen, nhead, d]张量是连续的, pos_ids: [seqlen,] dtype是int64
    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());

    int64_t seq_len  = in->shape()[0];
    int64_t n_head   = in->shape()[1];
    int64_t head_dim = in->shape()[2];

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
    int head_dim_fre = head_dim / 2;
    for(int i = 0; i < seq_len; i++){
        int64_t pos = pos_ptr[i * pos_s0];
        for(int j = 0; j < n_head; j++){
            int64_t in_base = i * in_s0 + j * in_s1;
            int64_t out_base = i * out_s0 + j * out_s1;
            for(int l = 0; l < head_dim_fre; l++){
                int64_t idx_a = l;
                int64_t idx_b = l + head_dim_fre;
                double freq = std::pow((double)theta, -2.0f * (double)l / (double)head_dim);
                double angle = (double)pos * freq;
                double cos_val = std::cos(angle);
                double sin_val = std::sin(angle);
                float val_a_f = llaisys::utils::cast<float>(in_ptr[in_base + idx_a * in_s2]);
                float val_b_f = llaisys::utils::cast<float>(in_ptr[in_base + idx_b * in_s2]);

                double val_a = static_cast<double>(val_a_f);
                double val_b = static_cast<double>(val_b_f);
                // a' = a * cos - b * sin
                // b' = b * cos + a * sin 
                double res_a = val_a * cos_val - val_b * sin_val;
                double res_b = val_b * cos_val + val_a * sin_val;

                out_ptr[out_base + idx_a * out_s2] = llaisys::utils::cast<T>(res_a);
                out_ptr[out_base + idx_b * out_s2] = llaisys::utils::cast<T>(res_b);
            }
        }
    }


}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
        auto dtype = in->dtype();
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
