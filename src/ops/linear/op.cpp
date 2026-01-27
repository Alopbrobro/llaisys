#include "op.hpp"

namespace llaisys::ops {
template<typename T>
void linear_cpu_kernel(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) { // Y = XW^T + offset
    T* Y_ptr = reinterpret_cast<T*>(out -> data());
    const T* W_ptr = reinterpret_cast<const T*>(weight -> data());
    const T* X_ptr = reinterpret_cast<const T*>(in -> data());

    const T* offset_ptr = nullptr;
    if (bias && bias->data()) {
        offset_ptr = reinterpret_cast<const T*>(bias->data());
    }

    int64_t Y_row_dim = out -> shape()[0];
    int64_t Y_clow_dim = out -> shape()[1];

    int64_t X_row_dim = in -> shape()[0];
    int64_t X_clow_dim = in -> shape()[1];

    int64_t W_row_dim = weight -> shape()[0];
    int64_t W_clow_dim = weight -> shape()[1];

    if(X_clow_dim != W_clow_dim){
        throw std::runtime_error("X、W的维度不搭");
    }
    if(Y_row_dim != X_row_dim || Y_clow_dim != W_row_dim){
        throw std::runtime_error("Y和X、W的维度不搭");
    }

    ptrdiff_t Y_row_stride = out -> strides()[0];
    ptrdiff_t Y_clow_stride = out -> strides()[1];

    ptrdiff_t X_row_stride = in -> strides()[0];
    ptrdiff_t X_clow_stride = in -> strides()[1];

    ptrdiff_t W_row_stride = weight -> strides()[0];
    ptrdiff_t W_clow_stride = weight -> strides()[1];

    for(int i = 0; i < X_row_dim; i ++){
         for(int j = 0; j < W_row_dim; j ++){
            float sum = 0.0f;
            for(int l = 0; l < X_clow_dim; l++){
                int64_t X_idx = i * X_row_stride+ l * X_clow_stride;
                int64_t W_idx = j * W_row_stride+ l * W_clow_stride;
                sum += llaisys::utils::cast<float>(X_ptr[X_idx]) * llaisys::utils::cast<float>(W_ptr[W_idx]);
            }
            if (bias && bias->data()) {
                sum += llaisys::utils::cast<float>(offset_ptr[j]);
            }
            int64_t Y_idx = i * Y_row_stride + j * Y_clow_stride;
            Y_ptr[Y_idx] = llaisys::utils::cast<T>(sum);
         }
    }
}
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    auto dtype = weight->dtype();
    if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F32) {
        linear_cpu_kernel<float>(out, in, weight, bias);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F16) { 
        linear_cpu_kernel<llaisys::fp16_t>(out, in, weight, bias);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_BF16) { 
        linear_cpu_kernel<llaisys::bf16_t>(out, in, weight, bias);
    }
    else {
        throw std::runtime_error("数据类型不支持");
    }
}
} // namespace llaisys::ops
