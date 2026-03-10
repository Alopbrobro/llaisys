#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/rms_norm_metax.hpp"
#endif

namespace llaisys::ops {
template<typename T>
void rms_norm_cpu_kernel(tensor_t out, tensor_t in, tensor_t weight, float eps) {//Y:out    in:X    weight:W
    const T* W_ptr = reinterpret_cast<const T*>(weight -> data());
    const T* X_ptr = reinterpret_cast<const T*>(in -> data());
    T* Y_ptr = reinterpret_cast<T*>(out -> data());

    int64_t Y_row_dim = out -> shape()[0];
    int64_t Y_clow_dim = out -> shape()[1];

    int64_t X_row_dim = in -> shape()[0];
    int64_t X_clow_dim = in -> shape()[1];

    int64_t W_clow_dim = weight -> shape()[0];

    if(X_clow_dim != W_clow_dim){
        throw std::runtime_error("X、W的维度不搭");
    }
    if(Y_clow_dim != X_clow_dim || Y_row_dim != X_row_dim){
        throw std::runtime_error("Y和X的维度不搭");
    }
    ptrdiff_t Y_row_stride = out -> strides()[0];
    ptrdiff_t Y_clow_stride = out -> strides()[1];

    ptrdiff_t X_row_stride = in -> strides()[0];
    ptrdiff_t X_clow_stride = in -> strides()[1];

    ptrdiff_t W_clow_stride = weight -> strides()[0];
    
    for(int i = 0; i < X_row_dim; i++){
        float sum = 0.0f;
        for(int j = 0; j < X_clow_dim; j++){
            float X_val = llaisys::utils::cast<float>(X_ptr[i * X_row_stride + j * X_clow_stride]);
            sum += X_val * X_val;
        }
        float mean = sum / X_clow_dim;
        float rms = std::sqrt(mean + eps);
        float inv_rms = 1.0f / rms;
        for(int j = 0; j < X_clow_dim; j++){
            float X_val = llaisys::utils::cast<float>(X_ptr[i * X_row_stride + j * X_clow_stride]);
            float W_val = llaisys::utils::cast<float>(W_ptr[j * W_clow_stride]);
            Y_ptr[i * Y_row_stride + j * Y_clow_stride] = llaisys::utils::cast<T>( X_val * W_val * inv_rms);
        }
    }
}
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    auto dtype = weight->dtype();

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rms_norm(out, in, weight, eps);
    }
#endif

#ifdef ENABLE_METAX_API
    if (out->deviceType() == LLAISYS_DEVICE_METAX) {
        return metax::rms_norm(out, in, weight, eps);
    }
#endif

    if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F32) {
        rms_norm_cpu_kernel<float>(out, in, weight, eps);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_F16) { 
        rms_norm_cpu_kernel<llaisys::fp16_t>(out, in, weight, eps);
    } 
    else if (dtype == llaisysDataType_t::LLAISYS_DTYPE_BF16) { 
        rms_norm_cpu_kernel<llaisys::bf16_t>(out, in, weight, eps);
    }
    else {
        throw std::runtime_error("数据类型不支持");
    }
}
} // namespace llaisys::ops
