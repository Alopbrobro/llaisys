#include "op.hpp"

namespace llaisys::ops {
template<typename T>
void swiglu_cpu_kernel(tensor_t out, tensor_t gate, tensor_t up) {
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* gate_ptr = reinterpret_cast<const T*>(gate->data());
    const T* up_ptr = reinterpret_cast<const T*>(up->data());

    int64_t seq_len = gate->shape()[0];
    int64_t dim     = gate->shape()[1];

    ptrdiff_t out_s0 = out->strides()[0];
    ptrdiff_t out_s1 = out->strides()[1];
    ptrdiff_t gate_s0 = gate->strides()[0];
    ptrdiff_t gate_s1 = gate->strides()[1];
    ptrdiff_t up_s0 = up->strides()[0];
    ptrdiff_t up_s1 = up->strides()[1];

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < dim; ++j) {
            
            int64_t gate_idx = i * gate_s0 + j * gate_s1;
            int64_t up_idx   = i * up_s0   + j * up_s1;
            int64_t out_idx  = i * out_s0  + j * out_s1;
            float val_gate = llaisys::utils::cast<float>(gate_ptr[gate_idx]);
            float val_up   = llaisys::utils::cast<float>(up_ptr[up_idx]);
            float silu_gate = val_gate / (1.0f + std::exp(-val_gate));
            float res = val_up * silu_gate;
            out_ptr[out_idx] = llaisys::utils::cast<T>(res);
        }
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    auto dtype = gate->dtype();
    if (dtype == LLAISYS_DTYPE_F32) {
        swiglu_cpu_kernel<float>(out, gate, up);
    } 
    else if (dtype == LLAISYS_DTYPE_F16) { 
        swiglu_cpu_kernel<llaisys::fp16_t>(out, gate, up);
    } 
    else if (dtype == LLAISYS_DTYPE_BF16) { 
        swiglu_cpu_kernel<llaisys::bf16_t>(out, gate, up);
    }
    else {
        throw std::runtime_error("SwiGLU: Unsupported dtype");
    }
}
} // namespace llaisys::ops
