#include "rope_nvidia.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "[CUDA ERROR] %s (code %d) at %s:%d\n",              \
                    cudaGetErrorString(err), (int)err, __FILE__, __LINE__);       \
            throw std::runtime_error(cudaGetErrorString(err));                    \
        }                                                                         \
    } while (0)

template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<__half>(__half v) { return __half2float(v); }
template<> __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline __half from_float<__half>(float v) { return __float2half(v); }
template<> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// RoPE: rotary position encoding
// Trig functions computed in float, I/O in T.
template<typename T>
__global__ void rope_kernel(
    T *out, const T *in, const int64_t *pos_ids,
    int64_t seq_len, int64_t n_head, int64_t head_dim, float theta,
    int64_t in_s0, int64_t in_s1, int64_t in_s2,
    int64_t out_s0, int64_t out_s1, int64_t out_s2,
    int64_t pos_s0
) {
    int64_t half_dim = head_dim / 2;
    int64_t total = seq_len * n_head * half_dim;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t l = tid % half_dim;
    int64_t h = (tid / half_dim) % n_head;
    int64_t s = tid / (half_dim * n_head);

    float pos = (float)pos_ids[s * pos_s0];
    float exponent = 2.0f * (float)l / (float)head_dim;
    float angle = pos / powf(theta, exponent);
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    int64_t idx_a = s * in_s0 + h * in_s1 + l * in_s2;
    int64_t idx_b = s * in_s0 + h * in_s1 + (l + half_dim) * in_s2;

    float val_a = to_float(in[idx_a]);
    float val_b = to_float(in[idx_b]);

    int64_t out_idx_a = s * out_s0 + h * out_s1 + l * out_s2;
    int64_t out_idx_b = s * out_s0 + h * out_s1 + (l + half_dim) * out_s2;

    out[out_idx_a] = from_float<T>(val_a * cos_val - val_b * sin_val);
    out[out_idx_b] = from_float<T>(val_b * cos_val + val_a * sin_val);
}

namespace llaisys::ops::nvidia {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    auto dtype = in->dtype();

    int64_t seq_len  = in->shape()[0];
    int64_t n_head   = in->shape()[1];
    int64_t head_dim = in->shape()[2];

    if (head_dim % 2 != 0) {
        throw std::runtime_error("rope: head_dim must be even");
    }

    int64_t total = seq_len * n_head * (head_dim / 2);
    int threads = 256;
    int blocks = ((int)total + threads - 1) / threads;

    auto is0 = in->strides()[0], is1 = in->strides()[1], is2 = in->strides()[2];
    auto os0 = out->strides()[0], os1 = out->strides()[1], os2 = out->strides()[2];
    auto ps0 = pos_ids->strides()[0];

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<blocks, threads>>>(
            (float*)out->data(), (const float*)in->data(), (const int64_t*)pos_ids->data(),
            seq_len, n_head, head_dim, theta, is0, is1, is2, os0, os1, os2, ps0);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<__half><<<blocks, threads>>>(
            (__half*)out->data(), (const __half*)in->data(), (const int64_t*)pos_ids->data(),
            seq_len, n_head, head_dim, theta, is0, is1, is2, os0, os1, os2, ps0);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<__nv_bfloat16><<<blocks, threads>>>(
            (__nv_bfloat16*)out->data(), (const __nv_bfloat16*)in->data(), (const int64_t*)pos_ids->data(),
            seq_len, n_head, head_dim, theta, is0, is1, is2, os0, os1, os2, ps0);
        break;
    default:
        throw std::runtime_error("NVIDIA rope: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
