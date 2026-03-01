#include "add_nvidia.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
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

// ---- FP32/FP16/BF16 conversion helpers ----
template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<__half>(__half v) { return __half2float(v); }
template<> __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline __half from_float<__half>(float v) { return __float2half(v); }
template<> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ---- Element-wise add kernel ----
template<typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = to_float(a[idx]);
        float vb = to_float(b[idx]);
        c[idx] = from_float<T>(va + vb);
    }
}

namespace llaisys::ops::nvidia {

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    int threads = 256;
    int blocks = ((int)numel + threads - 1) / threads;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b), numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(c),
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b), numel);
        break;
    default:
        throw std::runtime_error("NVIDIA add: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
