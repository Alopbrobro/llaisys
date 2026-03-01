#include "linear_nvidia.cuh"

#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <stdexcept>

#define CUBLAS_CHECK(call)                                                            \
    do {                                                                              \
        cublasStatus_t status = (call);                                               \
        if (status != CUBLAS_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "[cuBLAS ERROR] code %d at %s:%d\n",                      \
                    (int)status, __FILE__, __LINE__);                                 \
            throw std::runtime_error("cuBLAS call failed");                           \
        }                                                                             \
    } while (0)

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err = (call);                                                     \
        if (err != cudaSuccess) {                                                     \
            fprintf(stderr, "[CUDA ERROR] %s (code %d) at %s:%d\n",                  \
                    cudaGetErrorString(err), (int)err, __FILE__, __LINE__);           \
            throw std::runtime_error(cudaGetErrorString(err));                        \
        }                                                                             \
    } while (0)

// ---- Conversion helpers ----
template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<__half>(__half v) { return __half2float(v); }
template<> __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline __half from_float<__half>(float v) { return __float2half(v); }
template<> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ---- Bias add kernel (replaces the ones-vector GEMM approach) ----
template<typename T>
__global__ void add_bias_kernel(T *Y, const T *bias, int64_t M, int64_t N) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (tid >= total) return;
    int64_t j = tid % N;
    float y_val = to_float(Y[tid]);
    float b_val = to_float(bias[j]);
    Y[tid] = from_float<T>(y_val + b_val);
}

// Lazy-initialized thread-local cuBLAS handle
static cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        CUBLAS_CHECK(cublasCreate(&handle));
    }
    return handle;
}

namespace llaisys::ops::nvidia {

// Y = X * W^T + bias
// Uses cublasGemmEx to support F32/F16/BF16 with F32 compute.
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    auto dtype = weight->dtype();

    int64_t M = in->shape()[0];
    int64_t K = in->shape()[1];
    int64_t N = weight->shape()[0];

    cublasHandle_t handle = get_cublas_handle();

    float alpha = 1.0f;
    float beta  = 0.0f;

    cudaDataType_t cuda_dtype;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:  cuda_dtype = CUDA_R_32F;  break;
    case LLAISYS_DTYPE_F16:  cuda_dtype = CUDA_R_16F;  break;
    case LLAISYS_DTYPE_BF16: cuda_dtype = CUDA_R_16BF; break;
    default:
        throw std::runtime_error("NVIDIA linear: unsupported dtype");
    }

    // row-major Y = X * W^T  <=>  col-major Y^T = W * X^T
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_T,    // W stored row-major, viewed col-major => transpose
                              CUBLAS_OP_N,    // X stored row-major, X^T in col-major => no-transpose
                              (int)N, (int)M, (int)K,
                              &alpha,
                              weight->data(), cuda_dtype, (int)K,
                              in->data(),     cuda_dtype, (int)K,
                              &beta,
                              out->data(),    cuda_dtype, (int)N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));

    // Add bias if present
    if (bias && bias->data()) {
        int64_t total = M * N;
        int thr = 256, blk = ((int)total + thr - 1) / thr;
        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            add_bias_kernel<float><<<blk, thr>>>(
                (float*)out->data(), (const float*)bias->data(), M, N);
            break;
        case LLAISYS_DTYPE_F16:
            add_bias_kernel<__half><<<blk, thr>>>(
                (__half*)out->data(), (const __half*)bias->data(), M, N);
            break;
        case LLAISYS_DTYPE_BF16:
            add_bias_kernel<__nv_bfloat16><<<blk, thr>>>(
                (__nv_bfloat16*)out->data(), (const __nv_bfloat16*)bias->data(), M, N);
            break;
        default: break;
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace llaisys::ops::nvidia
