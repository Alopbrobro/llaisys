// =============================================================
// MetaX (沐曦) C500 — Linear kernel (Y = X * W^T + bias)
// 使用 mxBLAS (沐曦 BLAS 库) 进行矩阵乘法
//
// 注：当 MXMACA SDK 不可用时（当前开发环境），
// 使用 CUDA cuBLAS 头文件编译，到沐曦平台上替换为 mxBLAS。
// mxBLAS API 与 cuBLAS 保持高度兼容。
// =============================================================
#include "linear_metax.hpp"

#include "../../../utils.hpp"

// BLAS 库头文件
// 在沐曦平台上替换为: #include <mxblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <stdexcept>

#define BLAS_CHECK(call)                                                          \
    do {                                                                          \
        cublasStatus_t status = (call);                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[MetaX BLAS ERROR] code %d at %s:%d\n",             \
                    (int)status, __FILE__, __LINE__);                             \
            throw std::runtime_error("MetaX BLAS call failed");                   \
        }                                                                         \
    } while (0)

#define GPU_CHECK(call)                                                           \
    do {                                                                          \
        auto err = (call);                                                        \
        if (err != 0) {                                                           \
            fprintf(stderr, "[MetaX GPU ERROR] code %d at %s:%d\n",              \
                    (int)err, __FILE__, __LINE__);                                \
            throw std::runtime_error("MetaX GPU call failed");                    \
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

// ---- Bias add kernel ----
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

// ---- FP32→FP16 conversion kernel ----
__global__ void convert_f32_to_f16_kernel(__half *out, const float *in, int64_t n) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = __float2half(in[tid]);
}

// ---- FP16 bias add to FP32 output ----
__global__ void add_bias_f16_to_f32_kernel(float *Y, const __half *bias, int64_t M, int64_t N) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (tid >= total) return;
    int64_t j = tid % N;
    Y[tid] += __half2float(bias[j]);
}

// Lazy-initialized thread-local BLAS handle
// 注：在沐曦平台上 cublasHandle_t → mxblasHandle_t, cublasCreate → mxblasCreate
static cublasHandle_t get_blas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        BLAS_CHECK(cublasCreate(&handle));
    }
    return handle;
}

namespace llaisys::ops::metax {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    auto w_dtype = weight->dtype();
    auto in_dtype = in->dtype();
    auto out_dtype = out->dtype();

    int64_t M = in->shape()[0];
    int64_t K = in->shape()[1];
    int64_t N = weight->shape()[0];

    cublasHandle_t handle = get_blas_handle();

    float alpha = 1.0f;
    float beta  = 0.0f;

    // ---- Mixed precision path: FP16 weight + FP32 input → FP32 output ----
    if (w_dtype == LLAISYS_DTYPE_F16 && in_dtype == LLAISYS_DTYPE_F32 && out_dtype == LLAISYS_DTYPE_F32) {
        static thread_local __half *in_f16_buf = nullptr;
        static thread_local int64_t in_f16_cap = 0;

        int64_t in_elems = M * K;
        if (in_elems > in_f16_cap) {
            if (in_f16_buf) cudaFree(in_f16_buf);
            cudaMalloc(&in_f16_buf, in_elems * sizeof(__half));
            in_f16_cap = in_elems;
        }

        int thr = 256, blk = ((int)in_elems + thr - 1) / thr;
        convert_f32_to_f16_kernel<<<blk, thr>>>(in_f16_buf, (const float*)in->data(), in_elems);

        BLAS_CHECK(cublasGemmEx(handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                (int)N, (int)M, (int)K,
                                &alpha,
                                weight->data(), CUDA_R_16F, (int)K,
                                in_f16_buf,     CUDA_R_16F, (int)K,
                                &beta,
                                out->data(),    CUDA_R_32F, (int)N,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT));

        if (bias && bias->data()) {
            int64_t total = M * N;
            thr = 256; blk = ((int)total + thr - 1) / thr;
            if (bias->dtype() == LLAISYS_DTYPE_F16) {
                add_bias_f16_to_f32_kernel<<<blk, thr>>>(
                    (float*)out->data(), (const __half*)bias->data(), M, N);
            } else {
                add_bias_kernel<float><<<blk, thr>>>(
                    (float*)out->data(), (const float*)bias->data(), M, N);
            }
            GPU_CHECK(cudaGetLastError());
        }
        return;
    }

    // ---- Standard path ----
    cudaDataType_t cuda_dtype;
    switch (w_dtype) {
    case LLAISYS_DTYPE_F32:  cuda_dtype = CUDA_R_32F;  break;
    case LLAISYS_DTYPE_F16:  cuda_dtype = CUDA_R_16F;  break;
    case LLAISYS_DTYPE_BF16: cuda_dtype = CUDA_R_16BF; break;
    default:
        throw std::runtime_error("MetaX linear: unsupported dtype");
    }

    BLAS_CHECK(cublasGemmEx(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            (int)N, (int)M, (int)K,
                            &alpha,
                            weight->data(), cuda_dtype, (int)K,
                            in->data(),     cuda_dtype, (int)K,
                            &beta,
                            out->data(),    cuda_dtype, (int)N,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));

    if (bias && bias->data()) {
        int64_t total = M * N;
        int thr = 256, blk = ((int)total + thr - 1) / thr;
        switch (w_dtype) {
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
        GPU_CHECK(cudaGetLastError());
    }
}

} // namespace llaisys::ops::metax
