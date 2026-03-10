// =============================================================
// MetaX (沐曦) C500 — RMS Norm kernel
// =============================================================
#include "rms_norm_metax.hpp"

#include <mc_runtime_api.h>
#include <maca_fp16.h>
#include <maca_bfloat16.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>

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
template<> __device__ inline float to_float<__maca_bfloat16>(__maca_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline __half from_float<__half>(float v) { return __float2half(v); }
template<> __device__ inline __maca_bfloat16 from_float<__maca_bfloat16>(float v) { return __float2bfloat16(v); }

template<typename T>
__global__ void rms_norm_kernel(
    T *Y, const T *X, const T *W,
    int64_t rows, int64_t cols, float eps,
    int64_t y_row_stride, int64_t y_col_stride,
    int64_t x_row_stride, int64_t x_col_stride,
    int64_t w_stride
) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = to_float(X[row * x_row_stride + j * x_col_stride]);
        local_sum += val * val;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)cols + eps);

    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float x_val = to_float(X[row * x_row_stride + j * x_col_stride]);
        float w_val = to_float(W[j * w_stride]);
        Y[row * y_row_stride + j * y_col_stride] = from_float<T>(x_val * w_val * inv_rms);
    }
}

namespace llaisys::ops::metax {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    auto dtype = weight->dtype();

    int64_t rows = in->shape()[0];
    int64_t cols = in->shape()[1];

    int threads = 1;
    while (threads < cols && threads < 1024) threads <<= 1;
    size_t smem_size = threads * sizeof(float);

    auto ys0 = out->strides()[0], ys1 = out->strides()[1];
    auto xs0 = in->strides()[0], xs1 = in->strides()[1];
    auto ws0 = weight->strides()[0];

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float><<<(int)rows, threads, smem_size>>>(
            (float*)out->data(), (const float*)in->data(), (const float*)weight->data(),
            rows, cols, eps, ys0, ys1, xs0, xs1, ws0);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<__half><<<(int)rows, threads, smem_size>>>(
            (__half*)out->data(), (const __half*)in->data(), (const __half*)weight->data(),
            rows, cols, eps, ys0, ys1, xs0, xs1, ws0);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<__maca_bfloat16><<<(int)rows, threads, smem_size>>>(
            (__maca_bfloat16*)out->data(), (const __maca_bfloat16*)in->data(), (const __maca_bfloat16*)weight->data(),
            rows, cols, eps, ys0, ys1, xs0, xs1, ws0);
        break;
    default:
        throw std::runtime_error("MetaX rms_norm: unsupported dtype");
    }
    GPU_CHECK(mcGetLastError());
}

} // namespace llaisys::ops::metax
