// =============================================================
// MetaX (沐曦) C500 — Argmax kernel
// =============================================================
#include "argmax_metax.hpp"

#include <mc_runtime_api.h>
#include <maca_fp16.h>
#include <maca_bfloat16.h>
#include <cstdio>
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

struct ValIdx {
    float val;
    int idx;
};

template<typename T>
__global__ void argmax_kernel(
    int *out_idx, T *out_val, const T *vals,
    int64_t numel, int64_t stride
) {
    extern __shared__ ValIdx sdata_vi[];

    float local_max = -1e30f;
    int local_idx = 0;

    for (int64_t i = threadIdx.x; i < numel; i += blockDim.x) {
        float v = to_float(vals[i * stride]);
        if (v > local_max) {
            local_max = v;
            local_idx = (int)i;
        }
    }

    sdata_vi[threadIdx.x].val = local_max;
    sdata_vi[threadIdx.x].idx = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata_vi[threadIdx.x + s].val > sdata_vi[threadIdx.x].val) {
                sdata_vi[threadIdx.x] = sdata_vi[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_idx[0] = sdata_vi[0].idx;
        out_val[0] = from_float<T>(sdata_vi[0].val);
    }
}

namespace llaisys::ops::metax {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    auto dtype = vals->dtype();

    int64_t numel = 0;
    int64_t stride = 0;

    if (vals->ndim() == 1) {
        numel = vals->shape()[0];
        stride = vals->strides()[0];
    } else if (vals->ndim() == 2) {
        if (vals->shape()[0] != 1)
            throw std::runtime_error("argmax: 2D input must have shape [1, N]");
        numel = vals->shape()[1];
        stride = vals->strides()[1];
    } else {
        throw std::runtime_error("argmax: only supports 1D [N] or 2D [1, N]");
    }

    int threads = 256;
    if (threads > numel) {
        threads = 1;
        while (threads < numel) threads <<= 1;
        if (threads > 1024) threads = 1024;
    }
    size_t smem_size = threads * sizeof(ValIdx);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<float><<<1, threads, smem_size>>>(
            (int *)max_idx->data(), (float *)max_val->data(),
            (const float *)vals->data(), numel, stride);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<__half><<<1, threads, smem_size>>>(
            (int *)max_idx->data(), (__half *)max_val->data(),
            (const __half *)vals->data(), numel, stride);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<__maca_bfloat16><<<1, threads, smem_size>>>(
            (int *)max_idx->data(), (__maca_bfloat16 *)max_val->data(),
            (const __maca_bfloat16 *)vals->data(), numel, stride);
        break;
    default:
        throw std::runtime_error("MetaX argmax: unsupported dtype");
    }
    GPU_CHECK(mcGetLastError());
}

} // namespace llaisys::ops::metax
