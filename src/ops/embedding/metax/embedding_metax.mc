// =============================================================
// MetaX (沐曦) C500 — Embedding kernel
// =============================================================
#include "embedding_metax.hpp"

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

template<typename T>
__global__ void embedding_kernel(
    T *out, const int64_t *index, const T *weight,
    int64_t num_rows, int64_t embed_dim,
    int64_t w_row_stride, int64_t w_col_stride,
    int64_t o_row_stride, int64_t o_col_stride,
    int64_t i_stride
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_rows * embed_dim;
    if (tid >= total) return;

    int64_t i = tid / embed_dim;
    int64_t j = tid % embed_dim;

    int64_t token_id = index[i * i_stride];
    out[i * o_row_stride + j * o_col_stride] = weight[token_id * w_row_stride + j * w_col_stride];
}

__global__ void embedding_f16_to_f32_kernel(
    float *out, const int64_t *index, const __half *weight,
    int64_t num_rows, int64_t embed_dim,
    int64_t w_row_stride, int64_t w_col_stride,
    int64_t o_row_stride, int64_t o_col_stride,
    int64_t i_stride
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_rows * embed_dim;
    if (tid >= total) return;

    int64_t i = tid / embed_dim;
    int64_t j = tid % embed_dim;

    int64_t token_id = index[i * i_stride];
    out[i * o_row_stride + j * o_col_stride] = __half2float(weight[token_id * w_row_stride + j * w_col_stride]);
}

__global__ void embedding_bf16_to_f32_kernel(
    float *out, const int64_t *index, const __maca_bfloat16 *weight,
    int64_t num_rows, int64_t embed_dim,
    int64_t w_row_stride, int64_t w_col_stride,
    int64_t o_row_stride, int64_t o_col_stride,
    int64_t i_stride
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_rows * embed_dim;
    if (tid >= total) return;

    int64_t i = tid / embed_dim;
    int64_t j = tid % embed_dim;

    int64_t token_id = index[i * i_stride];
    out[i * o_row_stride + j * o_col_stride] = __bfloat162float(weight[token_id * w_row_stride + j * w_col_stride]);
}

namespace llaisys::ops::metax {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    auto dtype = weight->dtype();

    int64_t num_rows  = index->shape()[0];
    int64_t embed_dim = weight->shape()[1];

    int64_t w_row_stride = weight->strides()[0];
    int64_t w_col_stride = weight->strides()[1];
    int64_t o_row_stride = out->strides()[0];
    int64_t o_col_stride = out->strides()[1];
    int64_t i_stride     = index->strides()[0];

    int64_t total = num_rows * embed_dim;
    int threads = 256;
    int blocks = ((int)total + threads - 1) / threads;

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float><<<blocks, threads>>>(
            (float *)out->data(), (const int64_t *)index->data(), (const float *)weight->data(),
            num_rows, embed_dim, w_row_stride, w_col_stride, o_row_stride, o_col_stride, i_stride);
        break;
    case LLAISYS_DTYPE_F16:
        if (out->dtype() == LLAISYS_DTYPE_F32) {
            embedding_f16_to_f32_kernel<<<blocks, threads>>>(
                (float *)out->data(), (const int64_t *)index->data(), (const __half *)weight->data(),
                num_rows, embed_dim, w_row_stride, w_col_stride, o_row_stride, o_col_stride, i_stride);
        } else {
            embedding_kernel<__half><<<blocks, threads>>>(
                (__half *)out->data(), (const int64_t *)index->data(), (const __half *)weight->data(),
                num_rows, embed_dim, w_row_stride, w_col_stride, o_row_stride, o_col_stride, i_stride);
        }
        break;
    case LLAISYS_DTYPE_BF16:
        if (out->dtype() == LLAISYS_DTYPE_F32) {
            embedding_bf16_to_f32_kernel<<<blocks, threads>>>(
                (float *)out->data(), (const int64_t *)index->data(), (const __maca_bfloat16 *)weight->data(),
                num_rows, embed_dim, w_row_stride, w_col_stride, o_row_stride, o_col_stride, i_stride);
        } else {
            embedding_kernel<__maca_bfloat16><<<blocks, threads>>>(
                (__maca_bfloat16 *)out->data(), (const int64_t *)index->data(), (const __maca_bfloat16 *)weight->data(),
                num_rows, embed_dim, w_row_stride, w_col_stride, o_row_stride, o_col_stride, i_stride);
        }
        break;
    default:
        throw std::runtime_error("MetaX embedding: unsupported dtype");
    }
    GPU_CHECK(mcGetLastError());
}

} // namespace llaisys::ops::metax
