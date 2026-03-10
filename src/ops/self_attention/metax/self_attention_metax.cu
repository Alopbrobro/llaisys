// =============================================================
// MetaX (沐曦) C500 — Self-Attention kernel
// =============================================================
#include "self_attention_metax.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <limits>

#define GPU_CHECK(call)                                                           \
    do {                                                                          \
        auto err = (call);                                                        \
        if (err != 0) {                                                           \
            fprintf(stderr, "[MetaX GPU ERROR] code %d at %s:%d\n",              \
                    (int)err, __FILE__, __LINE__);                                \
            throw std::runtime_error("MetaX GPU call failed");                    \
        }                                                                         \
    } while (0)

#define BLAS_CHECK(call)                                                          \
    do {                                                                          \
        auto status = (call);                                                     \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[MetaX BLAS ERROR] code %d at %s:%d\n",             \
                    (int)status, __FILE__, __LINE__);                             \
            throw std::runtime_error("MetaX BLAS call failed");                   \
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

// ----------------------------------------------------------------
// Gather ALL Q heads into contiguous float buffer
// src: [seq_len, n_head, head_dim] with strides, type T
// dst: [n_head, seq_len, head_dim] contiguous float
// ----------------------------------------------------------------
template<typename T>
__global__ void gather_all_q_kernel(
    float *dst, const T *src,
    int64_t seq_len, int64_t n_head, int64_t head_dim,
    int64_t s0, int64_t s1, int64_t s2
) {
    int64_t total = n_head * seq_len * head_dim;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t d = tid % head_dim;
    int64_t s = (tid / head_dim) % seq_len;
    int64_t h = tid / (head_dim * seq_len);

    dst[h * seq_len * head_dim + s * head_dim + d] =
        to_float(src[s * s0 + h * s1 + d * s2]);
}

// ----------------------------------------------------------------
// Gather KV heads and expand for GQA
// src: [total_len, n_kv_head, dim] with strides, type T
// dst: [n_head, total_len, dim] contiguous float
//      Each KV head is repeated group_size times.
// ----------------------------------------------------------------
template<typename T>
__global__ void gather_expand_kv_kernel(
    float *dst, const T *src,
    int64_t total_len, int64_t n_head, int64_t n_kv_head, int64_t dim,
    int64_t group_size,
    int64_t s0, int64_t s1, int64_t s2
) {
    int64_t total = n_head * total_len * dim;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t d = tid % dim;
    int64_t t = (tid / dim) % total_len;
    int64_t h = tid / (dim * total_len);
    int64_t kv_h = h / group_size;

    dst[h * total_len * dim + t * dim + d] =
        to_float(src[t * s0 + kv_h * s1 + d * s2]);
}

// ----------------------------------------------------------------
// Scatter ALL head outputs back
// src: [n_head, seq_len, v_dim] contiguous float
// dst: [seq_len, n_head, v_dim] with strides, type T
// ----------------------------------------------------------------
template<typename T>
__global__ void scatter_all_heads_kernel(
    T *dst, const float *src,
    int64_t seq_len, int64_t n_head, int64_t v_dim,
    int64_t s0, int64_t s1, int64_t s2
) {
    int64_t total = n_head * seq_len * v_dim;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t d = tid % v_dim;
    int64_t s = (tid / v_dim) % seq_len;
    int64_t h = tid / (v_dim * seq_len);

    dst[s * s0 + h * s1 + d * s2] =
        from_float<T>(src[h * seq_len * v_dim + s * v_dim + d]);
}

// ---- Scale kernel (float, operates on all heads) ----
__global__ void scale_kernel(float *data, float scale, int64_t n) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] *= scale;
}

// ---- Causal mask for batched scores ----
// scores: [n_head, seq_len, total_len] contiguous
// Each head has the same mask: j > (total_len - seq_len + i) => -inf
__global__ void causal_mask_batched_kernel(
    float *scores,
    int64_t n_head, int64_t seq_len, int64_t total_len
) {
    int64_t per_head = seq_len * total_len;
    int64_t total = n_head * per_head;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t pos_in_head = tid % per_head;
    int64_t i = pos_in_head / total_len;
    int64_t j = pos_in_head % total_len;
    int64_t current_pos = total_len - seq_len + i;

    if (j > current_pos) scores[tid] = -1e30f;
}

// ---- Per-row softmax (float) ----
// data: [num_rows, row_len] — num_rows = n_head * seq_len
__global__ void softmax_row_kernel(float *data, int64_t num_rows, int64_t row_len) {
    int64_t row = blockIdx.x;
    if (row >= num_rows) return;

    extern __shared__ float sdata[];
    float *row_data = data + row * row_len;

    float local_max = -1e30f;
    for (int64_t j = threadIdx.x; j < row_len; j += blockDim.x) {
        float val = row_data[j];
        if (val > local_max) local_max = val;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sdata[threadIdx.x + s] > sdata[threadIdx.x])
            sdata[threadIdx.x] = sdata[threadIdx.x + s];
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int64_t j = threadIdx.x; j < row_len; j += blockDim.x) {
        float val = expf(row_data[j] - max_val);
        row_data[j] = val;
        local_sum += val;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / (sdata[0] + 1e-6f);
    __syncthreads();

    for (int64_t j = threadIdx.x; j < row_len; j += blockDim.x) {
        row_data[j] *= inv_sum;
    }
}

// ----------------------------------------------------------------
// Lazy-initialized thread-local cuBLAS / mxBLAS handle
// ----------------------------------------------------------------
static cublasHandle_t get_blas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        BLAS_CHECK(cublasCreate(&handle));
    }
    return handle;
}

// ----------------------------------------------------------------
// Cached GPU temp buffers — grow-only, never freed during inference.
// ----------------------------------------------------------------
static float *s_q_all = nullptr;
static float *s_k_all = nullptr;
static float *s_v_all = nullptr;
static float *s_scores = nullptr;
static float *s_o_all = nullptr;
static size_t s_q_all_sz = 0;
static size_t s_k_all_sz = 0;
static size_t s_v_all_sz = 0;
static size_t s_scores_sz = 0;
static size_t s_o_all_sz = 0;

static void ensure_buf(float *&ptr, size_t &cur, size_t need) {
    if (need <= cur) return;
    if (ptr) cudaFree(ptr);
    GPU_CHECK(cudaMalloc(&ptr, need));
    cur = need;
}

// ----------------------------------------------------------------
// Batched self-attention: all heads computed in parallel via
// cublasSgemmStridedBatched (mxBLAS on MetaX).
// ----------------------------------------------------------------
namespace llaisys::ops::metax {

template<typename T>
static void self_attention_impl(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    int64_t seq_len   = q->shape()[0];
    int64_t n_head    = q->shape()[1];
    int64_t head_dim  = q->shape()[2];
    int64_t total_len = k->shape()[0];
    int64_t n_kv_head = k->shape()[1];
    int64_t v_dim     = v->shape()[2];
    int64_t group_size = n_head / n_kv_head;

    const T *q_ptr = reinterpret_cast<const T *>(q->data());
    const T *k_ptr = reinterpret_cast<const T *>(k->data());
    const T *v_ptr = reinterpret_cast<const T *>(v->data());
    T *o_ptr = reinterpret_cast<T *>(attn_val->data());

    ptrdiff_t q_s0 = q->strides()[0], q_s1 = q->strides()[1], q_s2 = q->strides()[2];
    ptrdiff_t k_s0 = k->strides()[0], k_s1 = k->strides()[1], k_s2 = k->strides()[2];
    ptrdiff_t v_s0 = v->strides()[0], v_s1 = v->strides()[1], v_s2 = v->strides()[2];
    ptrdiff_t o_s0 = attn_val->strides()[0], o_s1 = attn_val->strides()[1], o_s2 = attn_val->strides()[2];

    cublasHandle_t handle = get_blas_handle();

    // Ensure cached temp buffers
    size_t q_need = (size_t)(n_head * seq_len * head_dim) * sizeof(float);
    size_t k_need = (size_t)(n_head * total_len * head_dim) * sizeof(float);
    size_t v_need = (size_t)(n_head * total_len * v_dim) * sizeof(float);
    size_t s_need = (size_t)(n_head * seq_len * total_len) * sizeof(float);
    size_t o_need = (size_t)(n_head * seq_len * v_dim) * sizeof(float);

    ensure_buf(s_q_all, s_q_all_sz, q_need);
    ensure_buf(s_k_all, s_k_all_sz, k_need);
    ensure_buf(s_v_all, s_v_all_sz, v_need);
    ensure_buf(s_scores, s_scores_sz, s_need);
    ensure_buf(s_o_all, s_o_all_sz, o_need);

    int thr = 256;

    // 1. Gather ALL Q heads: T -> float [n_head, seq_len, head_dim]
    {
        int64_t n = n_head * seq_len * head_dim;
        int blk = ((int)n + thr - 1) / thr;
        gather_all_q_kernel<T><<<blk, thr>>>(
            s_q_all, q_ptr, seq_len, n_head, head_dim, q_s0, q_s1, q_s2);
        GPU_CHECK(cudaGetLastError());
    }

    // 2. Gather + expand KV: T -> float [n_head, total_len, dim]
    {
        int64_t n = n_head * total_len * head_dim;
        int blk = ((int)n + thr - 1) / thr;
        gather_expand_kv_kernel<T><<<blk, thr>>>(
            s_k_all, k_ptr, total_len, n_head, n_kv_head, head_dim, group_size,
            k_s0, k_s1, k_s2);
        GPU_CHECK(cudaGetLastError());
    }
    {
        int64_t n = n_head * total_len * v_dim;
        int blk = ((int)n + thr - 1) / thr;
        gather_expand_kv_kernel<T><<<blk, thr>>>(
            s_v_all, v_ptr, total_len, n_head, n_kv_head, v_dim, group_size,
            v_s0, v_s1, v_s2);
        GPU_CHECK(cudaGetLastError());
    }

    // 3. Batched GEMM: scores[h] = Q[h] * K[h]^T for all heads
    {
        float alpha = 1.0f, beta = 0.0f;
        long long strideA = (long long)(total_len * head_dim);
        long long strideB = (long long)(seq_len * head_dim);
        long long strideC = (long long)(seq_len * total_len);
        BLAS_CHECK(cublasSgemmStridedBatched(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)total_len, (int)seq_len, (int)head_dim,
            &alpha,
            s_k_all, (int)head_dim, strideA,
            s_q_all, (int)head_dim, strideB,
            &beta,
            s_scores, (int)total_len, strideC,
            (int)n_head));
    }

    // 4. Scale + causal mask
    {
        int64_t ns = n_head * seq_len * total_len;
        int blk = ((int)ns + thr - 1) / thr;
        scale_kernel<<<blk, thr>>>(s_scores, scale, ns);
        GPU_CHECK(cudaGetLastError());
        causal_mask_batched_kernel<<<blk, thr>>>(s_scores, n_head, seq_len, total_len);
        GPU_CHECK(cudaGetLastError());
    }

    // 5. Softmax per row (n_head * seq_len rows)
    {
        int64_t num_rows = n_head * seq_len;
        int sthr = 1;
        while (sthr < total_len && sthr < 1024) sthr <<= 1;
        size_t smem = sthr * sizeof(float);
        softmax_row_kernel<<<(int)num_rows, sthr, smem>>>(s_scores, num_rows, total_len);
        GPU_CHECK(cudaGetLastError());
    }

    // 6. Batched GEMM: O[h] = probs[h] * V[h] for all heads
    {
        float alpha = 1.0f, beta = 0.0f;
        long long strideA = (long long)(total_len * v_dim);
        long long strideB = (long long)(seq_len * total_len);
        long long strideC = (long long)(seq_len * v_dim);
        BLAS_CHECK(cublasSgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)v_dim, (int)seq_len, (int)total_len,
            &alpha,
            s_v_all, (int)v_dim, strideA,
            s_scores, (int)total_len, strideB,
            &beta,
            s_o_all, (int)v_dim, strideC,
            (int)n_head));
    }

    // 7. Scatter ALL heads: float -> T [seq_len, n_head, v_dim]
    {
        int64_t n = n_head * seq_len * v_dim;
        int blk = ((int)n + thr - 1) / thr;
        scatter_all_heads_kernel<T><<<blk, thr>>>(
            o_ptr, s_o_all, seq_len, n_head, v_dim, o_s0, o_s1, o_s2);
        GPU_CHECK(cudaGetLastError());
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    auto dtype = q->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_impl<float>(attn_val, q, k, v, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_impl<__half>(attn_val, q, k, v, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_impl<__nv_bfloat16>(attn_val, q, k, v, scale);
        break;
    default:
        throw std::runtime_error("MetaX self_attention: unsupported dtype");
    }
}

} // namespace llaisys::ops::metax
