// ================================================================
// GPU Sample Operator — Temperature + Top-K + Softmax + Top-P + cuRAND
//
// Single-block design, no global temp buffers.
//
// Algorithm
// ---------
// 1. All threads cooperatively apply temperature scaling to logits.
// 2. K rounds of parallel max-reduction to find top-K candidates.
//    After each round the winning position is masked to −inf.
// 3. Thread 0 computes softmax over the K candidates, applies
//    top-P (nucleus) cutoff, re-normalises, and draws a sample
//    using the Philox PRNG.
//
// Shared-memory layout (bytes)
// ----------------------------
//   float  s_val  [blockDim.x]   — reduction values
//   int    s_idx  [blockDim.x]   — reduction indices
//   float  cand_v [top_k]        — candidate logit values
//   int    cand_i [top_k]        — candidate token indices
// ================================================================

#include "sample_nvidia.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>
#include <cstdio>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "[CUDA ERROR] %s (code %d) at %s:%d\n",           \
                    cudaGetErrorString(err), (int)err, __FILE__, __LINE__);    \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

__global__ void sample_kernel(
    int       *out_idx,
    float     *logits,
    int64_t    vocab_size,
    int64_t    stride,       // element stride in the logits buffer
    float      inv_temp,     // 1 / temperature  (1.0 = no scaling)
    int        top_k,
    float      top_p,
    uint64_t   seed)
{
    extern __shared__ char smem[];
    const int T   = blockDim.x;
    const int tid = threadIdx.x;

    float *s_val  = reinterpret_cast<float *>(smem);
    int   *s_idx  = reinterpret_cast<int   *>(s_val  + T);
    float *cand_v = reinterpret_cast<float *>(s_idx  + T);
    int   *cand_i = reinterpret_cast<int   *>(cand_v + top_k);

    // ---- Step 1: temperature scaling --------------------------------
    if (inv_temp != 1.0f) {
        for (int64_t i = tid; i < vocab_size; i += T)
            logits[i * stride] *= inv_temp;
        __syncthreads();
    }

    // ---- Step 2: K rounds of max-find + mask ------------------------
    for (int round = 0; round < top_k; ++round) {
        float local_max = -FLT_MAX;
        int   local_idx = 0;
        for (int64_t i = tid; i < vocab_size; i += T) {
            float v = logits[i * stride];
            if (v > local_max) { local_max = v; local_idx = static_cast<int>(i); }
        }
        s_val[tid] = local_max;
        s_idx[tid] = local_idx;
        __syncthreads();

        // block-level tree reduction
        for (int s = T >> 1; s > 0; s >>= 1) {
            if (tid < s && s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            cand_v[round] = s_val[0];
            cand_i[round] = s_idx[0];
            logits[s_idx[0] * stride] = -FLT_MAX;   // mask winner
        }
        __syncthreads();
    }

    // ---- Steps 3-5 (thread 0 only) ---------------------------------
    if (tid != 0) return;

    // 3. Softmax over candidates
    float max_logit = cand_v[0];
    float sum = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        cand_v[i] = expf(cand_v[i] - max_logit);
        sum += cand_v[i];
    }
    for (int i = 0; i < top_k; ++i)
        cand_v[i] /= sum;

    // 4. Top-P (nucleus) cutoff
    int cutoff = top_k;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            cum += cand_v[i];
            if (cum >= top_p) { cutoff = i + 1; break; }
        }
        // re-normalise
        float new_sum = 0.0f;
        for (int i = 0; i < cutoff; ++i) new_sum += cand_v[i];
        for (int i = 0; i < cutoff; ++i) cand_v[i] /= new_sum;
    }

    // 5. Draw random sample using Philox RNG
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, 0, 0, &rng);
    float r = curand_uniform(&rng);

    float cum = 0.0f;
    int   result = cand_i[0];
    for (int i = 0; i < cutoff; ++i) {
        cum += cand_v[i];
        if (r <= cum) { result = cand_i[i]; break; }
    }
    out_idx[0] = result;
}

// ================================================================

namespace llaisys::ops::nvidia {

void sample(tensor_t out_idx, tensor_t logits,
            float temperature, int top_k, float top_p, uint64_t seed)
{
    if (logits->dtype() != LLAISYS_DTYPE_F32)
        throw std::runtime_error("NVIDIA sample: only F32 logits supported");

    int64_t vocab_size = 0;
    int64_t stride     = 0;
    if (logits->ndim() == 1) {
        vocab_size = logits->shape()[0];
        stride     = logits->strides()[0];
    } else if (logits->ndim() == 2) {
        if (logits->shape()[0] != 1)
            throw std::runtime_error("sample: 2D input must have shape [1, N]");
        vocab_size = logits->shape()[1];
        stride     = logits->strides()[1];
    } else {
        throw std::runtime_error("sample: only supports 1D or 2D logits");
    }

    // Clamp top_k
    if (top_k <= 0 || top_k > static_cast<int>(vocab_size))
        top_k = static_cast<int>(vocab_size);
    if (top_k > 1024)
        top_k = 1024;   // cap to limit kernel rounds

    float inv_temp = (temperature > 0.0f && temperature != 1.0f)
                         ? (1.0f / temperature) : 1.0f;

    constexpr int threads = 256;
    size_t smem_bytes = threads * (sizeof(float) + sizeof(int))
                      + top_k  * (sizeof(float) + sizeof(int));

    sample_kernel<<<1, threads, smem_bytes>>>(
        reinterpret_cast<int *>(out_idx->data()),
        reinterpret_cast<float *>(logits->data()),
        vocab_size, stride, inv_temp, top_k, top_p, seed);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
