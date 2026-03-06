#include "dequantize_nvidia.cuh"

#include <cuda_runtime.h>
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

/// CUDA kernel: dequantize INT8 → FP32 with per-channel scale
/// out[i*cols + j] = (float)weight[i*cols + j] * scale[i]
__global__ void dequantize_i8_f32_kernel(
    float*       out,
    const int8_t* weight,
    const float*  scale,
    int64_t       rows,
    int64_t       cols)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (tid >= total) return;

    int64_t row = tid / cols;
    out[tid] = __int2float_rn(weight[tid]) * scale[row];
}

namespace llaisys::ops::nvidia {

void dequantize(tensor_t out, tensor_t weight, tensor_t scale) {
    int64_t rows = weight->shape()[0];
    int64_t cols = weight->shape()[1];
    int64_t total = rows * cols;

    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);

    dequantize_i8_f32_kernel<<<blocks, threads>>>(
        reinterpret_cast<float*>(out->data()),
        reinterpret_cast<const int8_t*>(weight->data()),
        reinterpret_cast<const float*>(scale->data()),
        rows, cols);

    CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────
// INT4 packed (uint8) → FP32  (per-group symmetric)
// ─────────────────────────────────────────────

/// Each thread processes ONE packed byte and writes TWO FP32 outputs.
/// The packed weight has shape [rows, cols/2]; each byte holds 2 INT4 values.
///   low nibble  (byte & 0x0F) - 8  → even column
///   high nibble (byte >> 4)   - 8  → odd  column
__global__ void dequantize_int4_f32_kernel(
    float*         out,
    const uint8_t* weight,
    const float*   scale,
    int64_t        rows,
    int64_t        cols,          // original cols (= packed_cols * 2)
    int64_t        packed_cols,   // cols / 2
    int64_t        num_groups,
    int            group_size)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; // tid over packed bytes
    int64_t total_packed = rows * packed_cols;
    if (tid >= total_packed) return;

    int64_t row = tid / packed_cols;
    int64_t pc  = tid % packed_cols;
    int64_t col_even = pc * 2;
    int64_t col_odd  = col_even + 1;

    uint8_t byte = weight[tid];
    int val_even = (int)(byte & 0x0F) - 8;
    int val_odd  = (int)(byte >> 4) - 8;

    int64_t group_even = col_even / group_size;
    int64_t group_odd  = col_odd  / group_size;
    const float* srow = scale + row * num_groups;

    float s_even = srow[group_even];
    float s_odd  = srow[group_odd];

    int64_t base = row * cols;
    out[base + col_even] = __int2float_rn(val_even) * s_even;
    out[base + col_odd]  = __int2float_rn(val_odd) * s_odd;
}

void dequantize_int4(tensor_t out, tensor_t weight, tensor_t scale, int group_size) {
    int64_t rows        = out->shape()[0];
    int64_t cols        = out->shape()[1];           // original columns
    int64_t packed_cols = weight->shape()[1];         // cols / 2
    int64_t num_groups  = scale->shape()[1];          // cols / group_size
    int64_t total_packed = rows * packed_cols;

    const int threads = 256;
    const int blocks  = (int)((total_packed + threads - 1) / threads);

    dequantize_int4_f32_kernel<<<blocks, threads>>>(
        reinterpret_cast<float*>(out->data()),
        reinterpret_cast<const uint8_t*>(weight->data()),
        reinterpret_cast<const float*>(scale->data()),
        rows, cols, packed_cols, num_groups, group_size);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
