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

// ─────────────────────────────────────────────
// AWQ INT4 packed (int32) → FP32 (per-group asymmetric, output-packed)
// ─────────────────────────────────────────────

/// Each thread processes ONE int32 from qweight and writes EIGHT FP32 outputs.
/// AWQ GEMM packing: output dimension is packed with interleaved bit order.
///   qweight: [in_features, out_packed]  int32  (out_packed = out_features / 8)
///   qzeros:  [num_groups, out_packed]   int32
///   scales:  [num_groups, out_features] float32
///   groups along in_features dimension: group_idx = in_idx / group_size
///
/// AWQ GEMM interleaved packing order: the 8 INT4 values within each int32 are
/// stored at bit positions [0,16,4,20,8,24,12,28] corresponding to output
/// column offsets [0,1,2,3,4,5,6,7] (element order [0,4,1,5,2,6,3,7]).
///
/// Output is TRANSPOSED: [out_features, in_features] for direct use in linear.
__global__ void dequantize_awq_int4_f32_kernel(
    float*         out,           // [out_features, in_features]
    const int32_t* qweight,       // [in_features, out_packed]
    const int32_t* qzeros,        // [num_groups, out_packed]
    const float*   scales,        // [num_groups, out_features]
    int64_t        in_features,
    int64_t        out_features,
    int64_t        out_packed,    // out_features / 8
    int64_t        num_groups,
    int            group_size)
{
    // AWQ GEMM interleaved bit shifts: output col i uses bit shift awq_shifts[i]
    constexpr int awq_shifts[8] = {0, 16, 4, 20, 8, 24, 12, 28};

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = in_features * out_packed;
    if (tid >= total) return;

    int64_t in_idx = tid / out_packed;
    int64_t pc     = tid % out_packed;
    int64_t group  = in_idx / group_size;

    int32_t packed_w = qweight[tid];
    int32_t packed_z = qzeros[group * out_packed + pc];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int val  = (packed_w >> awq_shifts[i]) & 0xF;
        int zero = (packed_z >> awq_shifts[i]) & 0xF;
        int64_t out_col = pc * 8 + i;
        float s = scales[group * out_features + out_col];
        // Write transposed: out[out_col][in_idx]
        out[out_col * in_features + in_idx] = __int2float_rn(val - zero) * s;
    }
}

void dequantize_awq_int4(tensor_t out, tensor_t qweight, tensor_t qzeros, tensor_t scales, int group_size) {
    int64_t out_features = out->shape()[0];      // output rows (transposed)
    int64_t in_features  = out->shape()[1];      // output cols (transposed)
    int64_t out_packed   = qweight->shape()[1];  // out_features / 8
    int64_t num_groups   = scales->shape()[0];
    int64_t total        = in_features * out_packed;

    const int threads = 256;
    const int blocks  = (int)((total + threads - 1) / threads);

    dequantize_awq_int4_f32_kernel<<<blocks, threads>>>(
        reinterpret_cast<float*>(out->data()),
        reinterpret_cast<const int32_t*>(qweight->data()),
        reinterpret_cast<const int32_t*>(qzeros->data()),
        reinterpret_cast<const float*>(scales->data()),
        in_features, out_features, out_packed, num_groups, group_size);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
