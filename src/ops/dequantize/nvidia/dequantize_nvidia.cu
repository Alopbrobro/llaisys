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

} // namespace llaisys::ops::nvidia
