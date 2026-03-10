// =============================================================
// Metax (沐曦) C500 Runtime API — 骨架实现
// =============================================================
// 当前为骨架版本：
//   - 若定义了 ENABLE_METAX_RUNTIME（即 MXMACA SDK 可用），
//     则调用 MXMACA Runtime API（maca_runtime_api.h）。
//   - 否则编译为 stub，所有函数抛出 "MetaX runtime not available"。
//
// 后续阶段 B 将在此文件中填入真实 MXMACA 调用。
// =============================================================

#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifdef ENABLE_METAX_RUNTIME
// ---- 真实 MXMACA SDK 路径 (阶段 B 实现) ----
#include <maca_runtime_api.h>

// MACA 错误检查宏
#define MACA_CHECK(call)                                                           \
    do {                                                                           \
        macaError_t err = (call);                                                  \
        if (err != macaSuccess) {                                                  \
            fprintf(stderr, "[MACA ERROR] %s (code %d) at %s:%d\n",               \
                    macaGetErrorString(err), (int)err, __FILE__, __LINE__);        \
            throw std::runtime_error(macaGetErrorString(err));                     \
        }                                                                          \
    } while (0)

static macaMemcpyKind toMacaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return macaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return macaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return macaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return macaMemcpyDeviceToDevice;
    default:                 return macaMemcpyDefault;
    }
}

namespace llaisys::device::metax {
namespace runtime_api {

int getDeviceCount() {
    int count = 0;
    MACA_CHECK(macaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    MACA_CHECK(macaSetDevice(device));
}

void deviceSynchronize() {
    MACA_CHECK(macaDeviceSynchronize());
}

llaisysStream_t createStream() {
    macaStream_t stream = NULL;
    MACA_CHECK(macaStreamCreate(&stream));
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    MACA_CHECK(macaStreamDestroy((macaStream_t)stream));
}

void streamSynchronize(llaisysStream_t stream) {
    MACA_CHECK(macaStreamSynchronize((macaStream_t)stream));
}

void *mallocDevice(size_t size) {
    void *ptr = NULL;
    MACA_CHECK(macaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    MACA_CHECK(macaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = NULL;
    MACA_CHECK(macaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    MACA_CHECK(macaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    MACA_CHECK(macaMemcpy(dst, src, size, toMacaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    MACA_CHECK(macaMemcpyAsync(dst, src, size, toMacaMemcpyKind(kind), (macaStream_t)stream));
}

} // namespace runtime_api
} // namespace llaisys::device::metax

#else
// ---- Stub 路径：无 MXMACA SDK 时编译此分支 ----

#define METAX_STUB_THROW(funcname) \
    throw std::runtime_error("MetaX runtime not available: " #funcname " (compile with ENABLE_METAX_RUNTIME)")

namespace llaisys::device::metax {
namespace runtime_api {

int getDeviceCount() {
    // Stub: 返回 0 表示没有设备，不抛异常
    return 0;
}

void setDevice(int) {
    METAX_STUB_THROW(setDevice);
}

void deviceSynchronize() {
    METAX_STUB_THROW(deviceSynchronize);
}

llaisysStream_t createStream() {
    METAX_STUB_THROW(createStream);
    return nullptr;
}

void destroyStream(llaisysStream_t) {
    METAX_STUB_THROW(destroyStream);
}

void streamSynchronize(llaisysStream_t) {
    METAX_STUB_THROW(streamSynchronize);
}

void *mallocDevice(size_t) {
    METAX_STUB_THROW(mallocDevice);
    return nullptr;
}

void freeDevice(void *) {
    METAX_STUB_THROW(freeDevice);
}

void *mallocHost(size_t) {
    METAX_STUB_THROW(mallocHost);
    return nullptr;
}

void freeHost(void *) {
    METAX_STUB_THROW(freeHost);
}

void memcpySync(void *, const void *, size_t, llaisysMemcpyKind_t) {
    METAX_STUB_THROW(memcpySync);
}

void memcpyAsync(void *, const void *, size_t, llaisysMemcpyKind_t, llaisysStream_t) {
    METAX_STUB_THROW(memcpyAsync);
}

} // namespace runtime_api
} // namespace llaisys::device::metax

#undef METAX_STUB_THROW
#endif // ENABLE_METAX_RUNTIME

// ---- 公共：组装 LlaisysRuntimeAPI 函数指针表 ----
namespace llaisys::device::metax {

static const LlaisysRuntimeAPI RUNTIME_API = {
    &runtime_api::getDeviceCount,
    &runtime_api::setDevice,
    &runtime_api::deviceSynchronize,
    &runtime_api::createStream,
    &runtime_api::destroyStream,
    &runtime_api::streamSynchronize,
    &runtime_api::mallocDevice,
    &runtime_api::freeDevice,
    &runtime_api::mallocHost,
    &runtime_api::freeHost,
    &runtime_api::memcpySync,
    &runtime_api::memcpyAsync};

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &RUNTIME_API;
}

} // namespace llaisys::device::metax
