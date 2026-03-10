#ifndef LLAISYS_DISTRIBUTED_H
#define LLAISYS_DISTRIBUTED_H

#include "../llaisys.h"

__C {
    typedef enum {
        LLAISYS_DIST_BACKEND_MOCK = 0,
        LLAISYS_DIST_BACKEND_NCCL = 1,
        LLAISYS_DIST_BACKEND_MPI = 2,
    } llaisysDistBackend_t;

    struct LlaisysDistConfig {
        llaisysDistBackend_t backend;
        int world_size;
        int rank;
        int local_device;
    };

    typedef struct LlaisysDistComm *llaisysDistComm_t;

    __export llaisysDistComm_t llaisysDistCommCreate(struct LlaisysDistConfig config);
    __export void llaisysDistCommDestroy(llaisysDistComm_t comm);

    __export llaisysDistBackend_t llaisysDistCommBackend(llaisysDistComm_t comm);
    __export int llaisysDistCommWorldSize(llaisysDistComm_t comm);
    __export int llaisysDistCommRank(llaisysDistComm_t comm);

    __export int llaisysDistBackendAvailable(llaisysDistBackend_t backend);

    __export void llaisysDistAllReduceSumF32(llaisysDistComm_t comm, float *data, size_t count);
    __export void llaisysDistBarrier(llaisysDistComm_t comm);

    // 获取内部 comm 实现指针 (供 C++ 内部使用, 返回 shared_ptr<Comm>*)
    __export void *llaisysDistCommGetImplPtr(llaisysDistComm_t comm);
}

#endif // LLAISYS_DISTRIBUTED_H
