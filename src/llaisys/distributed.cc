#include "llaisys/distributed.h"

#include "../distributed/comm.hpp"

#include <memory>
#include <stdexcept>

struct LlaisysDistComm {
    llaisys::distributed::comm_t impl;
};

static llaisys::distributed::Backend _to_backend(llaisysDistBackend_t backend) {
    switch (backend) {
        case LLAISYS_DIST_BACKEND_MOCK:
            return llaisys::distributed::Backend::Mock;
        case LLAISYS_DIST_BACKEND_NCCL:
            return llaisys::distributed::Backend::Nccl;
        case LLAISYS_DIST_BACKEND_MPI:
            return llaisys::distributed::Backend::Mpi;
        default:
            throw std::invalid_argument("unknown dist backend");
    }
}

static llaisysDistBackend_t _from_backend(llaisys::distributed::Backend backend) {
    switch (backend) {
        case llaisys::distributed::Backend::Mock:
            return LLAISYS_DIST_BACKEND_MOCK;
        case llaisys::distributed::Backend::Nccl:
            return LLAISYS_DIST_BACKEND_NCCL;
        case llaisys::distributed::Backend::Mpi:
            return LLAISYS_DIST_BACKEND_MPI;
        default:
            throw std::invalid_argument("unknown dist backend");
    }
}

__C llaisysDistComm_t llaisysDistCommCreate(struct LlaisysDistConfig config) {
    auto handle = new LlaisysDistComm;
    llaisys::distributed::Config cfg;
    cfg.backend = _to_backend(config.backend);
    cfg.world_size = config.world_size;
    cfg.rank = config.rank;
    cfg.local_device = config.local_device;
    handle->impl = llaisys::distributed::createComm(cfg);
    return handle;
}

__C void llaisysDistCommDestroy(llaisysDistComm_t comm) {
    if (comm == nullptr) {
        return;
    }
    delete comm;
}

__C llaisysDistBackend_t llaisysDistCommBackend(llaisysDistComm_t comm) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    return _from_backend(comm->impl->backend());
}

__C int llaisysDistCommWorldSize(llaisysDistComm_t comm) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    return comm->impl->worldSize();
}

__C int llaisysDistCommRank(llaisysDistComm_t comm) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    return comm->impl->rank();
}

__C int llaisysDistBackendAvailable(llaisysDistBackend_t backend) {
    return llaisys::distributed::backendAvailable(_to_backend(backend)) ? 1 : 0;
}

__C void llaisysDistAllReduceSumF32(llaisysDistComm_t comm, float *data, size_t count) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    comm->impl->allReduceSum(data, count);
}

__C void llaisysDistBarrier(llaisysDistComm_t comm) {
    if (comm == nullptr) {
        throw std::invalid_argument("comm is null");
    }
    comm->impl->barrier();
}

__C void *llaisysDistCommGetImplPtr(llaisysDistComm_t comm) {
    if (comm == nullptr) return nullptr;
    // 返回指向 shared_ptr<Comm> 的指针, 调用方需 reinterpret_cast
    return &comm->impl;
}
