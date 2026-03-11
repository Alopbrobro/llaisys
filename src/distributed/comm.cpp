#include "comm.hpp"

#include <stdexcept>

namespace llaisys::distributed {
std::shared_ptr<Comm> createMockComm(const Config &config);

#ifdef ENABLE_DIST_NCCL
// 真正的 NCCL 实现在 nccl_comm.cpp 中
std::shared_ptr<Comm> createNcclComm(const Config &config);
#endif

#ifdef ENABLE_DIST_MPI
// 真正的 MPI 实现在 mpi_comm.cpp 中
std::shared_ptr<Comm> createMpiComm(const Config &config);
#endif

std::shared_ptr<Comm> createComm(const Config &config) {
    switch (config.backend) {
        case Backend::Mock:
            return createMockComm(config);
        case Backend::Nccl:
#ifdef ENABLE_DIST_NCCL
            return createNcclComm(config);
#else
            throw std::runtime_error("NCCL backend is disabled at compile-time");
#endif
        case Backend::Mpi:
#ifdef ENABLE_DIST_MPI
            return createMpiComm(config);
#else
            throw std::runtime_error("MPI backend is disabled at compile-time");
#endif
        default:
            throw std::invalid_argument("unknown backend");
    }
}

bool backendAvailable(Backend backend) {
    switch (backend) {
        case Backend::Mock:
            return true;
        case Backend::Nccl:
#ifdef ENABLE_DIST_NCCL
            return true;
#else
            return false;
#endif
        case Backend::Mpi:
#ifdef ENABLE_DIST_MPI
            return true;
#else
            return false;
#endif
        default:
            return false;
    }
}

const char *backendToString(Backend backend) {
    switch (backend) {
        case Backend::Mock:
            return "mock";
        case Backend::Nccl:
            return "nccl";
        case Backend::Mpi:
            return "mpi";
        default:
            return "unknown";
    }
}
} // namespace llaisys::distributed
