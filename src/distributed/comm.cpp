#include "comm.hpp"

#include <stdexcept>

namespace llaisys::distributed {
std::shared_ptr<Comm> createMockComm(const Config &config);

#ifdef ENABLE_DIST_NCCL
class NcclCommStub : public Comm {
private:
    int _world_size;
    int _rank;

public:
    explicit NcclCommStub(const Config &config)
        : _world_size(config.world_size),
          _rank(config.rank) {
    }
    Backend backend() const override { return Backend::Nccl; }
    int worldSize() const override { return _world_size; }
    int rank() const override { return _rank; }
    void allReduceSum(float *, size_t) override {
        throw std::runtime_error("NCCL backend is not wired in stage A");
    }
    void barrier() override {
        throw std::runtime_error("NCCL backend is not wired in stage A");
    }
};

static std::shared_ptr<Comm> createNcclComm(const Config &config) {
    return std::make_shared<NcclCommStub>(config);
}
#endif

#ifdef ENABLE_DIST_MPI
class MpiCommStub : public Comm {
private:
    int _world_size;
    int _rank;

public:
    explicit MpiCommStub(const Config &config)
        : _world_size(config.world_size),
          _rank(config.rank) {
    }
    Backend backend() const override { return Backend::Mpi; }
    int worldSize() const override { return _world_size; }
    int rank() const override { return _rank; }
    void allReduceSum(float *, size_t) override {
        throw std::runtime_error("MPI backend is not wired in stage A");
    }
    void barrier() override {
        throw std::runtime_error("MPI backend is not wired in stage A");
    }
};

static std::shared_ptr<Comm> createMpiComm(const Config &config) {
    return std::make_shared<MpiCommStub>(config);
}
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
