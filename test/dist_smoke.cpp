#include "llaisys/distributed.h"

#include <iostream>
#include <stdexcept>

int main() {
    LlaisysDistConfig cfg{};
    cfg.backend = LLAISYS_DIST_BACKEND_MOCK;
    cfg.world_size = 1;
    cfg.rank = 0;
    cfg.local_device = 0;

    llaisysDistComm_t comm = llaisysDistCommCreate(cfg);
    if (comm == nullptr) {
        throw std::runtime_error("failed to create distributed comm");
    }

    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    llaisysDistAllReduceSumF32(comm, data, 4);
    llaisysDistBarrier(comm);

    if (llaisysDistCommWorldSize(comm) != 1 || llaisysDistCommRank(comm) != 0) {
        llaisysDistCommDestroy(comm);
        throw std::runtime_error("unexpected comm metadata");
    }

    std::cout << "[dist-smoke] backend=" << llaisysDistCommBackend(comm)
              << " world_size=" << llaisysDistCommWorldSize(comm)
              << " rank=" << llaisysDistCommRank(comm)
              << " data0=" << data[0] << std::endl;

    llaisysDistCommDestroy(comm);
    return 0;
}
