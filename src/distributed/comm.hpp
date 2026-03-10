#pragma once

#include <cstddef>
#include <memory>

namespace llaisys::distributed {
enum class Backend {
    Mock = 0,
    Nccl = 1,
    Mpi = 2,
};

struct Config {
    Backend backend;
    int world_size;
    int rank;
    int local_device;
};

class Comm {
public:
    virtual ~Comm() = default;

    virtual Backend backend() const = 0;
    virtual int worldSize() const = 0;
    virtual int rank() const = 0;

    virtual void allReduceSum(float *data, size_t count) = 0;
    virtual void barrier() = 0;
};

using comm_t = std::shared_ptr<Comm>;

comm_t createComm(const Config &config);
bool backendAvailable(Backend backend);
const char *backendToString(Backend backend);
} // namespace llaisys::distributed
