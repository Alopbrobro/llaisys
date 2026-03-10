#include "comm.hpp"

#include <stdexcept>

namespace llaisys::distributed {
class MockComm : public Comm {
private:
    int _world_size;
    int _rank;

public:
    explicit MockComm(const Config &config)
        : _world_size(config.world_size),
          _rank(config.rank) {
        if (_world_size <= 0) {
            throw std::invalid_argument("world_size must be > 0");
        }
        if (_rank < 0 || _rank >= _world_size) {
            throw std::invalid_argument("rank out of range");
        }
    }

    Backend backend() const override { return Backend::Mock; }
    int worldSize() const override { return _world_size; }
    int rank() const override { return _rank; }

    void allReduceSum(float *data, size_t count) override {
        if (data == nullptr && count != 0) {
            throw std::invalid_argument("data is null");
        }
    }

    void barrier() override {
    }
};

std::shared_ptr<Comm> createMockComm(const Config &config) {
    return std::make_shared<MockComm>(config);
}
} // namespace llaisys::distributed
