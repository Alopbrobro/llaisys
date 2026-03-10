#pragma once

#include "../device_resource.hpp"

namespace llaisys::device::metax {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id = 0);
    ~Resource() = default;
};
} // namespace llaisys::device::metax
