#pragma once

#include <cstddef>
#include "llaisys.h"

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::nvidia
