#pragma once

#include <cstddef>
#include "llaisys.h"

namespace llaisys::ops::metax {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::metax
