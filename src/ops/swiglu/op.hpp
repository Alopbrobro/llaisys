#pragma once

#include "../../tensor/tensor.hpp"
#include <cmath>

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
}
