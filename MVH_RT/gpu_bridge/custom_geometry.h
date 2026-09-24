#pragma once
#include "geometry.h"

namespace mvh_rt_gpu {
// One center per input peak, preserving peak indices, including class 0.
// The uniform sphere radius is stored by bridge.cpp.
void generateSphereCenters(const double *peaks, const int *classes,
                           float3 *centers, size_t count);
}
