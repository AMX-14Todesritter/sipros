#pragma once
#include "geometry.h"

namespace mvh_rt_gpu {
// Compact base and boundary-copy centers, with original global peak identities.
std::vector<uint64_t> sphereCenterOffsets(const double *peaks, size_t count,
                                         double tolerance);
void generateSphereCenters(const double *peaks, const int *classes, const uint64_t *offsets,
                           float3 *centers, unsigned *peakIndices,
                           size_t count, double tolerance);
}
