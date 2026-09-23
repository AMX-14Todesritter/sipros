#pragma once
#include "types.cuh"
#include <optix.h>

namespace mvh_rt_gpu {
void generateVertices(const double *peaks, float3 *vertices, size_t count);
void generateBaseTriangle(float3 *vertices);
void generateInstances(const mvh_cuda::Scan *scans, int count, const double *peaks,
                       OptixInstance *instances, OptixTraversableHandle base);
}
