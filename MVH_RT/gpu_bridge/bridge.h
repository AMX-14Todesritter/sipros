#pragma once
#include "types.cuh"
#include <optix.h>

namespace mvh_rt_gpu {
enum class GeometryKind { Triangles, InstancedTriangles, Spheres };

struct Params {
    const mvh_cuda::Scan *scans;
    const mvh_cuda::Candidate *candidates;
    const mvh_cuda::PeptideInput *peptides;
    const char *texts;
    const double *peaks;
    const int *classes;
    const short *hub;
    const double *lnTable;
    mvh_cuda::Result *results;
    mvh_cuda::Config cfg;
    const uint64_t *ionOffsets;
    const int *ionValid;
    const double *cachedIons;
    const OptixTraversableHandle *handles;
    int size, chargeStride, instanced;
    float rayOriginY;
    float rayTmax;
};
// Resources persist across peptide batches and reset for each input dataset.
void reset();
void prepare(
    const std::vector<mvh_cuda::Scan>& scans,
    const mvh_cuda::Scan* deviceScans,
    const double* devicePeaks,
    const int* deviceClasses,
    size_t peakCount,
    GeometryKind geometry,
    const mvh_cuda::Config& config);
void launch(Params params);
}
