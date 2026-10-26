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
    const uint64_t *sphereOffsets;
    const unsigned *spherePeakIndices;
    float rayOriginY;
    float rayTmax;
#ifdef MVH_MATCHING_ONLY
    // Test-only ABI: production Params remains unchanged without this define.
    const uint64_t* matchOffsets;
    const double* matchMz;
    uint64_t matchBase;
    int* matchClasses;
#endif
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
void launch(Params params, cudaEvent_t fusedStart = nullptr, cudaEvent_t fusedEnd = nullptr);
#ifdef MVH_MATCHING_ONLY
void stageMatching(Params params);
void launchStagedMatching();
#endif
}
