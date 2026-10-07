#pragma once
#include "types.cuh"
#include <optix.h>

namespace mvh_rt_gpu {
enum class GeometryKind { Triangles, InstancedTriangles, Spheres };

// Group capacity is a runtime parameter; device reductions use global buffers.
struct SharedTask { int peptideId, charge, groupId, first, count, theoryId; };
struct CollectedHit { unsigned ion, scan, peak; };
void setWorkspaceMiB(int size);
int workspaceMiB();
void setScanGroupSize(int size);
int scanGroupSize();

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
    int scanGroupSize;
    const uint64_t *sharedIonOffsets;
    const int *sharedIonValid;
    const double *sharedIons;
    const SharedTask *sharedTasks;
    const int *sharedCandidateIndices;
    const unsigned *sphereScanIndices;
    const uint64_t *groupSphereOffsets;
    uint64_t *hitCounts;
    const uint64_t *hitOffsets;
    CollectedHit *hits;
    uint64_t hitBase;
    int collectMode; // 0=count, 1=store; both traverse all intersections
    const int *scanGroups;
    unsigned *selectedPeaks; // optional contract observer, candidate-major
    int selectedPeakStride;
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
    const mvh_cuda::Config& config,
    const std::vector<mvh_cuda::Precursor>& precursors = {});
void launch(Params params);
}
