#pragma once
#include "types.cuh"
#include "../../mvh_cuda/matching/matching.cuh"
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
    mvh_matching::CandidateSummary* matchSummaries;
    // Optional matching-only mode. Null keeps the existing fused scoring path.
    const mvh_matching::Query *matchQueries;
    int *matchResults;
    const mvh_cuda::Candidate *matchGroups;
    const uint64_t *matchOffsets;
    const double *matchMz;
    uint64_t matchBase;
    mvh_matching::Statistics *matchStatistics;
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
    const mvh_cuda::Config& config,bool aggregate=false);
void launch(Params params);
// prepare(Spheres) must have completed. Only matching executes in this launch.
void launchMatching(const mvh_cuda::Scan* scans, const mvh_matching::Query* queries,
                    int count, int* results, const mvh_cuda::Config& config);
// Stage uploads launch parameters once, outside matching-only event timing.
// Buffers must stay alive; reset or a scoring launch invalidates this staging.
void stageMatching(const mvh_cuda::Scan* scans, const mvh_matching::Query* queries,
                   int count, int* results, const mvh_cuda::Config& config);
void launchStagedMatching();
void stageGroupedMatching(const mvh_cuda::Scan* scans,
    const mvh_cuda::Candidate* groups, const uint64_t* offsets, const double* mz,
    uint64_t base, int count, int* results, const mvh_cuda::Config& config,
    mvh_matching::Statistics* statistics = nullptr,
    mvh_matching::CandidateSummary* summaries=nullptr,const int* classes=nullptr);
}
