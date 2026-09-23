#pragma once
#include "types.cuh"
#include <optix.h>

namespace mvh_rt_gpu {
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
};
// Resources persist across peptide batches and reset for each input dataset.
void reset();
void prepare(const std::vector<mvh_cuda::Scan>& scans,
             const mvh_cuda::Scan *deviceScans, const double *devicePeaks,
             size_t peakCount, bool instanced = false);
void launch(Params params);
}
