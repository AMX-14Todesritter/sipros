#pragma once
#include "bridge.h"

// Internal dispatch interfaces. Application code uses bridge.h only.
namespace mvh_rt_gpu::sphere_backend {
void reset();
void prepare(const std::vector<mvh_cuda::Scan>& scans, const double* peaks,
             const int* classes, size_t peakCount, const mvh_cuda::Config& config,bool aggregate=false,int fusionMode=-1);
void launch(Params params);
void stage(Params params);
void launchStagedMatching();
}

namespace mvh_rt_gpu::triangle_backend {
void reset();
void prepare(const std::vector<mvh_cuda::Scan>& scans, const mvh_cuda::Scan* deviceScans,
             const double* peaks, size_t peakCount, bool instanced);
void launch(Params params);
}
