#pragma once
#include "bridge.h"
#include <functional>
#include <memory>
namespace mvh_rt_gpu {
struct GroupedGeometry {
    mvh_cuda::Buffer<float3> centers;
    mvh_cuda::Buffer<unsigned> peaks, scans;
    mvh_cuda::Buffer<int> scanGroups;
    mvh_cuda::Buffer<uint64_t> offsets;
    std::vector<uint64_t> hostOffsets;
    explicit GroupedGeometry(int scanCount) : scanGroups(scanCount) {}
};
std::unique_ptr<GroupedGeometry> prepareGroupedGeometry(
    const std::vector<mvh_cuda::Scan>& scans, const double *peaks, const int *classes,
    const std::vector<mvh_cuda::Precursor>& precursors, int groupSize, double tolerance);
void executeShared(Params params, const int *scanGroups, int groupCount,
                   const std::function<void(Params)>& trace);
}
