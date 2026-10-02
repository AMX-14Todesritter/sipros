#include "profiling.h"
#include "custom_geometry.h"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace mvh_rt_gpu {
namespace {
__global__ void countCenters(const double *peaks, uint64_t *counts,
                             size_t count, double tolerance) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i > count) return;
    if (i == count) { counts[i] = 0; return; }
    const double fraction = peaks[i] - floor(peaks[i]);
    counts[i] = 1 + (fraction <= tolerance) + (1.0 - fraction <= tolerance);
}
__global__ void sphereCentersKernel(const double *peaks, const uint64_t *offsets,
    float3 *centers, unsigned *peakIndices, size_t count, double tolerance) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double integer = floor(peaks[i]);
    const double fraction = peaks[i] - integer;
    uint64_t slot = offsets[i];
    centers[slot] = make_float3(float(integer), float(fraction), 0.0f);
    peakIndices[slot++] = unsigned(i);
    if (fraction <= tolerance) {
        centers[slot] = make_float3(float(integer - 1.0), float(fraction + 1.0), 0.0f);
        peakIndices[slot++] = unsigned(i);
    }
    if (1.0 - fraction <= tolerance) {
        centers[slot] = make_float3(float(integer + 1.0), float(fraction - 1.0), 0.0f);
        peakIndices[slot] = unsigned(i);
    }
}
}
std::vector<uint64_t> sphereCenterOffsets(const double *peaks, size_t count,
                                         double tolerance) {
    MVH_PROFILE_SCOPE("mvh/rt/geometry/sphereCenterOffsets");
    mvh_cuda::Buffer<uint64_t> counts(count + 1), offsets(count + 1);
    countCenters<<<(count + 256) / 256, 256>>>(peaks, counts.p, count, tolerance);
    mvh_cuda::check(cudaGetLastError());
    thrust::exclusive_scan(thrust::device_pointer_cast(counts.p),
        thrust::device_pointer_cast(counts.p + count + 1),
        thrust::device_pointer_cast(offsets.p));
    std::vector<uint64_t> result;
    offsets.read(result);
    return result;
}
void generateSphereCenters(const double *peaks, const uint64_t *offsets,
                           float3 *centers, unsigned *peakIndices,
                           size_t count, double tolerance) {
    MVH_PROFILE_SCOPE("mvh/rt/geometry/generateSphereCenters");
    if (count) sphereCentersKernel<<<(count + 255) / 256, 256>>>(
        peaks, offsets, centers, peakIndices, count, tolerance);
    mvh_cuda::check(cudaGetLastError());
}
}
