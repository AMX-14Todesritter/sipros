#include "custom_geometry.h"

namespace mvh_rt_gpu {
namespace {
__global__ void sphereCentersKernel(
    const double* peaks,
    const int* classes,
    float3* centers,
    size_t count)
{
    const size_t index =
        size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= count) return;

    centers[index] = make_float3(
        static_cast<float>(peaks[index]),
        static_cast<float>(classes[index]),
        0.0f);
}
}

void generateSphereCenters(
    const double* peaks,
    const int* classes,
    float3* centers,
    size_t count)
{
    if (count) {
        sphereCentersKernel<<<(count + 255) / 256, 256>>>(
            peaks, classes, centers, count);
    }
    mvh_cuda::check(cudaGetLastError());
}
}
