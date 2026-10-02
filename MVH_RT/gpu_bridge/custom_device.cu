#define MVH_OPTIX_DEVICE
#include "bridge.h"
#include "scoring.cuh"
#include <optix_device.h>

extern "C" { __constant__ mvh_rt_gpu::Params params; }

namespace {
constexpr unsigned NoPeak = ~0u;

struct SearchRay {
    float3 origin;
    float3 direction;
    float tmin;
    float tmax;
};

// Split the query using double subtraction before converting the fraction.
__device__ SearchRay makeSearchRay(double mz) {
    const double integer = floor(mz);
    return {make_float3(float(integer), float(mz - integer), 3.5f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, params.rayTmax};
}

// Closest-hit prefers the highest reachable class; within a class, nearest mass.
__device__ unsigned tracePeak(OptixTraversableHandle handle, double mz) {
    if (!handle) return NoPeak;
    const auto ray = makeSearchRay(mz);
    unsigned peakIndex = NoPeak;
    optixTrace(handle, ray.origin, ray.direction, ray.tmin, ray.tmax, 0.0f,
               255, OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, peakIndex);
    return peakIndex;
}

struct SphereCounter : mvh_cuda::IonCounter {
    __device__ SphereCounter(const mvh_cuda::Scan &scan, const mvh_cuda::Config &cfg,
                             const double *peaks, const int *classes, const short *hub)
        : IonCounter(scan, cfg, peaks, classes, hub) {}

    __device__ void add(double mz) {
        if (mz < scan.lower || mz > scan.upper) return;
        ++predicted;
        const auto handle = params.handles[&scan - params.scans];
        const unsigned peakIndex = tracePeak(handle, mz);
        if (peakIndex == NoPeak) {
            ++key[cfg.classes];
            return;
        }

        // Design point 3: scoring handoff. Class 0 can exist in the geometry,
        // but remains unscored under the existing MVH class convention.
        const int cls = classes[scan.peakOffset + peakIndex];
        if (cls > 0) { ++key[cls - 1]; ++matched; }
        else ++key[cfg.classes];
    }
};
}

// Design point 4: hit payload. Intersection itself is the built-in OptiX
// sphere module registered in rt_support.cpp; there is no custom IS here.
extern "C" __global__ void __closesthit__record() {
    const int scanId = params.candidates[optixGetLaunchIndex().x].scanId;
    const auto offset = params.scans[scanId].peakOffset;
    const auto primitive = params.sphereOffsets[offset] + optixGetPrimitiveIndex();
    optixSetPayload_0(params.spherePeakIndices[primitive] - unsigned(offset));
}

extern "C" __global__ void __miss__background() {
    optixSetPayload_0(NoPeak);
}

// One launch index scores one candidate. Its theoretical ions each call add().
extern "C" __global__ void __raygen__camera() {
    const int index = optixGetLaunchIndex().x;
    params.results[index] = mvh_cuda::scoreCandidate<SphereCounter>(index, params.scans,
        params.candidates, params.size, params.peptides, params.texts, params.peaks,
        params.classes, params.hub, params.lnTable, params.results, params.cfg,
        params.ionOffsets, params.ionValid, params.cachedIons, params.chargeStride);
}
