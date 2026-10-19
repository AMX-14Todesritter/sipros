#pragma once
#include <optix_device.h>
namespace mvh_rt_gpu::matching {
constexpr unsigned NoPeak = ~0u;

struct SearchRay {
    float3 origin;
    float3 direction;
    float tmin;
    float tmax;
};

// Split the query using double subtraction before converting the fraction.
__device__ SearchRay makeSearchRay(const Params& params, double mz) {
    const double integer = floor(mz);
    return {make_float3(float(integer), float(mz - integer), 0.5f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, params.rayTmax};
}

// Equal-radius spheres: earliest entry chooses the nearest fractional coordinate.
template<class Observer>
__device__ unsigned tracePeakObserved(const Params& params, OptixTraversableHandle handle,
                                     double mz, Observer& observer) {
    observer.begin();
    if (!handle) {observer.noHandle();observer.end(false);return NoPeak;}
    const auto ray = makeSearchRay(params, mz);
    unsigned peakIndex = NoPeak;
    observer.trace();
    optixTrace(handle, ray.origin, ray.direction, ray.tmin, ray.tmax, 0.0f,
               255, OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, peakIndex);
    observer.end(peakIndex != NoPeak);
    return peakIndex;
}
__device__ unsigned tracePeak(const Params& params, OptixTraversableHandle handle, double mz) {
    mvh_matching::NoopObserver observer;
    return tracePeakObserved(params,handle,mz,observer);
}

}
