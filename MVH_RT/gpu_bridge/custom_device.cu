#define MVH_OPTIX_DEVICE
#include "bridge.h"
#include "scoring.cuh"
#include <optix_device.h>
#include "sphere_matching.cuh"

extern "C" { __constant__ mvh_rt_gpu::Params params; }

namespace {
using mvh_rt_gpu::matching::NoPeak;
using mvh_rt_gpu::matching::tracePeak;

struct SphereCounter : mvh_cuda::IonCounter {
    __device__ SphereCounter(const mvh_cuda::Scan &scan, const mvh_cuda::Config &cfg,
                             const double *peaks, const int *classes, const short *hub)
        : IonCounter(scan, cfg, peaks, classes, hub) {}

    __device__ void add(double mz) {
        if (mz < scan.lower || mz > scan.upper) return;
        ++predicted;
        const auto handle = params.handles[&scan - params.scans];
        const unsigned peakIndex = tracePeak(params, handle, mz);
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
    const int launchIndex = optixGetLaunchIndex().x;
    const int scanId = params.matchQueries ? params.matchQueries[launchIndex].scanId
        : params.matchGroups ? params.matchGroups[launchIndex].scanId
                             : params.candidates[launchIndex].scanId;
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
    if (params.matchGroups) {
        const int scanId=params.matchGroups[index].scanId;
        const auto handle=params.handles[scanId];
#ifdef MVH_MATCHING_DIAGNOSTICS
        mvh_matching::CountingObserver observer;
#else
        mvh_matching::NoopObserver observer;
#endif
#ifdef MVH_MATCHING_AGGREGATE
        mvh_matching::CandidateSummary summary{};
#endif
        for(uint64_t j=params.matchOffsets[index]-params.matchBase;
            j<params.matchOffsets[index+1]-params.matchBase;++j) {
            const unsigned peak=mvh_rt_gpu::matching::tracePeakObserved(params,handle,params.matchMz[j],observer);
            const int selected=peak==NoPeak?mvh_matching::NoPeak:int(peak);
#ifdef MVH_MATCHING_AGGREGATE
            summary.add(selected,params.scans[scanId],params.classes,params.cfg.classes);
#else
            params.matchResults[j]=selected;
#endif
        }
#ifdef MVH_MATCHING_AGGREGATE
        params.matchSummaries[index]=summary;
#endif
#ifdef MVH_MATCHING_DIAGNOSTICS
        if(params.matchStatistics)params.matchStatistics[index]=observer.stats;
#endif
        return;
    }
    if (params.matchQueries) {
        const auto query = params.matchQueries[index];
        const unsigned peak = tracePeak(params, params.handles[query.scanId], query.mz);
        params.matchResults[index] = peak == NoPeak ? mvh_matching::NoPeak : int(peak);
        return;
    }
    params.results[index] = mvh_cuda::scoreCandidate<SphereCounter>(index, params.scans,
        params.candidates, params.size, params.peptides, params.texts, params.peaks,
        params.classes, params.hub, params.lnTable, params.results, params.cfg,
        params.ionOffsets, params.ionValid, params.cachedIons, params.chargeStride);
}
