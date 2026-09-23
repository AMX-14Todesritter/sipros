#define MVH_OPTIX_DEVICE
#include "bridge.h"
#include "scoring.cuh"
#include <optix_device.h>
#include <math_constants.h>

extern "C" { __constant__ mvh_rt_gpu::Params params; }

namespace {
struct RtCounter : mvh_cuda::IonCounter {
    __device__ RtCounter(const mvh_cuda::Scan&s,const mvh_cuda::Config&c,
                        const double*p,const int*cl,const short*h)
        : IonCounter(s,c,p,cl,h) {}
    __device__ void add(double mz) {
        if (mz < scan.lower || mz > scan.upper) return;
        ++predicted;
        const auto handle = params.handles[&scan - params.scans];
        unsigned right=~0u, left=~0u;
        unsigned rt=__float_as_uint(CUDART_INF_F), lt=rt;
        const float3 origin=make_float3(float(mz),0,0);
        const float limit=float(cfg.fragmentTolerance);
        if (handle) {
            optixTrace(handle,origin,make_float3(1,0,0),0.0f,limit,0.0f,
                255,OPTIX_RAY_FLAG_DISABLE_ANYHIT,0,1,0,right,rt);
            const float leftLimit=right!=~0u ? fminf(limit,__uint_as_float(rt)) : limit;
            optixTrace(handle,origin,make_float3(-1,0,0),0.0f,leftLimit,0.0f,
                255,OPTIX_RAY_FLAG_DISABLE_ANYHIT,0,1,0,left,lt);
        }
        unsigned best=right;
        if (left!=~0u && (right==~0u || __uint_as_float(lt)<__uint_as_float(rt))) best=left;
        const int cls=best==~0u ? 0 : classes[scan.peakOffset+best];
        if (cls>0) { ++key[cls-1]; ++matched; }
        else ++key[cfg.classes];
    }
};
}
extern "C" __global__ void __raygen__camera() {
    const int i=optixGetLaunchIndex().x;
    params.results[i]=mvh_cuda::scoreCandidate<RtCounter>(i,params.scans,
        params.candidates,params.size,params.peptides,params.texts,params.peaks,
        params.classes,params.hub,params.lnTable,params.results,params.cfg,
        params.ionOffsets,params.ionValid,params.cachedIons,params.chargeStride);
}
extern "C" __global__ void __closesthit__record() {
    optixSetPayload_0(params.instanced ? optixGetInstanceId() : optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}
extern "C" __global__ void __miss__background() {
    optixSetPayload_0(~0u);
    optixSetPayload_1(__float_as_uint(CUDART_INF_F));
}
