#define MVH_OPTIX_DEVICE
#include "bridge.h"
#include <optix_device.h>
extern "C" { __constant__ mvh_rt_gpu::Params params; }
namespace {
struct Query { uint64_t count; unsigned ion; int group; uint64_t start; };
}
extern "C" __global__ void __anyhit__record() {
    const auto pointer=(static_cast<unsigned long long>(optixGetPayload_1())<<32)|optixGetPayload_0();
    auto &q=*reinterpret_cast<Query*>(pointer);
    const auto primitive=params.groupSphereOffsets[q.group]+optixGetPrimitiveIndex();
    if(params.collectMode) {
        // A second traversal must fit the first traversal's exact allocation.
        const unsigned task=optixGetLaunchIndex().x;
        if(q.count>=params.hitOffsets[task+1]-params.hitBase-q.start) {
            params.hitCounts[task]=~uint64_t(0); optixIgnoreIntersection(); return;
        }
        params.hits[q.start+q.count]={q.ion,params.sphereScanIndices[primitive],params.spherePeakIndices[primitive]};
    }
    ++q.count;
    optixIgnoreIntersection();
}
extern "C" __global__ void __miss__background() {}
extern "C" __global__ void __raygen__camera() {
    const unsigned index=optixGetLaunchIndex().x;
    const auto &task=params.sharedTasks[index];
    Query q{};q.group=task.groupId;q.start=params.collectMode?params.hitOffsets[index]-params.hitBase:0;
    const bool valid=params.sharedIonValid[task.theoryId]>0;
    const auto handle=params.handles[task.groupId];
    if(valid && handle) {
        const uint64_t begin=params.sharedIonOffsets[task.theoryId];
        const uint64_t end=params.sharedIonOffsets[task.theoryId+1];
        for(uint64_t i=begin;i<end;++i) {
            const double mz=params.sharedIons[i];q.ion=unsigned(i-begin);
            const double integer=floor(mz);
            const auto pointer=reinterpret_cast<unsigned long long>(&q);
            unsigned lo=unsigned(pointer), hi=unsigned(pointer>>32);
            optixTrace(handle,make_float3(float(integer),float(mz-integer),0.5f),
                make_float3(0,0,-1),0.0f,params.rayTmax,0.0f,255,
                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,0,1,0,lo,hi);
        }
    }
    // Counts are checked by CUDA before reduction. No scores are produced here.
    if(!params.collectMode || params.hitCounts[index]!=~uint64_t(0))params.hitCounts[index]=q.count;
}
