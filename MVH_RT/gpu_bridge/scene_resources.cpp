#include "profiling.h"
#include "scene_resources.h"
#include <optix_stubs.h>
#include <algorithm>

namespace mvh_rt_gpu {
namespace {
size_t alignedAccelerationSize(size_t bytes) {
    return (bytes + OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT - 1)
           & ~size_t(OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT - 1);
}
}

void SceneResources::requireSameLayout(const std::vector<mvh_cuda::Scan>& scans) const {
    MVH_PROFILE_SCOPE("mvh/rt/SceneResources::requireSameLayout");
    if (layout.size() != scans.size())
        throw std::runtime_error("RT layout changed without reset");
    for (size_t i = 0; i < scans.size(); ++i) {
        if (layout[i].peakOffset != scans[i].peakOffset ||
            layout[i].peaks != scans[i].peaks || layout[i].skip != scans[i].skip)
            throw std::runtime_error("RT peak layout changed without reset");
    }
}

AccelerationStats SceneResources::build(const std::vector<OptixBuildInput>& inputs,
                                        const OptixAccelBuildOptions& options) {
    MVH_PROFILE_SCOPE("mvh/rt/SceneResources::build");
    if (inputs.size() != layout.size())
        throw std::runtime_error("RT build input count differs from scan layout");
    AccelerationStats stats;
    std::vector<OptixAccelBufferSizes> sizes(layout.size());
    std::vector<size_t> offsets(layout.size());
    std::vector<OptixTraversableHandle> scanHandles(layout.size());
    MVH_PROFILE_BEGIN(sizingRange, "mvh/rt/accel/memory_usage");
    const auto sizingStart = SetupClock::now();
    for (size_t i = 0; i < layout.size(); ++i) {
        if (layout[i].skip || !layout[i].peaks) continue;
        ++stats.activeScans;
        checkOptix(optixAccelComputeMemoryUsage(objects.context, &options, &inputs[i], 1, &sizes[i]));
        offsets[i] = stats.outputBytes;
        stats.outputBytes += alignedAccelerationSize(sizes[i].outputSizeInBytes);
        stats.scratchBytes = std::max(stats.scratchBytes, sizes[i].tempSizeInBytes);
    }
    MVH_PROFILE_END(sizingRange);
    stats.sizingSeconds = setupSeconds(sizingStart);

    MVH_PROFILE_BEGIN(buildRange, "mvh/rt/accel/build_and_wait");
    const auto buildStart = SetupClock::now();
    acceleration = std::make_unique<DeviceBuffer<unsigned char>>(stats.outputBytes);
    DeviceBuffer<unsigned char> scratch(stats.scratchBytes);
    // Default-stream ordering permits one scratch allocation for all scans.
    for (size_t i = 0; i < layout.size(); ++i) {
        if (layout[i].skip || !layout[i].peaks) continue;
        checkOptix(optixAccelBuild(objects.context, nullptr, &options, &inputs[i], 1,
            scratch.address(), scratch.count, acceleration->address() + offsets[i],
            sizes[i].outputSizeInBytes, &scanHandles[i], nullptr, 0));
    }
    checkCuda(cudaDeviceSynchronize());
    MVH_PROFILE_END(buildRange);
    stats.buildSeconds = setupSeconds(buildStart);
    handles = std::make_unique<DeviceBuffer<OptixTraversableHandle>>(scanHandles);
    return stats;
}

void SceneResources::initializeSbt() {
    MVH_PROFILE_SCOPE("mvh/rt/SceneResources::initializeSbt");
    SbtRecord<EmptyData> raygenRecord{}, missRecord{}, hitRecord{};
    checkOptix(optixSbtRecordPackHeader(objects.raygen, &raygenRecord));
    checkOptix(optixSbtRecordPackHeader(objects.miss, &missRecord));
    checkOptix(optixSbtRecordPackHeader(objects.hit, &hitRecord));
    checkCuda(cudaMemcpy(raygen.data, &raygenRecord, sizeof(raygenRecord), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(miss.data, &missRecord, sizeof(missRecord), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(hit.data, &hitRecord, sizeof(hitRecord), cudaMemcpyHostToDevice));
    sbt.raygenRecord = raygen.address();
    sbt.missRecordBase = miss.address();
    sbt.missRecordStrideInBytes = sizeof(missRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = hit.address();
    sbt.hitgroupRecordStrideInBytes = sizeof(hitRecord);
    sbt.hitgroupRecordCount = 1;
}

void SceneResources::launch(Params params) {
    MVH_PROFILE_SCOPE("mvh/rt/SceneResources::launch");
    params.handles = handles->data;
    MVH_PROFILE_BEGIN(paramsRange, "mvh/rt/launch/upload_params");
    checkCuda(cudaMemcpy(parameters.data, &params, sizeof(params), cudaMemcpyHostToDevice));
    MVH_PROFILE_END(paramsRange);
    MVH_PROFILE_SCOPE("mvh/rt/launch/optixLaunch_submit");
    checkOptix(optixLaunch(objects.pipeline, nullptr, parameters.address(), sizeof(params),
                          &sbt, params.size, 1, 1));
}
}
