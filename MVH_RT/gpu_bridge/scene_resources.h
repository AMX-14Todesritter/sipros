#pragma once
#include "bridge.h"
#include "rt_support.h"
#include <chrono>

namespace mvh_rt_gpu {
using SetupClock = std::chrono::steady_clock;
inline double setupSeconds(SetupClock::time_point start) {
    return std::chrono::duration<double>(SetupClock::now() - start).count();
}

struct AccelerationStats {
    size_t activeScans = 0;
    size_t outputBytes = 0;
    size_t scratchBytes = 0;
    double sizingSeconds = 0;
    double buildSeconds = 0;
};

// Shared OptiX plumbing only. Geometry buffers and search policy belong to
// sphere_backend.cpp or triangle_backend.cpp, not to this resource holder.
struct SceneResources {
    OptixObjects objects;
    std::vector<mvh_cuda::Scan> layout;
    std::unique_ptr<DeviceBuffer<unsigned char>> acceleration;
    std::unique_ptr<DeviceBuffer<OptixTraversableHandle>> handles;
    DeviceBuffer<SbtRecord<EmptyData>> raygen{1}, miss{1}, hit{1};
    DeviceBuffer<Params> parameters{1};
    OptixShaderBindingTable sbt{};

    void requireSameLayout(const std::vector<mvh_cuda::Scan>& scans) const;
    // Build-input pointer storage must outlive this synchronous operation.
    AccelerationStats build(const std::vector<OptixBuildInput>& inputs,
                            const OptixAccelBuildOptions& options);
    void initializeSbt();
    void launch(Params params);
};
}
