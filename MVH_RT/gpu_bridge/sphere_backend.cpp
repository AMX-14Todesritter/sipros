#include "profiling.h"
#include "backends.h"
#include "custom_geometry.h"
#include "scene_resources.h"
#include <cmath>
#include <iostream>
#include <limits>

namespace mvh_rt_gpu::sphere_backend {
namespace {
struct SphereState {
    SceneResources scene;
    std::unique_ptr<DeviceBuffer<float3>> centers;
    std::unique_ptr<DeviceBuffer<float>> radius;
    std::unique_ptr<DeviceBuffer<uint64_t>> offsets;
    std::unique_ptr<DeviceBuffer<unsigned>> peakIndices;
    double fragmentTolerance = 0;
    int classCount = 0;
    float rayOriginY = 0;
    float rayTmax = 0;
};
std::unique_ptr<SphereState> state;

void validateConfiguration(const mvh_cuda::Config& config) {
    MVH_PROFILE_SCOPE("mvh/rt/sphere/validateConfiguration");
    const float radius = static_cast<float>(config.fragmentTolerance);
    if (!std::isfinite(config.fragmentTolerance) || radius <= 0 || radius >= 0.5f)
        throw std::runtime_error("Sphere RT requires a radius in (0, 0.5) for integer-column separation and ray origin z=0.5");
    if (config.classes < 1 || config.classes > mvh_cuda::MaxClasses)
        throw std::runtime_error("Sphere RT class count is outside the supported range");
}
}

void reset() { MVH_PROFILE_SCOPE("mvh/rt/sphere/reset"); state.reset(); }

void prepare(const std::vector<mvh_cuda::Scan>& scans, const double* peaks,
             const int* classes, size_t peakCount, const mvh_cuda::Config& config) {
    MVH_PROFILE_SCOPE("mvh/rt/sphere/prepare");
    validateConfiguration(config);
    if (peakCount && (!peaks || !classes))
        throw std::runtime_error("Sphere RT requires peak masses and classes");
    if (state) {
        state->scene.requireSameLayout(scans);
        if (state->fragmentTolerance != config.fragmentTolerance || state->classCount != config.classes)
            throw std::runtime_error("Sphere RT configuration changed without reset");
        return;
    }

    const auto start = SetupClock::now();
    auto next = std::make_unique<SphereState>();
    next->scene.layout = scans;
    next->fragmentTolerance = config.fragmentTolerance;
    next->classCount = config.classes;
    initializeOptix(next->scene.objects);
    createPipeline(next->scene.objects, MVH_GPU_RT_CUSTOM_PTX_PATH, false, PrimitiveKind::Sphere);

    const float radius = static_cast<float>(config.fragmentTolerance);
    const double pipelineSeconds = setupSeconds(start);
    MVH_PROFILE_BEGIN(geometryRange, "mvh/rt/sphere/geometry_and_wait");
    const auto geometryStart = SetupClock::now();
    const auto offsets = sphereCenterOffsets(peaks, peakCount, config.fragmentTolerance);
    const size_t centerCount = offsets.back();
    next->offsets = std::make_unique<DeviceBuffer<uint64_t>>(offsets);
    next->centers = std::make_unique<DeviceBuffer<float3>>(centerCount);
    next->peakIndices = std::make_unique<DeviceBuffer<unsigned>>(centerCount);
    next->radius = std::make_unique<DeviceBuffer<float>>(std::vector<float>{radius});
    generateSphereCenters(peaks, next->offsets->data, next->centers->data,
                          next->peakIndices->data, peakCount, config.fragmentTolerance);
    next->rayOriginY = 0.5f; // Retained Params ABI field; custom rays use z=0.5.
    next->rayTmax = 1.0f;
    checkCuda(cudaDeviceSynchronize());
    MVH_PROFILE_END(geometryRange);
    const double geometrySeconds = setupSeconds(geometryStart);

    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    unsigned flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    std::vector<OptixBuildInput> inputs(scans.size());
    std::vector<CUdeviceptr> centerAddresses(scans.size());
    // Build inputs reference these host addresses until scene.build completes.
    CUdeviceptr radiusAddress = next->radius->address();
    MVH_PROFILE_BEGIN(inputRange, "mvh/rt/sphere/build_inputs");
    const auto sizingStart = SetupClock::now();
    for (size_t i = 0; i < scans.size(); ++i) {
        const auto& scan = scans[i];
        if (scan.skip || !scan.peaks) continue;
        centerAddresses[i] = next->centers->address() + offsets[scan.peakOffset] * sizeof(float3);
        inputs[i].type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        auto& input = inputs[i].sphereArray;
        input.vertexBuffers = &centerAddresses[i];
        input.vertexStrideInBytes = sizeof(float3);
        const auto count = offsets[scan.peakOffset + scan.peaks] - offsets[scan.peakOffset];
        if (count > std::numeric_limits<unsigned>::max())
            throw std::runtime_error("Sphere count exceeds OptiX scan index capacity");
        input.numVertices = unsigned(count);
        input.radiusBuffers = &radiusAddress;
        input.radiusStrideInBytes = sizeof(float);
        input.singleRadius = 1;
        input.flags = &flags;
        input.numSbtRecords = 1;
    }
    MVH_PROFILE_END(inputRange);
    const double inputSeconds = setupSeconds(sizingStart);
    const auto stats = next->scene.build(inputs, options);
    next->scene.initializeSbt();
    state = std::move(next);
    std::cout << "[RT GPU setup] scans=" << scans.size() << " seconds=" << setupSeconds(start)
              << " geometry=split-mz-spheres active_scans=" << stats.activeScans
              << " pipeline_seconds=" << pipelineSeconds << " geometry_seconds=" << geometrySeconds
              << " sizing_seconds=" << inputSeconds + stats.sizingSeconds
              << " build_seconds=" << stats.buildSeconds << " as_bytes=" << stats.outputBytes
              << " scratch_bytes=" << stats.scratchBytes
              << " geometry_bytes=" << centerCount * (sizeof(float3) + sizeof(unsigned)) + offsets.size() * sizeof(uint64_t) + sizeof(float)
              << " instance_bytes=" << sizeof(OptixInstance) << '\n';
}

void launch(Params params) {
    MVH_PROFILE_SCOPE("mvh/rt/sphere/launch");
    if (!state) throw std::runtime_error("Sphere RT resources not prepared");
    if (params.cfg.fragmentTolerance != state->fragmentTolerance || params.cfg.classes != state->classCount)
        throw std::runtime_error("Sphere RT launch configuration differs from prepared geometry");
    params.sphereOffsets = state->offsets->data;
    params.spherePeakIndices = state->peakIndices->data;
    params.rayOriginY = state->rayOriginY;
    params.rayTmax = state->rayTmax;
    params.instanced = false;
    state->scene.launch(params);
}
}
