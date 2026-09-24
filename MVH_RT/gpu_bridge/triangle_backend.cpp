#include "backends.h"
#include "geometry.h"
#include "scene_resources.h"
#include <optix_stubs.h>
#include <iostream>

namespace mvh_rt_gpu::triangle_backend {
namespace {
struct TriangleState {
    SceneResources scene;
    bool instanced = false;
    std::unique_ptr<DeviceBuffer<float3>> vertices;
    std::unique_ptr<DeviceBuffer<OptixInstance>> instances;
    std::unique_ptr<DeviceBuffer<unsigned char>> baseGas;
};
std::unique_ptr<TriangleState> state;

OptixBuildInput triangleInput(CUdeviceptr* address, unsigned count, unsigned* flags) {
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexBuffers = address;
    input.triangleArray.numVertices = count;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes = sizeof(float3);
    input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    input.triangleArray.flags = flags;
    input.triangleArray.numSbtRecords = 1;
    return input;
}

void prepareInstances(TriangleState& next, const std::vector<mvh_cuda::Scan>& scans,
                      const mvh_cuda::Scan* deviceScans, const double* peaks, size_t peakCount,
                      const OptixAccelBuildOptions& options, unsigned& flags) {
    unsigned maxId = 0;
    checkOptix(optixDeviceContextGetProperty(next.scene.objects.context,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID, &maxId, sizeof(maxId)));
    for (const auto& scan : scans)
        if (scan.peaks > 0 && unsigned(scan.peaks - 1) > maxId)
            throw std::runtime_error("Scan exceeds OptiX instance ID limit");

    next.vertices = std::make_unique<DeviceBuffer<float3>>(3);
    generateBaseTriangle(next.vertices->data);
    CUdeviceptr address = next.vertices->address();
    auto input = triangleInput(&address, 3, &flags);
    OptixAccelBufferSizes sizes{};
    checkOptix(optixAccelComputeMemoryUsage(next.scene.objects.context, &options, &input, 1, &sizes));
    DeviceBuffer<unsigned char> scratch(sizes.tempSizeInBytes);
    next.baseGas = std::make_unique<DeviceBuffer<unsigned char>>(sizes.outputSizeInBytes);
    OptixTraversableHandle baseHandle = 0;
    checkOptix(optixAccelBuild(next.scene.objects.context, nullptr, &options, &input, 1,
        scratch.address(), scratch.count, next.baseGas->address(), next.baseGas->count,
        &baseHandle, nullptr, 0));
    checkCuda(cudaDeviceSynchronize());
    next.instances = std::make_unique<DeviceBuffer<OptixInstance>>(peakCount);
    generateInstances(deviceScans, int(scans.size()), peaks, next.instances->data, baseHandle);
}
}

void reset() { state.reset(); }

void prepare(const std::vector<mvh_cuda::Scan>& scans, const mvh_cuda::Scan* deviceScans,
             const double* peaks, size_t peakCount, bool instanced) {
    if (state) {
        state->scene.requireSameLayout(scans);
        if (state->instanced != instanced)
            throw std::runtime_error("Triangle RT mode changed without reset");
        return;
    }
    const auto start = SetupClock::now();
    auto next = std::make_unique<TriangleState>();
    next->instanced = instanced;
    next->scene.layout = scans;
    initializeOptix(next->scene.objects);
    createPipeline(next->scene.objects, MVH_GPU_RT_PTX_PATH, instanced);
    const double pipelineSeconds = setupSeconds(start);

    const auto geometryStart = SetupClock::now();
    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    unsigned flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    if (instanced) {
        prepareInstances(*next, scans, deviceScans, peaks, peakCount, options, flags);
    } else {
        next->vertices = std::make_unique<DeviceBuffer<float3>>(3 * peakCount);
        generateVertices(peaks, next->vertices->data, peakCount);
    }
    checkCuda(cudaDeviceSynchronize());
    const double geometrySeconds = setupSeconds(geometryStart);

    const auto sizingStart = SetupClock::now();
    std::vector<OptixBuildInput> inputs(scans.size());
    std::vector<CUdeviceptr> addresses(scans.size());
    for (size_t i = 0; i < scans.size(); ++i) {
        const auto& scan = scans[i];
        if (scan.skip || !scan.peaks) continue;
        if (instanced) {
            inputs[i].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            inputs[i].instanceArray.instances = next->instances->address() + scan.peakOffset * sizeof(OptixInstance);
            inputs[i].instanceArray.numInstances = scan.peaks;
        } else {
            addresses[i] = next->vertices->address() + scan.peakOffset * 3 * sizeof(float3);
            inputs[i] = triangleInput(&addresses[i], scan.peaks * 3, &flags);
        }
    }
    const double inputSeconds = setupSeconds(sizingStart);
    const auto stats = next->scene.build(inputs, options);
    next->scene.initializeSbt();
    state = std::move(next);
    std::cout << "[RT GPU setup] scans=" << scans.size() << " seconds=" << setupSeconds(start)
              << " geometry=" << (instanced ? "shared-gas-instances" : "gpu-triangles")
              << " active_scans=" << stats.activeScans << " pipeline_seconds=" << pipelineSeconds
              << " geometry_seconds=" << geometrySeconds
              << " sizing_seconds=" << inputSeconds + stats.sizingSeconds
              << " build_seconds=" << stats.buildSeconds << " as_bytes=" << stats.outputBytes
              << " scratch_bytes=" << stats.scratchBytes << " geometry_bytes="
              << (instanced ? peakCount * sizeof(OptixInstance) + 3 * sizeof(float3)
                            : peakCount * 3 * sizeof(float3))
              << " instance_bytes=" << sizeof(OptixInstance) << '\n';
}

void launch(Params params) {
    if (!state) throw std::runtime_error("Triangle RT resources not prepared");
    params.instanced = state->instanced;
    state->scene.launch(params);
}
}
