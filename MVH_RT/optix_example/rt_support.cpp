#include "rt_support.h"

// Define the OptiX function table in exactly one translation unit.
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void checkOptix(OptixResult result) {
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error("OptiX error code " + std::to_string(int(result)));
}
void logCallback(unsigned int level, const char *tag, const char *message, void *) {
    std::cerr << "[OptiX " << level << ":" << tag << "] " << message << '\n';
}

void initializeOptix(OptixObjects &objects) {
    checkCuda(cudaSetDevice(0));
    checkCuda(cudaFree(nullptr)); // Initialize the CUDA primary context.
    const OptixResult initialized = optixInit();
    if (initialized != OPTIX_SUCCESS)
        throw std::runtime_error("optixInit failed (" + std::to_string(int(initialized)) +
            "). Check that libnvoptix.so.1 from a compatible NVIDIA driver is visible in this container.");
    OptixDeviceContextOptions options{};
    options.logCallbackFunction = logCallback;
    options.logCallbackLevel = 3;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    checkOptix(optixDeviceContextCreate(nullptr, &options, &objects.context));
}

std::vector<ScanPeak> makeScene(
    const std::vector<double>& mzValues,
    const std::vector<char>& classes,
    int scanId)
{
    if (mzValues.size() != classes.size())
        throw std::runtime_error("Peak/class size mismatch");

    std::vector<ScanPeak> peaks;
    peaks.reserve(mzValues.size());

    for (size_t i = 0; i < mzValues.size(); ++i) {
        ScanPeak peak{};
        peak.mz = make_float3(
            static_cast<float>(mzValues[i]), 0.0f, 0.0f);
        peak.scanId = scanId;
        peak.intensityClass = static_cast<int>(classes[i]);

        peaks.push_back(peak);
    }

    return peaks;
}

std::vector<float3> makeVertices(
    const std::vector<ScanPeak>& peaks)
{
    std::vector<float3> vertices;
    vertices.reserve(peaks.size() * 3);

    for (const ScanPeak& peak : peaks) {
        const float3 p = peak.mz;

        // A triangle perpendicular to x, containing p in its interior.
        vertices.push_back(
            make_float3(p.x, p.y - 1.0f, p.z - 1.0f));
        vertices.push_back(
            make_float3(p.x, p.y + 1.0f, p.z - 1.0f));
        vertices.push_back(
            make_float3(p.x, p.y, p.z + 1.0f));
    }

    return vertices;
}

Ray makeRay(double mz, double tolerance)
{
    Ray ray;

    ray.origin = make_float3(static_cast<float>(mz), 0.0f, 0.0f);
    ray.mz = mz;
    ray.tmin = 0.0f;
    ray.tmax = static_cast<float>(tolerance);
    return ray;
}

std::vector<Ray> generateRays(const double* mzValues, size_t count,double tolerance ) {
    std::vector<Ray> rays;
    rays.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        rays.push_back(makeRay(mzValues[i], tolerance));
    }
    return rays;
}



OptixProgramGroup createProgramGroup(OptixDeviceContext context, OptixProgramGroupDesc description) {
    OptixProgramGroupOptions options{};
    OptixProgramGroup group = nullptr;
    char log[4096]{};
    size_t length = sizeof(log);
    const auto result = optixProgramGroupCreate(context, &description, 1, &options, log, &length, &group);
    if (result != OPTIX_SUCCESS) std::cerr << log << '\n';
    checkOptix(result);
    return group;
}

void createPipeline(OptixObjects &objects, const fs::path &ptxPath, bool instanced, PrimitiveKind primitive) {
    const bool spheres = primitive == PrimitiveKind::Sphere;
    std::ifstream stream(ptxPath, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open PTX: " + ptxPath.string());
    const std::string ptx((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    OptixModuleCompileOptions moduleOptions{};
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    OptixPipelineCompileOptions compileOptions{};
    compileOptions.traversableGraphFlags = instanced ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
                                                    : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    compileOptions.numPayloadValues = 2; // Peak index and optional distance; sphere uses the first slot.
    compileOptions.numAttributeValues = 2;
    compileOptions.pipelineLaunchParamsVariableName = "params";
    compileOptions.usesPrimitiveTypeFlags = spheres ? OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE
                                                   : OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    char log[4096]{};
    size_t logSize = sizeof(log);
    auto result = optixModuleCreate(objects.context, &moduleOptions, &compileOptions,
                                    ptx.data(), ptx.size(), log, &logSize, &objects.module);
    if (result != OPTIX_SUCCESS) std::cerr << log << '\n';
    checkOptix(result);

    OptixProgramGroupDesc description{};
    description.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    description.raygen.module = objects.module;
    description.raygen.entryFunctionName = "__raygen__camera";
    objects.raygen = createProgramGroup(objects.context, description);
    description = {};
    description.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    description.miss.module = objects.module;
    description.miss.entryFunctionName = "__miss__background";
    objects.miss = createProgramGroup(objects.context, description);
    description = {};
    description.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    if (spheres) {
        // Built-in spheres use OptiX's intersector, not a user-written IS.
        // Keep these build flags consistent with bridge.cpp's GAS options.
        OptixBuiltinISOptions builtinOptions{};
        builtinOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
        builtinOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        checkOptix(optixBuiltinISModuleGet(objects.context, &moduleOptions,
            &compileOptions, &builtinOptions, &objects.sphereIntersection));
    }
    description.hitgroup.moduleIS = objects.sphereIntersection;
    description.hitgroup.entryFunctionNameIS = nullptr;
    description.hitgroup.moduleCH = objects.module;
    description.hitgroup.entryFunctionNameCH = "__closesthit__record";
    objects.hit = createProgramGroup(objects.context, description);

    OptixProgramGroup groups[] = {objects.raygen, objects.miss, objects.hit};
    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = 1; // No recursive reflection/refraction rays.
    logSize = sizeof(log);
    result = optixPipelineCreate(objects.context, &compileOptions, &linkOptions,
                                 groups, 3, log, &logSize, &objects.pipeline);
    if (result != OPTIX_SUCCESS) std::cerr << log << '\n';
    checkOptix(result);

    // Derive stack sizes from the compiled programs instead of guessing sizes.
    OptixStackSizes sizes{};
    for (const auto group : groups)
        checkOptix(optixUtilAccumulateStackSizes(group, &sizes, objects.pipeline));
    unsigned int traversal = 0, state = 0, continuation = 0;
    checkOptix(optixUtilComputeStackSizes(&sizes, 1, 0, 0, &traversal, &state, &continuation));
    checkOptix(optixPipelineSetStackSize(objects.pipeline, traversal, state, continuation, instanced ? 2 : 1));
}

//BVH share
ScanRtResources::ScanRtResources(
    OptixDeviceContext context,
    const std::vector<double>& mzValues,
    const std::vector<char>& classes,
    int scanId)
    : peaks_(makeScene(mzValues, classes, scanId)),
      devicePeaks_(peaks_),
      deviceVertices_(makeVertices(peaks_))
{
    if (peaks_.empty())
        throw std::runtime_error("Cannot build GAS for an empty scan");

    CUdeviceptr vertexAddress = deviceVertices_.address();

    unsigned int geometryFlags[] = {
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    };

    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    auto& triangles = input.triangleArray;
    triangles.vertexBuffers = &vertexAddress;
    triangles.numVertices =
        static_cast<unsigned int>(deviceVertices_.count);
    triangles.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangles.vertexStrideInBytes = sizeof(float3);
    triangles.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    triangles.flags = geometryFlags;
    triangles.numSbtRecords = 1;

    OptixAccelBuildOptions buildOptions{};
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes{};
    checkOptix(optixAccelComputeMemoryUsage(
        context, &buildOptions, &input, 1, &sizes));

    // Scratch is needed only while building the GAS.
    DeviceBuffer<unsigned char> scratch(sizes.tempSizeInBytes);

    // The GAS buffer must remain alive for all subsequent queries.
    gasMemory_ = std::make_unique<DeviceBuffer<unsigned char>>(
        sizes.outputSizeInBytes);

    checkOptix(optixAccelBuild(
        context,
        nullptr,
        &buildOptions,
        &input,
        1,
        scratch.address(),
        scratch.count,
        gasMemory_->address(),
        gasMemory_->count,
        &gas_,
        nullptr,
        0));

    // Finish the build before the local scratch buffer is released.
    checkCuda(cudaDeviceSynchronize());

    std::cout << "Built GAS: primitives=" << peaks_.size()
              << " bytes=" << gasMemory_->count << '\n';
}

//query RT
std::vector<RayResult> traceRays(
    const OptixObjects& objects,
    const ScanRtResources& scan,
    const std::vector<Ray>& rays)
{
    if (rays.empty())
        return {};

    // Select programs and attach this scan's peak data.
    SbtRecord<EmptyData> raygen{}, miss{};
    SbtRecord<HitGroupData> hit{};

    checkOptix(optixSbtRecordPackHeader(objects.raygen, &raygen));
    checkOptix(optixSbtRecordPackHeader(objects.miss, &miss));
    checkOptix(optixSbtRecordPackHeader(objects.hit, &hit));

    hit.data.peaks = scan.devicePeaks();

    DeviceBuffer<SbtRecord<EmptyData>> deviceRaygen(
        std::vector<SbtRecord<EmptyData>>{raygen});

    DeviceBuffer<SbtRecord<EmptyData>> deviceMiss(
        std::vector<SbtRecord<EmptyData>>{miss});

    DeviceBuffer<SbtRecord<HitGroupData>> deviceHit(
        std::vector<SbtRecord<HitGroupData>>{hit});

    OptixShaderBindingTable sbt{};
    sbt.raygenRecord = deviceRaygen.address();

    sbt.missRecordBase = deviceMiss.address();
    sbt.missRecordStrideInBytes = sizeof(miss);
    sbt.missRecordCount = 1;

    sbt.hitgroupRecordBase = deviceHit.address();
    sbt.hitgroupRecordStrideInBytes = sizeof(hit);
    sbt.hitgroupRecordCount = 1;

    // Per-query buffers; the scan's GAS remains owned by the caller.
    DeviceBuffer<Ray> deviceRays(rays);
    DeviceBuffer<RayResult> results(rays.size());

    LaunchParams params{};
    params.traversable = scan.handle();
    params.peaks = scan.devicePeaks();
    params.rays = deviceRays.data;
    params.results = results.data;
    params.width = static_cast<unsigned int>(rays.size());
    params.height = 1;

    DeviceBuffer<LaunchParams> deviceParams(
        std::vector<LaunchParams>{params});

    checkOptix(optixLaunch(
        objects.pipeline,
        nullptr,
        deviceParams.address(),
        sizeof(params),
        &sbt,
        params.width,
        params.height,
        1));

    // Complete tracing before downloading and releasing query buffers.
    checkCuda(cudaDeviceSynchronize());

    return results.download();
}

OptixObjects::~OptixObjects()
{
    if (pipeline) optixPipelineDestroy(pipeline);
    if (hit) optixProgramGroupDestroy(hit);
    if (miss) optixProgramGroupDestroy(miss);
    if (raygen) optixProgramGroupDestroy(raygen);
    if (sphereIntersection) optixModuleDestroy(sphereIntersection);
    if (module) optixModuleDestroy(module);
    if (context) optixDeviceContextDestroy(context);
}
