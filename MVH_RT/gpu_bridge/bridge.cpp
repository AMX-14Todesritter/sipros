#include "bridge.h"
#include "geometry.h"
#include "rt_support.h"
#include <optix_stubs.h>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace mvh_rt_gpu {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
size_t alignAS(size_t n) {
    return (n+OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT-1)&~size_t(OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT-1);
}
struct State {
    OptixObjects objects;
    std::vector<mvh_cuda::Scan> layout;
    bool instanced=false;
    // Shared allocations replace per-scan allocations. All builds use one stream.
    std::unique_ptr<DeviceBuffer<float3>> vertices;
    std::unique_ptr<DeviceBuffer<OptixInstance>> instances;
    std::unique_ptr<DeviceBuffer<unsigned char>> baseGas, acceleration;
    std::unique_ptr<DeviceBuffer<OptixTraversableHandle>> handles;
    DeviceBuffer<SbtRecord<EmptyData>> raygen{1},miss{1},hit{1};
    DeviceBuffer<Params> parameters{1};
    OptixShaderBindingTable sbt{};
};
std::unique_ptr<State> state;
OptixBuildInput triangles(CUdeviceptr *address,unsigned count,unsigned *flags) {
    OptixBuildInput input{};
    input.type=OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexBuffers=address;
    input.triangleArray.numVertices=count;
    input.triangleArray.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes=sizeof(float3);
    input.triangleArray.indexFormat=OPTIX_INDICES_FORMAT_NONE;
    input.triangleArray.flags=flags;
    input.triangleArray.numSbtRecords=1;
    return input;
}
}
void reset() { state.reset(); }
void prepare(const std::vector<mvh_cuda::Scan>& scans,
             const mvh_cuda::Scan *deviceScans,const double *devicePeaks,
             size_t peakCount,bool instanced) {
    if (state) {
        if (state->layout.size()!=scans.size() || state->instanced!=instanced)
            throw std::runtime_error("RT layout changed without reset");
        for(size_t i=0;i<scans.size();++i)
            if(state->layout[i].peakOffset!=scans[i].peakOffset ||
               state->layout[i].peaks!=scans[i].peaks || state->layout[i].skip!=scans[i].skip)
                throw std::runtime_error("RT peak layout changed without reset");
        return;
    }
    const auto start=Clock::now();
    auto next=std::make_unique<State>();
    next->instanced=instanced;next->layout=scans;
    initializeOptix(next->objects);
    createPipeline(next->objects,MVH_GPU_RT_PTX_PATH,instanced);
    const double pipelineSeconds=elapsed(start);
    const auto geometryStart=Clock::now();
    OptixAccelBuildOptions options{};
    options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation=OPTIX_BUILD_OPERATION_BUILD;
    unsigned flags=OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    OptixTraversableHandle baseHandle=0;
    if(instanced) {
        unsigned maxId=0;
        checkOptix(optixDeviceContextGetProperty(next->objects.context,
            OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID,&maxId,sizeof(maxId)));
        for(const auto &s:scans)
            if(s.peaks>0 && unsigned(s.peaks-1)>maxId)
                throw std::runtime_error("Scan exceeds OptiX instance ID limit");
        next->vertices=std::make_unique<DeviceBuffer<float3>>(3);
        generateBaseTriangle(next->vertices->data);
        CUdeviceptr address=next->vertices->address();
        auto input=triangles(&address,3,&flags);
        OptixAccelBufferSizes sizes{};
        checkOptix(optixAccelComputeMemoryUsage(next->objects.context,&options,&input,1,&sizes));
        DeviceBuffer<unsigned char> scratch(sizes.tempSizeInBytes);
        next->baseGas=std::make_unique<DeviceBuffer<unsigned char>>(sizes.outputSizeInBytes);
        checkOptix(optixAccelBuild(next->objects.context,nullptr,&options,&input,1,
            scratch.address(),scratch.count,next->baseGas->address(),next->baseGas->count,
            &baseHandle,nullptr,0));
        checkCuda(cudaDeviceSynchronize());
        next->instances=std::make_unique<DeviceBuffer<OptixInstance>>(peakCount);
        generateInstances(deviceScans,int(scans.size()),devicePeaks,next->instances->data,baseHandle);
    } else {
        next->vertices=std::make_unique<DeviceBuffer<float3>>(3*peakCount);
        generateVertices(devicePeaks,next->vertices->data,peakCount);
    }
    checkCuda(cudaDeviceSynchronize());
    const double geometrySeconds=elapsed(geometryStart);
    const auto planStart=Clock::now();
    std::vector<OptixBuildInput> inputs(scans.size());
    std::vector<CUdeviceptr> addresses(scans.size());
    std::vector<OptixAccelBufferSizes> sizes(scans.size());
    std::vector<size_t> offsets(scans.size());
    std::vector<OptixTraversableHandle> handles(scans.size());
    size_t outputBytes=0,scratchBytes=0,activeScans=0;
    for(size_t i=0;i<scans.size();++i) {
        const auto &s=scans[i];
        if(s.skip || !s.peaks)continue;
        ++activeScans;
        if(instanced) {
            inputs[i].type=OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            inputs[i].instanceArray.instances=next->instances->address()+s.peakOffset*sizeof(OptixInstance);
            inputs[i].instanceArray.numInstances=s.peaks;
        } else {
            addresses[i]=next->vertices->address()+s.peakOffset*3*sizeof(float3);
            inputs[i]=triangles(&addresses[i],s.peaks*3,&flags);
        }
        checkOptix(optixAccelComputeMemoryUsage(next->objects.context,&options,&inputs[i],1,&sizes[i]));
        offsets[i]=outputBytes;
        outputBytes+=alignAS(sizes[i].outputSizeInBytes);
        scratchBytes=std::max(scratchBytes,sizes[i].tempSizeInBytes);
    }
    const double planSeconds=elapsed(planStart);
    const auto buildStart=Clock::now();
    next->acceleration=std::make_unique<DeviceBuffer<unsigned char>>(outputBytes);
    DeviceBuffer<unsigned char> scratch(scratchBytes);
    // Default-stream ordering allows reuse of scratch without synchronizing every scan.
    for(size_t i=0;i<scans.size();++i) {
        if(scans[i].skip || !scans[i].peaks)continue;
        checkOptix(optixAccelBuild(next->objects.context,nullptr,&options,&inputs[i],1,
            scratch.address(),scratch.count,next->acceleration->address()+offsets[i],
            sizes[i].outputSizeInBytes,&handles[i],nullptr,0));
    }
    checkCuda(cudaDeviceSynchronize());
    const double buildSeconds=elapsed(buildStart);
    next->handles=std::make_unique<DeviceBuffer<OptixTraversableHandle>>(handles);
    SbtRecord<EmptyData> raygen{},miss{},hit{};
    checkOptix(optixSbtRecordPackHeader(next->objects.raygen,&raygen));
    checkOptix(optixSbtRecordPackHeader(next->objects.miss,&miss));
    checkOptix(optixSbtRecordPackHeader(next->objects.hit,&hit));
    checkCuda(cudaMemcpy(next->raygen.data,&raygen,sizeof(raygen),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(next->miss.data,&miss,sizeof(miss),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(next->hit.data,&hit,sizeof(hit),cudaMemcpyHostToDevice));
    next->sbt.raygenRecord=next->raygen.address();
    next->sbt.missRecordBase=next->miss.address();
    next->sbt.missRecordStrideInBytes=sizeof(miss);next->sbt.missRecordCount=1;
    next->sbt.hitgroupRecordBase=next->hit.address();
    next->sbt.hitgroupRecordStrideInBytes=sizeof(hit);next->sbt.hitgroupRecordCount=1;
    state=std::move(next);
    std::cout << "[RT GPU setup] scans=" << scans.size() << " seconds=" << elapsed(start)
              << " geometry=" << (instanced ? "shared-gas-instances" : "gpu-triangles")
              << " active_scans=" << activeScans << " pipeline_seconds=" << pipelineSeconds
              << " geometry_seconds=" << geometrySeconds << " sizing_seconds=" << planSeconds
              << " build_seconds=" << buildSeconds << " as_bytes=" << outputBytes
              << " scratch_bytes=" << scratchBytes << " geometry_bytes="
              << (instanced ? peakCount*sizeof(OptixInstance)+3*sizeof(float3) : peakCount*3*sizeof(float3))
              << " instance_bytes=" << sizeof(OptixInstance) << '\n';
}
void launch(Params p) {
    if(!p.size)return;
    if(!state)throw std::runtime_error("RT resources not prepared");
    p.handles=state->handles->data;p.instanced=state->instanced;
    checkCuda(cudaMemcpy(state->parameters.data,&p,sizeof(p),cudaMemcpyHostToDevice));
    checkOptix(optixLaunch(state->objects.pipeline,nullptr,state->parameters.address(),sizeof(p),
                          &state->sbt,p.size,1,1));
}
}
