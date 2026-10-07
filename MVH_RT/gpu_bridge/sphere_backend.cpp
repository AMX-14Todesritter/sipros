#include "backends.h"
#include "shared_theory.h"
#include "scene_resources.h"
#include <cmath>
#include <iostream>
#include <limits>
namespace mvh_rt_gpu {
namespace { int configuredGroupSize=8, configuredWorkspaceMiB=512; }
namespace sphere_backend {
namespace {
struct SphereState {
    SceneResources scene;
    std::vector<mvh_cuda::Scan> scans;
    std::unique_ptr<GroupedGeometry> geometry;
    std::unique_ptr<DeviceBuffer<float>> radius;
    int groupSize=0;
    double tolerance=0;
};
std::unique_ptr<SphereState> state;
}
void reset(){state.reset();}
void prepare(const std::vector<mvh_cuda::Scan>& scans,const double* peaks,
             const int* classes,size_t peakCount,const mvh_cuda::Config& config,
             const std::vector<mvh_cuda::Precursor>& precursors) {
    const float radius=std::nextafter(float(config.fragmentTolerance)+8*std::numeric_limits<float>::epsilon(),std::numeric_limits<float>::infinity());
    if(!std::isfinite(config.fragmentTolerance)||config.fragmentTolerance<=0||radius>=0.5f)
        throw std::runtime_error("Sphere RT requires a radius in (0,0.5)");
    if(config.classes!=3)throw std::runtime_error("Shared sphere RT requires 3 classes");
    if(peakCount>std::numeric_limits<unsigned>::max())throw std::runtime_error("Peak identity exceeds unsigned capacity");
    if(state) {
        if(state->scans.size()!=scans.size()||state->groupSize!=configuredGroupSize||state->tolerance!=config.fragmentTolerance)
            throw std::runtime_error("Sphere configuration changed without reset");
        for(size_t i=0;i<scans.size();++i)
            if(state->scans[i].peakOffset!=scans[i].peakOffset||state->scans[i].peaks!=scans[i].peaks||state->scans[i].skip!=scans[i].skip)
                throw std::runtime_error("Sphere layout changed without reset");
        return;
    }
    auto next=std::make_unique<SphereState>();next->scans=scans;
    next->groupSize=configuredGroupSize;next->tolerance=config.fragmentTolerance;
    next->geometry=prepareGroupedGeometry(scans,peaks,classes,precursors,configuredGroupSize,config.fragmentTolerance);
    auto &g=*next->geometry;const size_t groups=g.hostOffsets.size()-1;
    initializeOptix(next->scene.objects);
    createPipeline(next->scene.objects,MVH_GPU_RT_CUSTOM_PTX_PATH,false,PrimitiveKind::Sphere,true);
    next->scene.layout.resize(groups);
    next->radius=std::make_unique<DeviceBuffer<float>>(std::vector<float>{radius});
    OptixAccelBuildOptions options{};options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;options.operation=OPTIX_BUILD_OPERATION_BUILD;
    unsigned flags=OPTIX_GEOMETRY_FLAG_NONE;
    std::vector<OptixBuildInput> inputs(groups);std::vector<CUdeviceptr> addresses(groups);
    CUdeviceptr radiusAddress=next->radius->address();
    for(size_t i=0;i<groups;++i) {
        const uint64_t count=g.hostOffsets[i+1]-g.hostOffsets[i];
        if(count>uint64_t(std::numeric_limits<int>::max()))throw std::runtime_error("GAS sphere count exceeds int capacity");
        next->scene.layout[i].peaks=int(count);if(!count)continue;
        addresses[i]=reinterpret_cast<CUdeviceptr>(g.centers.p+g.hostOffsets[i]);
        inputs[i].type=OPTIX_BUILD_INPUT_TYPE_SPHERES;auto &a=inputs[i].sphereArray;
        a.vertexBuffers=&addresses[i];a.vertexStrideInBytes=sizeof(float3);a.numVertices=unsigned(count);
        a.radiusBuffers=&radiusAddress;a.radiusStrideInBytes=sizeof(float);a.singleRadius=1;a.flags=&flags;a.numSbtRecords=1;
    }
    const auto stats=next->scene.build(inputs,options);next->scene.initializeSbt();state=std::move(next);
    std::cout<<"[RT GPU setup] scans="<<scans.size()<<" geometry=precursor-grouped-mixed-class-spheres"
        <<" scan_group_size="<<configuredGroupSize<<" groups="<<groups<<" grouping=min-neutral-mass"
        <<" class_priority=3,2,1 active_gas="<<stats.activeScans<<" as_bytes="<<stats.outputBytes
        <<" scratch_bytes="<<stats.scratchBytes<<" geometry_bytes="<<g.centers.n*20+g.offsets.n*8+g.scanGroups.n*4<<'\n';
}
void launch(Params p) {
    if(!state)throw std::runtime_error("Sphere resources not prepared");
    if(p.cfg.fragmentTolerance!=state->tolerance||p.cfg.classes!=3)throw std::runtime_error("Sphere launch configuration differs");
    if(!p.size)return;
    auto &g=*state->geometry;p.scanGroupSize=state->groupSize;
    p.spherePeakIndices=g.peaks.p;p.sphereScanIndices=g.scans.p;p.groupSphereOffsets=g.offsets.p;
    p.rayTmax=1.0f;p.instanced=false;
    executeShared(p,g.scanGroups.p,int(g.hostOffsets.size()-1),[&](Params q){state->scene.launch(q);});
}
}
void setScanGroupSize(int size) {
    if(size<1)throw std::runtime_error("RT scan group size must be positive");
    if(sphere_backend::state&&size!=configuredGroupSize)throw std::runtime_error("RT group size changed without reset");
    configuredGroupSize=size;
}
int scanGroupSize(){return configuredGroupSize;}
void setWorkspaceMiB(int size){if(size<16)throw std::runtime_error("RT workspace must be at least 16 MiB");configuredWorkspaceMiB=size;}
int workspaceMiB(){return configuredWorkspaceMiB;}
}
