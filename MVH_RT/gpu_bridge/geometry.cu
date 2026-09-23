#include "geometry.h"

namespace mvh_rt_gpu {
__device__ void triangle(float3 *v, float x) {
    v[0]=make_float3(x,-1,-1);
    v[1]=make_float3(x,1,-1);
    v[2]=make_float3(x,0,1);
}
__global__ void verticesKernel(const double *peaks,float3 *vertices,size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) triangle(vertices+3*i,float(peaks[i]));
}
__global__ void baseKernel(float3 *vertices) { triangle(vertices,0); }
__global__ void instancesKernel(const mvh_cuda::Scan *scans,const double *peaks,
                                OptixInstance *instances,OptixTraversableHandle base) {
    const auto s=scans[blockIdx.x];
    if(s.skip)return;
    for(int i=threadIdx.x;i<s.peaks;i+=blockDim.x) {
        OptixInstance instance{};
        instance.transform[0]=instance.transform[5]=instance.transform[10]=1.0f;
        instance.transform[3]=float(peaks[s.peakOffset+i]);
        instance.instanceId=i;
        instance.visibilityMask=255;
        instance.flags=OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle=base;
        instances[s.peakOffset+i]=instance;
    }
}
void generateVertices(const double *peaks,float3 *vertices,size_t count) {
    if(count)verticesKernel<<<(count+255)/256,256>>>(peaks,vertices,count);
    mvh_cuda::check(cudaGetLastError());
}
void generateBaseTriangle(float3 *vertices) {
    baseKernel<<<1,1>>>(vertices);mvh_cuda::check(cudaGetLastError());
}
void generateInstances(const mvh_cuda::Scan *scans,int count,const double *peaks,
                       OptixInstance *instances,OptixTraversableHandle base) {
    if(count)instancesKernel<<<count,256>>>(scans,peaks,instances,base);
    mvh_cuda::check(cudaGetLastError());
}
}
