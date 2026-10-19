#include "matching.cuh"
#include <cmath>

namespace mvh_matching {
__global__ void queryKernel(const Scan* scans, const double* peaks, const short* hub,
                            const Query* queries, int count, double tolerance, int* results) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Query q = queries[i];
    #ifdef MVH_MATCHING_LINEAR
    NoopObserver observer;
    results[i] = nearestPeakLinearObserved(q.mz,tolerance,scans[q.scanId],peaks,observer);
#else
    results[i] = nearestPeak(q.mz, tolerance, scans[q.scanId], peaks, hub);
#endif
}

void launchCuda(const Scan* scans, const double* peaks, const short* hub,
                const Query* queries, int count, double tolerance, int* results,
                cudaStream_t stream) {
    if (count < 0 || !std::isfinite(tolerance) || tolerance <= 0)
        throw std::invalid_argument("Invalid matching count/tolerance");
    if (!count) return;
    queryKernel<<<(uint64_t(count) + 127) / 128, 128, 0, stream>>>(scans, peaks, hub, queries,
                                                     count, tolerance, results);
    mvh_cuda::check(cudaGetLastError());
}

template<bool Summary>
__device__ void groupedBody(const Scan* scans, const double* peaks, const short* hub,
    const mvh_cuda::Candidate* groups, const uint64_t* offsets, const double* mz,
    uint64_t base, int count, double tolerance, int* results, Statistics* statistics,
    const int* classes,int classCount,CandidateSummary* summaries) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const Scan& scan=scans[groups[i].scanId];
#ifdef MVH_MATCHING_DIAGNOSTICS
    CountingObserver observer;
#else
    NoopObserver observer;
#endif
    CandidateSummary summary{};
    for(uint64_t j=offsets[i]-base;j<offsets[i+1]-base;++j) {
        #ifdef MVH_MATCHING_LINEAR
        const int peak=nearestPeakLinearObserved(mz[j],tolerance,scan,peaks,observer);
#else
        const int peak=nearestPeakObserved(mz[j],tolerance,scan,peaks,hub,observer);
#endif
        if constexpr(Summary)summary.add(peak,scan,classes,classCount);
        else results[j]=peak;
    }
    if constexpr(Summary)summaries[i]=summary;
#ifdef MVH_MATCHING_DIAGNOSTICS
    if(statistics)statistics[i]=observer.stats;
#endif
}

__global__ void groupedKernel(const Scan* scans,const double* peaks,const short* hub,
    const mvh_cuda::Candidate* groups,const uint64_t* offsets,const double* mz,
    uint64_t base,int count,double tolerance,int* results,Statistics* statistics) {
    groupedBody<false>(scans,peaks,hub,groups,offsets,mz,base,count,tolerance,
        results,statistics,nullptr,0,nullptr);
}
__global__ void groupedAggregateKernel(const Scan* scans,const double* peaks,const short* hub,
    const int* classes,int classCount,const mvh_cuda::Candidate* groups,const uint64_t* offsets,
    const double* mz,uint64_t base,int count,double tolerance,CandidateSummary* results,
    Statistics* statistics) {
    groupedBody<true>(scans,peaks,hub,groups,offsets,mz,base,count,tolerance,
        nullptr,statistics,classes,classCount,results);
}
void launchCudaGroupedSummary(const Scan* scans,const double* peaks,const short* hub,
    const int* classes,int classCount,const mvh_cuda::Candidate* groups,
    const uint64_t* offsets,const double* mz,uint64_t base,int count,double tolerance,
    CandidateSummary* results,Statistics* statistics) {
    if(count<0 || classCount<1 || classCount>mvh_cuda::MaxClasses || !std::isfinite(tolerance) || tolerance<=0)
        throw std::invalid_argument("Invalid candidate summary configuration");
    if(!count)return;
    groupedAggregateKernel<<<(uint64_t(count)+127)/128,128>>>(scans,peaks,hub,classes,classCount,
        groups,offsets,mz,base,count,tolerance,results,statistics);
    mvh_cuda::check(cudaGetLastError());
}

void launchCudaGrouped(const Scan* scans, const double* peaks, const short* hub,
    const mvh_cuda::Candidate* groups, const uint64_t* offsets, const double* mz,
    uint64_t base, int count, double tolerance, int* results, cudaStream_t stream, Statistics* statistics) {
    if(count<0 || !std::isfinite(tolerance) || tolerance<=0)
        throw std::invalid_argument("Invalid grouped matching count/tolerance");
    if(!count) return;
    groupedKernel<<<(uint64_t(count)+127)/128,128,0,stream>>>(scans,peaks,hub,
        groups,offsets,mz,base,count,tolerance,results,statistics);
    mvh_cuda::check(cudaGetLastError());
}
}
