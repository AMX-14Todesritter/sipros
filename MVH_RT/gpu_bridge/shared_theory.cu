#include "profiling.h"
#include "shared_theory.h"
#include "sequence_ions.cuh"
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
namespace mvh_rt_gpu {
using mvh_cuda::Buffer;
using mvh_cuda::check;
namespace {
__device__ double lnCombin(int n,int k,const double *table){if(n<0||k<0||n<k)return -1;return (table[n]-table[n-k])-table[k];}
template<class T> T scalar(const T *p) { T v;check(cudaMemcpy(&v,p,sizeof(v),cudaMemcpyDeviceToHost));return v; }
template<class T> void prefix(Buffer<T>& counts,Buffer<T>& offsets) {
    offsets.resize(counts.n);
    thrust::exclusive_scan(thrust::device_pointer_cast(counts.p),thrust::device_pointer_cast(counts.p+counts.n),thrust::device_pointer_cast(offsets.p));
}
__global__ void minMass(const mvh_cuda::Precursor *ps,int n,double *mass) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto p=ps[i];auto *address=reinterpret_cast<unsigned long long*>(mass+p.scanId);
    unsigned long long old=*address,assumed;
    while(p.mass<__longlong_as_double(old)) {
        assumed=old;old=atomicCAS(address,assumed,__double_as_longlong(p.mass));if(old==assumed)break;
    }
}
__global__ void geometryCounts(const mvh_cuda::Scan *scans,const int *order,int n,
                              const double *peaks,const int *classes,double tolerance,
                              uint64_t *counts,int *groups,int k) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    int s=order[i];groups[s]=i/k;const auto scan=scans[s];uint64_t count=0;
    if(!scan.skip)for(int j=0;j<scan.peaks;++j) {
        uint64_t p=scan.peakOffset+j;if(classes[p]<1||classes[p]>3)continue;
        double f=peaks[p]-floor(peaks[p]);count+=1+(f<=tolerance)+(1-f<=tolerance);
    }
    counts[i]=count;
}
__global__ void packGeometry(const mvh_cuda::Scan *scans,const int *order,int n,
                            const double *peaks,const int *classes,double tolerance,
                            const uint64_t *offsets,float3 *centers,unsigned *ids,unsigned *owners) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    int s=order[i];const auto scan=scans[s];uint64_t out=offsets[i];
    if(!scan.skip)for(int j=0;j<scan.peaks;++j) {
        uint64_t p=scan.peakOffset+j;if(classes[p]<1||classes[p]>3)continue;
        double integer=floor(peaks[p]),f=peaks[p]-integer;
        centers[out]=make_float3(float(integer),float(f),0);ids[out]=unsigned(p);owners[out++]=s;
        if(f<=tolerance){centers[out]=make_float3(float(integer-1),float(f+1),0);ids[out]=unsigned(p);owners[out++]=s;}
        if(1-f<=tolerance){centers[out]=make_float3(float(integer+1),float(f-1),0);ids[out]=unsigned(p);owners[out++]=s;}
    }
}
__global__ void groupOffsets(const uint64_t *scanOffsets,int n,int k,uint64_t *groups,int ng) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<=ng)groups[i]=scanOffsets[min(n,i*k)];
}
__global__ void candidateKeys(const mvh_cuda::Candidate *cs,int n,const int *groups,
                              int groupBits,uint64_t *keys,int *indices,int *error) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const auto c=cs[i];
    if(c.peptideId<0||c.charge<0||uint64_t(c.charge)>((uint64_t(1)<<(32-groupBits))-1)) {atomicExch(error,1);return;}
    keys[i]=(uint64_t(c.peptideId)<<32)|(uint64_t(c.charge)<<groupBits)|unsigned(groups[c.scanId]);indices[i]=i;
}
// Extend candidate chunks through the last peptide/charge key. Thus a theory
// key is generated in exactly one chunk in the batch, even across many groups.
__global__ void chunkEnd(const uint64_t *keys,int n,int tentative,int bits,int *end) {
    uint64_t key=keys[tentative-1]>>bits;int lo=tentative,hi=n;
    while(lo<hi){int mid=lo+(hi-lo)/2;if((keys[mid]>>bits)<=key)lo=mid+1;else hi=mid;}*end=lo;
}
__global__ void chunkFlags(const uint64_t *keys,int n,int bits,int *tasks,int *theories) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    tasks[i]=i==0||keys[i]!=keys[i-1];theories[i]=i==0||(keys[i]>>bits)!=(keys[i-1]>>bits);
}
__global__ void makeTasks(const uint64_t *keys,int n,int bits,const int *taskIds,const int *theoryIds,
                         SharedTask *tasks,int *ends,SharedTask *theories) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    uint64_t key=keys[i];int t=taskIds[i],q=theoryIds[i];
    int peptide=int(key>>32),charge=int((key&0xffffffffu)>>bits),group=int(key&((uint64_t(1)<<bits)-1));
    if(i==0||key!=keys[i-1])tasks[t]={peptide,charge,group,i,0,q};
    if(i==n-1||key!=keys[i+1])ends[t]=i+1;
    if(i==0||(key>>bits)!=(keys[i-1]>>bits))theories[q]={peptide,charge,0,0,0,q};
}
struct Count {uint64_t n=0;__device__ void add(double){++n;}};
struct Store {double *out;uint64_t n=0;__device__ void add(double mz){out[n++]=mz;}};
__global__ void countTheory(Params p,const SharedTask *keys,int n,uint64_t *counts,int *valid) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const auto t=keys[i];
    if(p.chargeStride&&t.charge<p.chargeStride) {
        uint64_t key=uint64_t(t.peptideId)*p.chargeStride+t.charge;
        valid[i]=p.ionValid[key]>0?1:-1;counts[i]=valid[i]>0?p.ionOffsets[key+1]-p.ionOffsets[key]:0;return;
    }
    Count sink;bool ok=mvh_cuda::CalculateSequenceIons(p.texts+p.peptides[t.peptideId].text,t.charge,p.cfg,sink);
    valid[i]=ok?1:-1;counts[i]=ok?sink.n:0;
}
__global__ void generateTheory(Params p,const SharedTask *keys,int n,const uint64_t *offsets,const int *valid,double *ions) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||valid[i]<1)return;const auto t=keys[i];
    if(p.chargeStride&&t.charge<p.chargeStride) {
        uint64_t key=uint64_t(t.peptideId)*p.chargeStride+t.charge;
        for(uint64_t j=p.ionOffsets[key];j<p.ionOffsets[key+1];++j)ions[offsets[i]+j-p.ionOffsets[key]]=p.cachedIons[j];return;
    }
    Store sink{ions+offsets[i]};mvh_cuda::CalculateSequenceIons(p.texts+p.peptides[t.peptideId].text,t.charge,p.cfg,sink);
}
__global__ void finishTasks(SharedTask *tasks,const int *ends,int n,const int *indices,
                            const mvh_cuda::Candidate *cs,const uint64_t *ions,
                            uint64_t *slots,uint64_t *matrix,int *scanIds) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;auto &t=tasks[i];t.count=ends[i]-t.first;
    int count=0;int *list=scanIds+t.first;
    for(int j=t.first;j<ends[i];++j) {
        int scan=cs[indices[j]].scanId,lo=0,hi=count;
        while(lo<hi){int mid=lo+(hi-lo)/2;if(list[mid]<scan)lo=mid+1;else hi=mid;}
        if(lo<count&&list[lo]==scan)continue;
        for(int pos=count;pos>lo;--pos)list[pos]=list[pos-1];list[lo]=scan;++count;
    }
    slots[i]=count;matrix[i]=uint64_t(count)*(ions[t.theoryId+1]-ions[t.theoryId]);
}
__global__ void fitTasks(const uint64_t *hits,const uint64_t *matrix,int first,int n,uint64_t budget,int *end) {
    int lo=first,hi=n;
    while(lo<hi) {
        int mid=lo+(hi-lo+1)/2;
        uint64_t bytes=(hits[mid]-hits[first])*sizeof(CollectedHit)+(matrix[mid]-matrix[first])*sizeof(unsigned);
        if(bytes<=budget)lo=mid;else hi=mid-1;
    }
    *end=lo;
}
__global__ void checkCounts(const uint64_t *counts,const uint64_t *offsets,int n,int *error) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&counts[i]!=offsets[i+1]-offsets[i])atomicExch(error,1);
}
__device__ int findScan(const int *scans,int n,int s) {
    int lo=0,hi=n;while(lo<hi){int mid=lo+(hi-lo)/2;if(scans[mid]<s)lo=mid+1;else hi=mid;}return lo<n&&scans[lo]==s?lo:-1;
}
// One CUDA thread owns each task's complete hit list and matrix. Original peak
// IDs make repeated any-hit invocations and split copies idempotent. No atomics
// or arrival-order decisions participate in choosing the winner.
__global__ void reduceHits(Params p,const uint64_t *slotCounts,const int *scanIds,
                           const uint64_t *matrixOffsets,uint64_t matrixBase,unsigned *best) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=p.size)return;const auto t=p.sharedTasks[i];
    const int *scans=scanIds+t.first;int ns=int(slotCounts[i]);
    uint64_t begin=p.sharedIonOffsets[t.theoryId],ni=p.sharedIonOffsets[t.theoryId+1]-begin;
    unsigned *winners=best+matrixOffsets[i]-matrixBase;
    for(uint64_t j=0;j<ni*ns;++j)winners[j]=~0u;
    for(uint64_t j=p.hitOffsets[i]-p.hitBase;j<p.hitOffsets[i+1]-p.hitBase;++j) {
        const auto h=p.hits[j];int slot=findScan(scans,ns,h.scan);if(slot<0||h.ion>=ni)continue;
        const auto scan=p.scans[h.scan];double mz=p.sharedIons[begin+h.ion];
        if(scan.skip||mz<scan.lower||mz>scan.upper)continue;
        int cls=p.classes[h.peak];double distance=fabs(mz-p.peaks[h.peak]);
        if(cls<1||cls>3||!(distance<p.cfg.fragmentTolerance))continue;
        unsigned &old=winners[uint64_t(h.ion)*ns+slot];
        if(old==~0u||cls>p.classes[old]||
           (cls==p.classes[old]&&(distance<fabs(mz-p.peaks[old])||
            (distance==fabs(mz-p.peaks[old])&&h.peak<old))))old=h.peak;
    }

    for(int s=0;s<ns;++s) {
        const auto scan=p.scans[scans[s]];mvh_cuda::Result result{};
        if(!scan.skip) {
            if(p.sharedIonValid[t.theoryId]<1)result.status=-1;
            else {
                int keys[mvh_cuda::MaxClasses+1]={};result.status=mvh_cuda::ResultInsufficient;
                for(uint64_t j=0;j<ni;++j) {
                    double mz=p.sharedIons[begin+j];if(mz<scan.lower||mz>scan.upper)continue;
                    ++result.predicted;unsigned peak=winners[j*ns+s];
                    if(peak==~0u)++keys[p.cfg.classes];else{++result.matched;++keys[p.classes[peak]-1];}
                }
                if(result.matched&&result.matched>=p.cfg.minMatched) {
                    double value=0;for(int c=0;c<=p.cfg.classes;++c)value+=lnCombin(scan.counts[c],keys[c],p.lnTable);
                    value-=lnCombin(scan.totalBins,result.predicted,p.lnTable);
                    result.score=-value;result.status=mvh_cuda::ResultScored;
                }
            }
        }
        for(int pos=t.first;pos<t.first+t.count;++pos)
            if(p.candidates[p.sharedCandidateIndices[pos]].scanId==scans[s]) {
                const int candidate=p.sharedCandidateIndices[pos];p.results[candidate]=result;
                if(p.selectedPeaks)for(uint64_t j=0;j<ni&&j<unsigned(p.selectedPeakStride);++j)
                    p.selectedPeaks[uint64_t(candidate)*p.selectedPeakStride+j]=winners[j*ns+s];
            }
    }
}
}
std::unique_ptr<GroupedGeometry> prepareGroupedGeometry(const std::vector<mvh_cuda::Scan>& hs,
    const double *peaks,const int *classes,const std::vector<mvh_cuda::Precursor>& precursors,int k,double tolerance) {
    MVH_PROFILE_SCOPE("mvh/rt/shared/geometry_pack");
    const int n=int(hs.size());if(hs.size()>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("Too many scans");
    auto g=std::make_unique<GroupedGeometry>(n);const int ng=n?1+(n-1)/k:0;
    Buffer<mvh_cuda::Scan> scans(hs);Buffer<double> masses(n);Buffer<int> order(n);
    // Synthetic API fixtures must provide explicit precursor masses too.
    if(n&&!precursors.size())throw std::runtime_error("Precursor-grouped BVH requires precursor metadata");
    for(const auto &p:precursors)if(p.scanId<0||p.scanId>=n||!std::isfinite(p.mass)||p.mass<0)throw std::runtime_error("Invalid precursor grouping metadata");
    std::vector<double> initial(n,std::numeric_limits<double>::infinity());masses.upload(initial);
    Buffer<mvh_cuda::Precursor> ps(precursors);
    if(ps.n)minMass<<<(ps.n+255)/256,256>>>(ps.p,int(ps.n),masses.p);
    thrust::sequence(thrust::device_pointer_cast(order.p),thrust::device_pointer_cast(order.p+n));
    thrust::stable_sort_by_key(thrust::device_pointer_cast(masses.p),thrust::device_pointer_cast(masses.p+n),thrust::device_pointer_cast(order.p));
    Buffer<uint64_t> counts(n+1),offsets(n+1);check(cudaMemset(counts.p,0,counts.n*8));
    if(n)geometryCounts<<<(n+127)/128,128>>>(scans.p,order.p,n,peaks,classes,tolerance,counts.p,g->scanGroups.p,k);
    prefix(counts,offsets);uint64_t total=scalar(offsets.p+n);
    g->centers.resize(total);g->peaks.resize(total);g->scans.resize(total);g->offsets.resize(ng+1);
    if(n)packGeometry<<<(n+127)/128,128>>>(scans.p,order.p,n,peaks,classes,tolerance,offsets.p,g->centers.p,g->peaks.p,g->scans.p);
    groupOffsets<<<(ng+256)/256,256>>>(offsets.p,n,k,g->offsets.p,ng);
    mvh_cuda::synced();g->offsets.read(g->hostOffsets);return g;
}
void executeShared(Params p,const int *scanGroups,int groupCount,const std::function<void(Params)>& trace) {
    MVH_PROFILE_SCOPE("mvh/rt/shared/execute");
    const int n=p.size;int bits=0;while((uint64_t(1)<<bits)<unsigned(groupCount))++bits;
    if(bits>31)throw std::runtime_error("Too many BVH groups for key encoding");
    const uint64_t budget=uint64_t(workspaceMiB())*1024*1024;
    const auto started=std::chrono::steady_clock::now();
    Buffer<uint64_t> sortedKeys(n);Buffer<int> sortedIndices(n),error(1);
    {
        MVH_PROFILE_SCOPE("mvh/rt/shared/candidate_radix_sort");
        Buffer<uint64_t> keys(n);Buffer<int> indices(n);check(cudaMemset(error.p,0,4));
        candidateKeys<<<(n+255)/256,256>>>(p.candidates,n,scanGroups,bits,keys.p,indices.p,error.p);
        if(scalar(error.p))throw std::runtime_error("Candidate key cannot be encoded");
        size_t bytes=0;check(cub::DeviceRadixSort::SortPairs(nullptr,bytes,keys.p,sortedKeys.p,indices.p,sortedIndices.p,n));
        Buffer<unsigned char> scratch(bytes);
        check(cub::DeviceRadixSort::SortPairs(scratch.p,bytes,keys.p,sortedKeys.p,indices.p,sortedIndices.p,n));
        mvh_cuda::synced();
    }
    const auto sortedAt=std::chrono::steady_clock::now();
    uint64_t totalTasks=0,totalKeys=0,totalIons=0,totalHits=0,maxTemp=0;int launches=0;
    double theorySeconds=0,traceSeconds=0,reduceSeconds=0;
    Buffer<int> endpoint(1);
    for(int first=0;first<n;) {
        // A conservative candidate tile bounds task metadata; the final key is
        // extended so no peptide/charge is generated twice across tiles.
        int tile=65536,end=0;bool prepared=false;
        std::unique_ptr<Buffer<int>> taskIds,theoryIds,ends;
        std::unique_ptr<Buffer<SharedTask>> tasks,keys;
        std::unique_ptr<Buffer<uint64_t>> ionOffsets;
        std::unique_ptr<Buffer<int>> valid;
        int nt=0,nk=0,cn=0;uint64_t ionCount=0;
        MVH_PROFILE_BEGIN(theoryRange,"mvh/rt/shared/theory_and_tasks");
        auto theoryStarted=std::chrono::steady_clock::now();
        while(!prepared) {
            int tentative=int(std::min(int64_t(n),int64_t(first)+tile));
            chunkEnd<<<1,1>>>(sortedKeys.p,n,tentative,bits,endpoint.p);end=scalar(endpoint.p);cn=end-first;
            Buffer<int> tf(cn),qf(cn);taskIds=std::make_unique<Buffer<int>>(cn);theoryIds=std::make_unique<Buffer<int>>(cn);
            chunkFlags<<<(cn+255)/256,256>>>(sortedKeys.p+first,cn,bits,tf.p,qf.p);
            // Inclusive scan minus one gives the run ID for every candidate.
            thrust::inclusive_scan(thrust::device_pointer_cast(tf.p),thrust::device_pointer_cast(tf.p+cn),thrust::device_pointer_cast(taskIds->p));
            thrust::inclusive_scan(thrust::device_pointer_cast(qf.p),thrust::device_pointer_cast(qf.p+cn),thrust::device_pointer_cast(theoryIds->p));
            nt=scalar(taskIds->p+cn-1);nk=scalar(theoryIds->p+cn-1);
            // Reuse flag storage for zero-based IDs via a small CUDA transform.
            thrust::transform(thrust::device_pointer_cast(taskIds->p),thrust::device_pointer_cast(taskIds->p+cn),thrust::device_pointer_cast(taskIds->p),thrust::placeholders::_1-1);
            thrust::transform(thrust::device_pointer_cast(theoryIds->p),thrust::device_pointer_cast(theoryIds->p+cn),thrust::device_pointer_cast(theoryIds->p),thrust::placeholders::_1-1);
            tasks=std::make_unique<Buffer<SharedTask>>(nt);keys=std::make_unique<Buffer<SharedTask>>(nk);ends=std::make_unique<Buffer<int>>(nt);
            makeTasks<<<(cn+255)/256,256>>>(sortedKeys.p+first,cn,bits,taskIds->p,theoryIds->p,tasks->p,ends->p,keys->p);
            Buffer<uint64_t> ic(nk+1);check(cudaMemset(ic.p,0,ic.n*8));valid=std::make_unique<Buffer<int>>(nk);
            countTheory<<<(nk+127)/128,128>>>(p,keys->p,nk,ic.p,valid->p);
            ionOffsets=std::make_unique<Buffer<uint64_t>>(nk+1);prefix(ic,*ionOffsets);ionCount=scalar(ionOffsets->p+nk);
            uint64_t meta=uint64_t(cn)*16+uint64_t(nt)*64+uint64_t(nk)*40;
            if(ionCount*8+meta>budget/2) {
                if(tile<=1)throw std::runtime_error("Single peptide/charge theory exceeds RT workspace budget");tile=std::max(1,tile/2);
            } else prepared=true;
        }
        taskIds.reset();theoryIds.reset();
        Buffer<double> ions(ionCount);
        generateTheory<<<(nk+127)/128,128>>>(p,keys->p,nk,ionOffsets->p,valid->p,ions.p);
        p.sharedIonOffsets=ionOffsets->p;p.sharedIonValid=valid->p;p.sharedIons=ions.p;
        p.sharedCandidateIndices=sortedIndices.p+first;
        Buffer<uint64_t> slotCounts(nt+1),matrixCounts(nt+1),matrixOffsets(nt+1);
        check(cudaMemset(slotCounts.p,0,slotCounts.n*8));check(cudaMemset(matrixCounts.p,0,matrixCounts.n*8));
        Buffer<int> taskScanIds(cn);
        finishTasks<<<(nt+127)/128,128>>>(tasks->p,ends->p,nt,p.sharedCandidateIndices,p.candidates,ionOffsets->p,slotCounts.p,matrixCounts.p,taskScanIds.p);
        prefix(matrixCounts,matrixOffsets);
        keys.reset();ends.reset();mvh_cuda::synced();
        MVH_PROFILE_END(theoryRange);
        theorySeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-theoryStarted).count();
        Buffer<uint64_t> hitCounts(nt+1),hitOffsets(nt+1);check(cudaMemset(hitCounts.p,0,hitCounts.n*8));
        p.size=nt;p.sharedTasks=tasks->p;p.collectMode=0;p.hitCounts=hitCounts.p;
        auto at=std::chrono::steady_clock::now();
        MVH_PROFILE_BEGIN(countRange,"mvh/rt/shared/count_hits");trace(p);mvh_cuda::synced();prefix(hitCounts,hitOffsets);
        MVH_PROFILE_END(countRange);
        uint64_t hits=scalar(hitOffsets.p+nt);traceSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-at).count();
        const uint64_t persistent=ionCount*8+uint64_t(nk)*12+uint64_t(nt)*72+uint64_t(cn)*4;
        if(persistent>=budget)throw std::runtime_error("RT chunk metadata exceeds workspace");
        for(int a=0;a<nt;) {
            MVH_PROFILE_BEGIN(allocationRange,"mvh/rt/shared/fit_and_allocate_hits");
            fitTasks<<<1,1>>>(hitOffsets.p,matrixOffsets.p,a,nt,budget-persistent,endpoint.p);int b=scalar(endpoint.p);
            if(b==a)throw std::runtime_error("Single shared task hit list exceeds RT workspace; lower K or increase --rt-workspace-mib");
            uint64_t hb=scalar(hitOffsets.p+a),he=scalar(hitOffsets.p+b),mb=scalar(matrixOffsets.p+a),me=scalar(matrixOffsets.p+b);
            Buffer<CollectedHit> collected(he-hb);Buffer<unsigned> best(me-mb);
            Params q=p;q.size=b-a;q.sharedTasks=tasks->p+a;q.hitCounts=hitCounts.p+a;q.hitOffsets=hitOffsets.p+a;
            q.hits=collected.p;q.hitBase=hb;q.collectMode=1;
            MVH_PROFILE_END(allocationRange);
            MVH_PROFILE_BEGIN(storeRange,"mvh/rt/shared/store_hits");
            check(cudaMemset(q.hitCounts,0,q.size*8));at=std::chrono::steady_clock::now();trace(q);mvh_cuda::synced();
            check(cudaMemset(error.p,0,4));checkCounts<<<(q.size+255)/256,256>>>(q.hitCounts,q.hitOffsets,q.size,error.p);
            if(scalar(error.p))throw std::runtime_error("Any-hit count/store disagreement; no partial results accepted");
            MVH_PROFILE_END(storeRange);
            traceSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-at).count();
            at=std::chrono::steady_clock::now();
            MVH_PROFILE_BEGIN(reduceRange,"mvh/rt/shared/reduce_and_mvh");
            reduceHits<<<(q.size+127)/128,128>>>(q,slotCounts.p+a,taskScanIds.p,matrixOffsets.p+a,mb,best.p);mvh_cuda::synced();
            MVH_PROFILE_END(reduceRange);
            reduceSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-at).count();
            maxTemp=std::max(maxTemp,persistent+(he-hb)*sizeof(CollectedHit)+(me-mb)*4);++launches;a=b;
        }
        totalTasks+=nt;totalKeys+=nk;totalIons+=ionCount;totalHits+=hits;
        first=end;
    }
    const auto finished=std::chrono::steady_clock::now();
    std::cout<<"[RT shared tasks] candidates="<<n<<" tasks="<<totalTasks<<" theory_keys="<<totalKeys
        <<" generated_ions="<<totalIons<<" collected_hits="<<totalHits<<" store_launches="<<launches
        <<" workspace_max_bytes="<<maxTemp<<" workspace_budget_bytes="<<budget<<'\n';
    std::cout<<"[RT stages] sort_s="<<std::chrono::duration<double>(sortedAt-started).count()
        <<" theory_and_tasks_s="<<theorySeconds<<" trace_collect_s="<<traceSeconds<<" reduce_score_s="<<reduceSeconds
        <<" total_s="<<std::chrono::duration<double>(finished-started).count()<<'\n';
}
}
