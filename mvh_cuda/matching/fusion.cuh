#pragma once
#include "matching.cuh"
#include "../cuda/ion_math.cuh"
namespace mvh_matching {
// Identical output layout in all three modes, including padding-independent validation.
struct FusionOutput { CandidateSummary summary{}; mvh_cuda::Result result{}; };
// Sphere intersections use float geometry. Reject boundary false positives
// using the same strict double-precision predicate as CPU/CUDA matching.
__host__ __device__ inline int validateRtPeak(int peak,double mz,const Scan& scan,
 const double* peaks,double tolerance){
 if(peak<0)return NoPeak;
 return fabs(mz-peaks[scan.peakOffset+peak])<tolerance?peak:NoPeak;
}
struct BinLookup {
 const Scan& scan;const double* peaks;const short* hub;double tolerance;
 __host__ __device__ int operator()(double mz)const{return nearestPeak(mz,tolerance,scan,peaks,hub);}
};
template<class Lookup>struct FusionSink {
 const Scan& scan;const mvh_cuda::Config& cfg;const int* classes;Lookup lookup;
 CandidateSummary summary{};int predicted=0,matched=0;
 __host__ __device__ void add(double mz){
  if(mz<scan.lower || mz>scan.upper)return;
  const int peak=lookup(mz);summary.add(peak,scan,classes,cfg.classes);
  ++predicted;if(peak>=0 && classes[scan.peakOffset+peak]>0)++matched;
 }
};
template<int Mode,class Lookup>
__host__ __device__ FusionOutput evaluateFusion(const Scan& scan,
 const mvh_cuda::Candidate& candidate,const mvh_cuda::PeptideInput* peptides,
 const char* texts,const int* classes,const double* table,const mvh_cuda::Config& cfg,
 const double* mz,uint64_t first,uint64_t last,unsigned char validity,Lookup lookup){
 FusionOutput out{};if(scan.skip)return out;
 FusionSink<Lookup> sink{scan,cfg,classes,lookup};bool valid;
 if constexpr(Mode==0){
  valid=validity==1;
  for(uint64_t j=first;j<last;++j)sink.add(mz[j]);
 }else valid=mvh_cuda::CalculateSequenceIons(texts+peptides[candidate.peptideId].text,candidate.charge,cfg,sink);
 out.summary=sink.summary;
 if(!valid){out.result.status=-1;return out;}
 out.result.predicted=sink.predicted;out.result.matched=sink.matched;
 out.result.status=mvh_cuda::ResultInsufficient;
 if constexpr(Mode==2){
  if(sink.matched && sink.matched>=cfg.minMatched){
   double value=0;
   for(int k=0;k<=cfg.classes;++k)value+=mvh_cuda::lnCombin(scan.counts[k],int(sink.summary.bins[k]),table);
   value-=mvh_cuda::lnCombin(scan.totalBins,sink.predicted,table);
   out.result.score=-value;out.result.status=mvh_cuda::ResultScored;
  }
 }
 return out;
}
}
