#pragma once
#include "types.cuh"
#include <cmath>
#include "diagnostics.cuh"

// Matching has no dependency on peptide generation, intensity classes or MVH.
// Peak IDs are local to a scan. -1 denotes a miss; class-zero peaks are hits.
namespace mvh_matching {
using mvh_cuda::Scan;
constexpr int NoPeak = -1;
struct Query { int scanId; double mz; };
// bins[0..classes-1] are scored intensity classes; bins[classes] is
// misses plus class-zero hits, exactly the scorer's accumulation convention.
struct CandidateSummary {
    uint32_t peakHits=0;
    uint32_t bins[mvh_cuda::MaxClasses+1]{};
    __host__ __device__ void add(int peak,const Scan& scan,const int* classes,int count) {
        const int cls=peak<0?0:classes[scan.peakOffset+peak];
        peakHits+=peak>=0;
        if(cls>0)++bins[cls-1];else ++bins[count];
    }
};
static_assert(sizeof(CandidateSummary)==40);
template<class Observer>
__host__ __device__ inline int nearestPeakObserved(double mz, double tolerance, const Scan &scan,
                        const double *peaks, const short *hub, Observer& observer) {
    observer.begin();
    if (!scan.peaks) {observer.emptyScan();observer.end(false);return NoPeak;}

    const int lower = int(mz - tolerance);
    const int upper = int(mz + tolerance);
    if (upper < scan.lowest || lower > scan.highest) {observer.outsideRange();observer.end(false);return NoPeak;}

    const int firstBucket = lower >= scan.lowest ? lower - scan.lowest : 0;
    const int lastBucket = upper <= scan.highest ? upper - scan.lowest
                                                 : scan.highest - scan.lowest;
    int bestPeak = NoPeak;
    double bestDistance = tolerance;
    for (int bucket = firstBucket; bucket <= lastBucket; ++bucket) {
        const int first = hub[scan.hubOffset + 2 * bucket];
        const int last = hub[scan.hubOffset + 2 * bucket + 1];
        observer.bucket(first == -1);
        if (first == -1) continue;

        for (int i = first; i < last; ++i) {
            observer.checked();
            const auto peakIndex = scan.peakOffset + i;

            const double distance = fabs(mz - peaks[peakIndex]);
            // Bucket membership is only a coarse filter, not a tolerance test.
            if (!(distance < tolerance)) continue;

            if (distance < bestDistance) {
                bestPeak = i;
                bestDistance = distance;
            }
        }
    }
    observer.end(bestPeak != NoPeak);
    return bestPeak;
}
__host__ __device__ inline int nearestPeak(double mz, double tolerance, const Scan& scan,
                        const double* peaks, const short* hub) {
    NoopObserver observer;
    return nearestPeakObserved(mz,tolerance,scan,peaks,hub,observer);
}

// Deliberately exhaustive control: no hub reads, range rejection or early exit.
template<class Observer>
__host__ __device__ inline int nearestPeakLinearObserved(double mz,double tolerance,
    const Scan& scan,const double* peaks,Observer& observer) {
    observer.begin();
    if(!scan.peaks){observer.emptyScan();observer.end(false);return NoPeak;}
    int best=NoPeak;double distance=tolerance;
    for(int i=0;i<scan.peaks;++i){
        observer.checked();
        const double d=fabs(mz-peaks[scan.peakOffset+i]);
        if(d<distance){best=i;distance=d;}
    }
    observer.end(best!=NoPeak);return best;
}

// Device arrays must be resident and remain alive until the stream completes.
// Index construction/upload and synchronization are the caller's responsibility.
void launchCuda(const Scan* scans, const double* peaks, const short* hub,
                const Query* queries, int count, double tolerance, int* results,
                cudaStream_t stream = nullptr);
// One thread per captured candidate, preserving its fragment-query order.
void launchCudaGrouped(const Scan* scans, const double* peaks, const short* hub,
                       const mvh_cuda::Candidate* groups, const uint64_t* offsets,
                       const double* mz, uint64_t base, int count, double tolerance,
                       int* results, cudaStream_t stream = nullptr, Statistics* statistics = nullptr);
void launchCudaGroupedSummary(const Scan* scans,const double* peaks,const short* hub,
    const int* classes,int classCount,const mvh_cuda::Candidate* groups,
    const uint64_t* offsets,const double* mz,uint64_t base,int count,double tolerance,
    CandidateSummary* results,Statistics* statistics=nullptr);
}
