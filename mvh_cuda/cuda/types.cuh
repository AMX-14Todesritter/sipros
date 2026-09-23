#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace mvh_cuda {
constexpr int MaxClasses=8, MaxLength=128, MaxText=512, TopN=50;
inline void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Buffer {
    T *p=nullptr; size_t n=0;
    explicit Buffer(size_t count):n(count){if(n) check(cudaMalloc(&p,n*sizeof(T)));}
    explicit Buffer(const std::vector<T>&v):Buffer(v.size()){if(n) check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    Buffer(const Buffer&)=delete; Buffer&operator=(const Buffer&)=delete;
    ~Buffer(){if(p)cudaFree(p);}
    void read(std::vector<T>&v){v.resize(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));}
};
inline void synced(){check(cudaGetLastError());check(cudaDeviceSynchronize());}
struct Config {
    int classes,minClassCount,maxPeaks,minMatched,minLength,smart,bIon,yIon;
    double tic,multiplier,fragmentTolerance,parentTolerance,mzLow,mzHigh,water,proton,yWater;
    double mass[256];
    double missingMass;
};
struct RawScan { uint64_t offset; int count,charge; double parentMass,parentMz; };
struct PrepResult { int count,skip,totalBins,counts[MaxClasses+1]; double sum,max; };
struct Scan {
    uint64_t peakOffset,hubOffset,candidateOffset;
    int peaks,lowest,highest,candidates,skip,topCount,counts[MaxClasses+1],totalBins;
    double lower,upper;
};
// Candidate order is stable within each scan. Peptide and precursor IDs refer
// to this batch and the immutable sorted precursor table, respectively.
struct Candidate { int peptideId, precursorId, scanId, charge; };
struct PeptideInput { uint64_t text; int sequenceId; };
struct Precursor { double mass; int scanId, charge; };
struct MassWindow { double lower, upper; };
struct MassRange { int first, last; };
struct ScanCounts {
    unsigned long long calls=0, successes=0, predicted=0, matched=0;
    int topCount=0, error=0;
};
// Explicit names keep host restoration and device decisions in agreement.
enum ResultStatus { ResultSkipped = 0, ResultMerged = 1, ResultInsufficient = 2, ResultScored = 3, ResultAccepted = 4 };
struct CudaEvent {
    cudaEvent_t event{};
    CudaEvent() { check(cudaEventCreate(&event)); }
    ~CudaEvent() { cudaEventDestroy(event); }
    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;
};
struct Result { double score; int status,predicted,matched; }; // 1 merged, 2 insufficient, 3 scored; negative error
struct ScoringEvent { Candidate candidate; Result result; };
struct Top { double score; int sequenceId; };
struct Rule { int fromLen,toLen; char from[MaxText],to[MaxText]; };
__device__ inline bool alpha(char c){return (c>='A'&&c<='Z')||(c>='a'&&c<='z');}
}
