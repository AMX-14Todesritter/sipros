#pragma once
#include <cstdint>

namespace mvh_matching {
// All counters refer to one execution, not warmup/repeat totals.
struct Statistics {
    uint64_t queries=0, hits=0, buckets=0, emptyBuckets=0, peakChecks=0;
    uint64_t emptyScans=0, outsideRanges=0, onlyEmptyBuckets=0;
    uint64_t traces=0, noHandles=0, maxChecks=0;
    // Checked peaks/query: 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, >=65.
    uint64_t checkHistogram[9]{};
};
struct NoopObserver {
    __host__ __device__ void begin() {}
    __host__ __device__ void emptyScan() {}
    __host__ __device__ void outsideRange() {}
    __host__ __device__ void bucket(bool) {}
    __host__ __device__ void checked() {}
    __host__ __device__ void trace() {}
    __host__ __device__ void noHandle() {}
    __host__ __device__ void end(bool) {}
};
struct CountingObserver {
    Statistics stats{};
    uint64_t checks=0;
    bool visited=false, nonempty=false;
    __host__ __device__ void begin() {checks=0;visited=false;nonempty=false;}
    __host__ __device__ void emptyScan() {++stats.emptyScans;}
    __host__ __device__ void outsideRange() {++stats.outsideRanges;}
    __host__ __device__ void bucket(bool empty) {
        ++stats.buckets;stats.emptyBuckets+=empty;visited=true;nonempty|=!empty;
    }
    __host__ __device__ void checked() {++checks;++stats.peakChecks;}
    __host__ __device__ void trace() {++stats.traces;}
    __host__ __device__ void noHandle() {++stats.noHandles;}
    __host__ __device__ void end(bool hit) {
        ++stats.queries;stats.hits+=hit;
        stats.onlyEmptyBuckets+=visited&&!nonempty;
        if(checks>stats.maxChecks)stats.maxChecks=checks;
        int bin=0;
        if(checks==1)bin=1;else if(checks==2)bin=2;else if(checks>2) {
            bin=3;uint64_t limit=4;
            while(checks>limit && bin<8){++bin;limit*=2;}
        }
        ++stats.checkHistogram[bin];
    }
};
}
