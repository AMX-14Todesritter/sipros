#pragma once
#include <cstdint>
namespace mvh_flow {
enum Metric {
    Entered, Rejected, DirectCalls, DirectIons, OfferedIons, Queries, Hits,
    ScoredHits, WithHit, WithQuery, Mvh, Invalid, CacheCountCalls, CacheStoreCalls,
    CacheCountIons, CacheStoreIons, SearchCalls, BucketVisits, PeakChecks,
    RtTraces, RtClosestHits, RtMisses, BackendZeroQueries, InvalidQueries, InvalidHits, MetricCount
};
constexpr int Shards = 4096;
struct Counters { unsigned long long v[MetricCount]; };
}
