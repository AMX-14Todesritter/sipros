#pragma once
#include <cstdint>
namespace mvh_flow {
enum Metric {
    Entered, Rejected, DirectCalls, DirectIons, OfferedIons, Queries, Hits,
    ScoredHits, WithHit, WithQuery, Mvh, Invalid, CacheCountCalls, CacheStoreCalls,
    CacheCountIons, CacheStoreIons, MetricCount
};
constexpr int Shards = 4096;
struct Counters { unsigned long long v[MetricCount]; };
}
