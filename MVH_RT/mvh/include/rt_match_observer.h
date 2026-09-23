#pragma once

#include <functional>
#include <vector>

class MS2Scan;

namespace mvh_rt {

// Installed before search and only read while OpenMP workers are running.
using MatchObserver =
    std::function<void(MS2Scan*, const std::vector<double>&)>;
inline MatchObserver matchObserver;

// Clear captured resource references on both normal and exceptional exits.
struct ObserverScope {
    ObserverScope() = default;
    ~ObserverScope() { matchObserver = {}; }
    ObserverScope(const ObserverScope&) = delete;
    ObserverScope& operator=(const ObserverScope&) = delete;
};

} // namespace mvh_rt
