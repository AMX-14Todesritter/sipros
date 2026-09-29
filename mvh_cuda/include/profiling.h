#pragma once
#include "flow_host.h"

// Host-only, optional NVTX ranges. Disabled builds evaluate no labels and
// contain no range objects or NVTX calls. No GPU synchronization is added.
#ifdef MVH_ENABLE_PROFILING
#include <nvtx3/nvToolsExt.h>

namespace mvh_profile {
class ScopedRange {
public:
    explicit ScopedRange(const char* label, bool start = true) : label_(label) {
        if (start) resume();
    }
    ~ScopedRange() { end(); }
    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;

    // End/resume allows one range per generation batch, rather than one event
    // per peptide. End nested ranges first: NVTX push/pop is a thread stack.
    void end() {
        if (active_) {
#ifdef MVH_ENABLE_FLOW_COUNTERS
            mvh_flow::time(label_,std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count());
#endif
            nvtxRangePop(); active_ = false;
        }
    }
    void resume() {
        if (!active_) {
#ifdef MVH_ENABLE_FLOW_COUNTERS
            start_=std::chrono::steady_clock::now();
#endif
            nvtxRangePushA(label_); active_ = true;
        }
    }
private:
#ifdef MVH_ENABLE_FLOW_COUNTERS
    std::chrono::steady_clock::time_point start_;
#endif
    const char* label_;
    bool active_ = false;
};
}
#define MVH_PROFILE_JOIN_IMPL(a, b) a##b
#define MVH_PROFILE_JOIN(a, b) MVH_PROFILE_JOIN_IMPL(a, b)
#define MVH_PROFILE_SCOPE(label) \
    ::mvh_profile::ScopedRange MVH_PROFILE_JOIN(mvhProfileRange_, __LINE__)(label)
#define MVH_PROFILE_BEGIN(variable, label) ::mvh_profile::ScopedRange variable(label)
#define MVH_PROFILE_DEFER(variable, label) ::mvh_profile::ScopedRange variable(label, false)
#define MVH_PROFILE_END(variable) variable.end()
#define MVH_PROFILE_RESUME(variable) variable.resume()
#else
#define MVH_PROFILE_SCOPE(label) ((void)0)
#define MVH_PROFILE_BEGIN(variable, label) ((void)0)
#define MVH_PROFILE_DEFER(variable, label) ((void)0)
#define MVH_PROFILE_END(variable) ((void)0)
#define MVH_PROFILE_RESUME(variable) ((void)0)
#endif
