#pragma once

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
        if (active_) { nvtxRangePop(); active_ = false; }
    }
    void resume() {
        if (!active_) { nvtxRangePushA(label_); active_ = true; }
    }
private:
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
