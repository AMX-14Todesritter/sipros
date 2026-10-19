#pragma once
// The ordinary executable contains neither NVTX calls nor a runtime switch.
#ifdef MVH_MATCHING_NVTX
#include <nvtx3/nvToolsExt.h>
#endif
namespace mvh_matching {
struct ProfileRange {
    explicit ProfileRange(const char* name) {
#ifdef MVH_MATCHING_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~ProfileRange() {
#ifdef MVH_MATCHING_NVTX
        nvtxRangePop();
#endif
    }
    ProfileRange(const ProfileRange&)=delete;
    ProfileRange& operator=(const ProfileRange&)=delete;
};
}
