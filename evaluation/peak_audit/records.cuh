// Included inside mvh_cuda namespace in an isolated diagnostic source overlay.
struct PeakAuditExample {
    int candidate,scanIndex,queryIndex,expectedPeak,actualPeak,expectedClass,actualClass;
    double mz,tolerance,expectedMz,actualMz;
};
struct PeakAuditState {
    unsigned long long shards[4096][7]; // queries, oracle hits, misses, identity/class diff, actual hits, actual misses
    unsigned long long exampleCount;
    PeakAuditExample examples[128];
};
