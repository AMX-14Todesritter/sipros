#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mvh {
constexpr std::uint32_t snapshotVersion = 1;
struct ScanInput {
    std::int32_t id = 0, parentScanId = 0, parentCharge = 0;
    double parentMz = 0, parentNeutralMass = 0, parentMass = 0;
    bool highRes1 = true, highRes2 = true, skip = false;
    std::string retentionTime, scanType;
    std::vector<std::int32_t> parentCharges;
    std::vector<double> parentMzs;
    double mzLower = 0, mzUpper = 0, sumIntensity = 0, maxIntensity = 0;
    std::int32_t totalPeakBins = 0;
    std::vector<std::int32_t> intensityCounts;
    std::vector<double> peaks;
    std::vector<std::uint8_t> classes;
    std::int32_t lowestMass = 0, highestMass = 0;
    std::vector<std::int32_t> massHub;
};
struct PrecursorInput {
    double mass = 0;
    std::int32_t charge = 0;
    std::uint64_t scan = 0; // index into scans, never an instrument scan number
};
struct Snapshot {
    std::string buildIdentity, configText, inputFile, inputSuffix;
    double minObservedMz = 0, maxObservedMz = 0, maxScanMass = 0;
    std::int32_t maxPrecursorCharge = 0, intensityClassCount = 0;
    std::vector<ScanInput> scans; // original order, including skipped scans
    std::vector<PrecursorInput> precursors; // original sorted order, including ties
};
std::string buildIdentity();
std::string readText(const std::string &path);
void validate(const Snapshot &snapshot);
void writeSnapshot(const std::string &path, const Snapshot &snapshot);
Snapshot readSnapshot(const std::string &path, const std::string &configText);
}
