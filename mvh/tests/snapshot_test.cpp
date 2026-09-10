#include "mvh/snapshot.h"
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <functional>
#include <fstream>

void rejected(const std::function<void()> &call) {
    try { call(); } catch (const std::exception &) { return; }
    throw std::runtime_error("Invalid snapshot was accepted");
}
int main(int argc, char **argv) {
    try {
        if (argc != 2 || !std::filesystem::create_directory(argv[1])) throw std::runtime_error("Pass a new test directory");
        const auto path = (std::filesystem::path(argv[1])/"test.mvh").string();
        mvh::Snapshot s;
        s.buildIdentity = mvh::buildIdentity(); s.configText = "config"; s.inputFile = "not-required.ft2"; s.inputSuffix = "ft2";
        s.minObservedMz = 100; s.maxObservedMz = 200; s.maxScanMass = 1002; s.maxPrecursorCharge = 2; s.intensityClassCount = 3;
        mvh::ScanInput scan; scan.id = 99; scan.parentCharge = 2; scan.parentMz = 501; scan.parentNeutralMass = 1000;
        scan.mzLower = 100; scan.mzUpper = 200; scan.totalPeakBins = 5000;
        scan.peaks = {100.25, 100.75, 102.0}; scan.classes = {1, 2, 3}; scan.intensityCounts = {1, 1, 1, 4997};
        scan.lowestMass = 100; scan.highestMass = 102; scan.massHub = {0, 2, -1, -1, 2, 3};
        s.scans.push_back(scan); s.precursors = {{1000, 2, 0}, {1000, 2, 0}}; // keep ties/duplicates
        mvh::writeSnapshot(path, s);
        auto copy = mvh::readSnapshot(path, "config");
        if (copy.scans[0].peaks != scan.peaks || copy.scans[0].massHub != scan.massHub || copy.precursors.size() != 2)
            throw std::runtime_error("Roundtrip mismatch");
        rejected([&]{ mvh::readSnapshot(path, "different"); });
        rejected([&]{ mvh::writeSnapshot(path, s); });
        auto bad = s; bad.precursors[0].scan = 10; rejected([&]{ mvh::validate(bad); });
        bad = s; bad.precursors[1].mass = 999; rejected([&]{ mvh::validate(bad); });
        bad = s; bad.scans[0].massHub[1] = 1; rejected([&]{ mvh::validate(bad); });
        bad = s; bad.scans[0].peaks[1] = 99; rejected([&]{ mvh::validate(bad); });
        bad = s; bad.scans[0].classes[0] = 0; rejected([&]{ mvh::validate(bad); });
        bad = s; bad.scans[0].intensityCounts[0] = 2; rejected([&]{ mvh::validate(bad); });
        bad = s; bad.scans[0].peaks[0] = std::numeric_limits<double>::quiet_NaN(); rejected([&]{ mvh::validate(bad); });
        bad = s; bad.scans[0].skip = true; rejected([&]{ mvh::validate(bad); });
        std::fstream f(path, std::ios::binary|std::ios::in|std::ios::out); f.seekp(10); f.put('X'); f.close();
        rejected([&]{ mvh::readSnapshot(path, "config"); });
        std::cout << "PASS: snapshot roundtrip, duplicate precursor order, config/overwrite/corruption and structure rejection\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
