#include "bridge.h"
#include "custom_geometry.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
struct QueryCase {
    std::string name;
    std::vector<double> peaks;
    std::vector<int> classes;
    double query;
    int expectedClass;
};

// Exercise the agreed sphere layout away from ambiguous float boundaries.
// These are framework/priority tests, not a proof of double-precision parity.
void checkQueries(double tolerance, int classCount) {
    using namespace mvh_cuda;
    const double q = 100.0;
    std::vector<QueryCase> cases{
        {"empty scan", {}, {}, q, 0},
        {"exact zero mass difference", {q}, {1}, q, 1},
        {"rounded float zero", {100.075691}, {2}, 100.07569046688, 2},
        {"positive interior", {q}, {1}, q + tolerance * 0.5, 1},
        {"negative interior", {q}, {2}, q - tolerance * 0.5, 2},
        {"clearly outside", {q}, {3}, q + tolerance * 1.5, 0},
        {"reachable higher class over nearest mass", {q, q + tolerance * 0.75}, {1, 3}, q, 3},
        {"outside higher class", {q, q + tolerance * 1.5}, {2, 3}, q, 2},
        {"distinct nearby masses", {q, q + tolerance * 0.25, q + tolerance * 0.5}, {1, 2, 3}, q, 3},
        {"class zero is unscored", {q}, {0}, q, 0},
        {"positive class above zero", {q, q + tolerance * 0.5}, {0, 1}, q, 1},
        {"same class candidates", {q - tolerance * 0.5, q}, {2, 2}, q, 2},
        {"large mass", {8192.0}, {2}, 8192.0, 2},
        {"zero mass", {0.0}, {1}, 0.0, 1},
        {"upper integer copy", {100.0 - tolerance * 0.25}, {2}, 100.0 + tolerance * 0.25, 2},
        {"lower integer copy", {100.0 + tolerance * 0.25}, {3}, 100.0 - tolerance * 0.25, 3},
        {"large mass fractional precision", {8192.123456}, {2}, 8192.123456 + tolerance * 0.5, 2},
        {"integer boundary outside", {100.0 - tolerance * 0.75}, {2}, 100.0 + tolerance * 0.75, 0},
    };
    if (classCount == 4)
        cases.push_back({"class four above ray origin", {q, q + tolerance * 0.75}, {3, 4}, q + tolerance * 0.75, 3});

    if (classCount == 4)
        cases.push_back({"class four alone is unreachable", {q}, {4}, q, 0});

    Config cfg{};
    cfg.classes = classCount; cfg.minMatched = 1; cfg.fragmentTolerance = tolerance;
    std::vector<Scan> hostScans;
    std::vector<double> hostPeaks, hostIons;
    std::vector<int> hostClasses, valid;
    std::vector<uint64_t> offsets;
    std::vector<Candidate> hostCandidates;
    std::vector<PeptideInput> hostPeptides;
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto &test = cases[i];
        Scan scan{};
        scan.peakOffset = hostPeaks.size(); scan.peaks = test.peaks.size();
        scan.lower = 0; scan.upper = 10000; scan.totalBins = 1000;
        int observedCount = 0;
        for (int cls = 0; cls < classCount; ++cls) {
            scan.counts[cls] = 32 << cls;
            observedCount += scan.counts[cls];
        }
        scan.counts[classCount] = scan.totalBins - observedCount;
        hostScans.push_back(scan);
        hostPeaks.insert(hostPeaks.end(), test.peaks.begin(), test.peaks.end());
        hostClasses.insert(hostClasses.end(), test.classes.begin(), test.classes.end());
        hostCandidates.push_back({int(i), 0, int(i), 1});
        hostPeptides.push_back({0, int(i)});
        offsets.push_back(i); offsets.push_back(i);
        valid.push_back(0); valid.push_back(1);
        hostIons.push_back(test.query);
    }
    offsets.push_back(hostIons.size());
    std::vector<double> table(1001);
    for (int i = 1; i <= 1000; ++i) table[i] = table[i - 1] + std::log(double(i));
    Buffer<Scan> scans(hostScans);
    Buffer<double> peaks(hostPeaks), ions(hostIons), lnTable(table);
    Buffer<int> classes(hostClasses), ionValid(valid);
    Buffer<uint64_t> ionOffsets(offsets);
    Buffer<Candidate> candidates(hostCandidates);
    Buffer<PeptideInput> peptides(hostPeptides);
    Buffer<Result> results(cases.size());
    // Verify compact copies retain the exact original peak identity and split layout.
    const auto hostOffsets = mvh_rt_gpu::sphereCenterOffsets(peaks.p, peaks.n, tolerance);
    Buffer<uint64_t> centerOffsets(hostOffsets);
    Buffer<float3> centers(hostOffsets.back());
    Buffer<unsigned> identities(hostOffsets.back());
    mvh_rt_gpu::generateSphereCenters(peaks.p, classes.p, centerOffsets.p, centers.p,
                                      identities.p, peaks.n, tolerance);
    synced();
    std::vector<float3> actualCenters;
    std::vector<unsigned> actualIdentities;
    centers.read(actualCenters); identities.read(actualIdentities);
    for (size_t i = 0; i < hostPeaks.size(); ++i) {
        const double integer = std::floor(hostPeaks[i]);
        const double fraction = hostPeaks[i] - integer;
        std::vector<float3> expected{make_float3(float(integer), float(fraction), 0)};
        if (fraction <= tolerance)
            expected.push_back(make_float3(float(integer - 1), float(fraction + 1), 0));
        if (1 - fraction <= tolerance)
            expected.push_back(make_float3(float(integer + 1), float(fraction - 1), 0));
        if (hostOffsets[i + 1] - hostOffsets[i] != expected.size())
            throw std::runtime_error("Wrong boundary copy count");
        for (size_t j = 0; j < expected.size(); ++j) {
            const auto k = hostOffsets[i] + j;
            if (actualCenters[k].x != expected[j].x || actualCenters[k].y != expected[j].y ||
                actualCenters[k].z != float(hostClasses[i]) || actualIdentities[k] != i)
                throw std::runtime_error("Wrong split layout or peak identity");
        }
    }
    mvh_rt_gpu::Params params{};
    params.scans = scans.p; params.candidates = candidates.p; params.peptides = peptides.p;
    params.peaks = peaks.p; params.classes = classes.p; params.lnTable = lnTable.p;
    params.results = results.p; params.cfg = cfg; params.cachedIons = ions.p;
    params.ionOffsets = ionOffsets.p; params.ionValid = ionValid.p;
    params.size = cases.size(); params.chargeStride = 2;
    for (int repeat = 0; repeat < 2; ++repeat) {
        mvh_rt_gpu::prepare(hostScans, scans.p, peaks.p, classes.p, peaks.n,
                            mvh_rt_gpu::GeometryKind::Spheres, cfg);
        mvh_rt_gpu::launch(params);
        synced();
        std::vector<Result> actual;
        results.read(actual);
        for (size_t i = 0; i < cases.size(); ++i) {
            const int cls = cases[i].expectedClass;
            const auto &result = actual[i];
            if (result.predicted != 1 || result.matched != int(cls > 0) ||
                result.status != (cls ? ResultScored : ResultInsufficient))
                throw std::runtime_error("Scoring eligibility mismatch: " + cases[i].name);
            if (cls) {
                const int count = hostScans[i].counts[cls - 1];
                const double expected = (table[1000] - table[999]) - (table[count] - table[count - 1]);
                if (std::abs(result.score - expected) > 1e-12)
                    throw std::runtime_error("Wrong class-height MVH score: " + cases[i].name);
            }
        }
    }
    bool rejected = false;
    try {
        auto changed = cfg;
        changed.fragmentTolerance *= 2;
        mvh_rt_gpu::prepare(hostScans, scans.p, peaks.p, classes.p, peaks.n,
                            mvh_rt_gpu::GeometryKind::Spheres, changed);
    } catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("Stale sphere radius was reused");
    rejected = false;
    params.cfg.fragmentTolerance *= 2;
    try { mvh_rt_gpu::launch(params); }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("Launch accepted mismatched tolerance");
    rejected = false;
    params.cfg = cfg;
    params.cfg.classes += 1;
    try { mvh_rt_gpu::launch(params); }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("Launch accepted a changed class range");
    rejected = false;
    try {
        mvh_rt_gpu::prepare(hostScans, scans.p, peaks.p, classes.p, peaks.n,
                            mvh_rt_gpu::GeometryKind::Spheres, params.cfg);
    } catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("GAS reused with a changed class range");
    mvh_rt_gpu::reset();
    for (double invalidRadius : {0.0, 0.5, 1.0, std::numeric_limits<double>::quiet_NaN()}) {
        auto invalid = cfg;
        invalid.fragmentTolerance = invalidRadius;
        rejected = false;
        try {
            mvh_rt_gpu::prepare(hostScans, scans.p, peaks.p, classes.p, peaks.n,
                                mvh_rt_gpu::GeometryKind::Spheres, invalid);
        } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("Invalid radius was accepted");
    }
    std::cout << "PASS: split sphere layout/class height/boundary copies/zero class/reuse, queries="
              << cases.size() << " classes=" << classCount << " radius=" << tolerance << '\n';
}
}

int main() {
    try {
        for (int classCount : {3, 4})
            for (double tolerance : {0.125, 0.01}) checkQueries(tolerance, classCount);
        return 0;
    } catch (const std::exception &error) {
        mvh_rt_gpu::reset();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
