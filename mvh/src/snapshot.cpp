#include "mvh/snapshot.h"
#include "mvh/build_identity.h"
#include "zlib.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mvh {
namespace {
void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(std::string("MVH snapshot: ") + message);
}
using Bytes = std::vector<unsigned char>;
struct Writer {
    Bytes bytes;
    void u32(std::uint32_t v) { for (int i = 0; i < 4; ++i) bytes.push_back((v >> (8*i)) & 255); }
    void u64(std::uint64_t v) { for (int i = 0; i < 8; ++i) bytes.push_back((v >> (8*i)) & 255); }
    void integer(std::int32_t v) { u32(static_cast<std::uint32_t>(v)); }
    void number(double v) { std::uint64_t bits; std::memcpy(&bits, &v, 8); u64(bits); }
    void text(const std::string &s) { u64(s.size()); bytes.insert(bytes.end(), s.begin(), s.end()); }
    template<class T, class F> void array(const std::vector<T> &v, F write) {
        u64(v.size()); for (const auto &x : v) write(x);
    }
};
struct Reader {
    const Bytes &bytes;
    std::size_t pos = 0, limit;
    std::size_t count(std::size_t width) {
        auto n = u64();
        check(n <= (limit-pos)/width && n <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()), "invalid array length");
        return static_cast<std::size_t>(n);
    }
    std::uint64_t word(int n) {
        check(static_cast<std::size_t>(n) <= limit-pos, "truncated file");
        std::uint64_t v = 0;
        for (int i = 0; i < n; ++i) v |= std::uint64_t(bytes[pos++]) << (8*i);
        return v;
    }
    std::uint32_t u32() { return static_cast<std::uint32_t>(word(4)); }
    std::uint64_t u64() { return word(8); }
    std::int32_t integer() { auto v = u32(); std::int32_t x; std::memcpy(&x, &v, 4); return x; }
    double number() { auto bits = u64(); double v; std::memcpy(&v, &bits, 8); check(std::isfinite(v), "nonfinite value"); return v; }
    bool boolean() { auto v = u32(); check(v <= 1, "invalid boolean"); return v != 0; }
    std::string text() { auto n = count(1); std::string s(bytes.begin()+pos, bytes.begin()+pos+n); pos += n; return s; }
    template<class T, class F> std::vector<T> array(std::size_t width, F read) {
        auto n = count(width); std::vector<T> v; v.reserve(n);
        for (std::size_t i = 0; i < n; ++i) v.push_back(read());
        return v;
    }
};
std::uint32_t checksum(const Bytes &b, std::size_t n) {
    uLong value = crc32(0L, Z_NULL, 0);
    for (std::size_t offset = 0; offset < n;) {
        auto chunk = static_cast<uInt>(std::min<std::size_t>(n-offset, 1u << 20));
        value = crc32(value, b.data()+offset, chunk); offset += chunk;
    }
    return static_cast<std::uint32_t>(value);
}
}
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "IEEE754 binary64 required");
std::string buildIdentity() { return MVH_BUILD_IDENTITY; }
std::string readText(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    check(bool(f), "cannot open input/config file");
    std::string value((std::istreambuf_iterator<char>(f)), {});
    check(!f.bad(), "input/config read failed"); return value;
}
void validate(const Snapshot &s) {
    check(!s.scans.empty() && !s.precursors.empty(), "no scans or precursors");
    check(s.scans.size() <= std::numeric_limits<int>::max() && s.precursors.size() <= std::numeric_limits<int>::max(), "too many entries");
    check(!s.inputFile.empty() && !s.configText.empty(), "missing identity/configuration");
    check(s.intensityClassCount > 0 && s.intensityClassCount < 127, "invalid intensity class count");
    check(std::isfinite(s.minObservedMz) && std::isfinite(s.maxObservedMz) && s.minObservedMz >= 0 && s.minObservedMz <= s.maxObservedMz, "invalid global mz bounds");
    check(std::isfinite(s.maxScanMass) && s.maxScanMass >= 0 && s.maxPrecursorCharge > 0, "invalid precursor limits");
    for (const auto &scan : s.scans) {
        for (double v : {scan.parentMz, scan.parentNeutralMass, scan.parentMass, scan.mzLower, scan.mzUpper, scan.sumIntensity, scan.maxIntensity})
            check(std::isfinite(v), "nonfinite scan value");
        check(scan.parentCharge > 0 && scan.parentMz >= 0 && scan.parentNeutralMass >= 0, "invalid scan precursor");
        check(scan.parentCharges.size() == scan.parentMzs.size(), "precursor hypothesis lengths differ");
        for (std::size_t i = 0; i < scan.parentMzs.size(); ++i)
            check(std::isfinite(scan.parentMzs[i]) && scan.parentMzs[i] >= 0 && scan.parentCharges[i] > 0, "invalid precursor hypothesis");
        check(scan.mzLower >= 0 && scan.mzUpper >= scan.mzLower && scan.totalPeakBins >= 0, "invalid scan bounds/bins");
        check(scan.peaks.size() == scan.classes.size() && scan.peaks.size() <= 32767, "invalid peak count");
        std::vector<std::int32_t> counts(s.intensityClassCount+1, 0);
        for (std::size_t i = 0; i < scan.peaks.size(); ++i) {
            check(std::isfinite(scan.peaks[i]) && scan.peaks[i] >= 0 && scan.peaks[i] < std::numeric_limits<int>::max()/2, "invalid peak mass");
            check(i == 0 || scan.peaks[i] > scan.peaks[i-1], "peaks not strictly sorted");
            check(scan.classes[i] > 0 && scan.classes[i] <= s.intensityClassCount, "invalid peak class");
            ++counts[scan.classes[i]-1];
        }
        if (scan.skip) {
            check(scan.peaks.empty() && scan.massHub.empty() && scan.intensityCounts.empty() && scan.totalPeakBins == 0, "invalid skipped scan");
        } else {
            check(scan.totalPeakBins >= static_cast<int>(scan.peaks.size()), "negative void count");
            counts.back() = scan.totalPeakBins - scan.peaks.size();
            check(counts == scan.intensityCounts, "intensity counts inconsistent with peaks");
        }
        if (scan.peaks.empty()) {
            check(scan.massHub.empty() && scan.lowestMass == 0 && scan.highestMass == 0, "nonempty index for empty peaks");
        } else {
            const int low = static_cast<int>(scan.peaks.front()), high = static_cast<int>(scan.peaks.back());
            check(scan.lowestMass == low && scan.highestMass == high, "bucket bounds mismatch");
            check(scan.massHub.size() == std::size_t(high-low+1)*2, "bucket length mismatch");
            std::vector<std::int32_t> hub(scan.massHub.size(), -1);
            for (std::size_t i = 0; i < scan.peaks.size(); ++i) {
                auto slot = 2 * (static_cast<int>(scan.peaks[i])-low);
                if (hub[slot] == -1) hub[slot] = static_cast<int>(i);
                hub[slot+1] = static_cast<int>(i+1);
            }
            check(hub == scan.massHub, "bucket index inconsistent with peaks");
        }
    }
    double previous = -1;
    for (const auto &p : s.precursors) {
        check(std::isfinite(p.mass) && p.mass >= 0 && p.mass >= previous, "unsorted/invalid precursor mass");
        check(p.scan < s.scans.size() && p.charge > 0 && p.charge <= s.maxPrecursorCharge, "invalid precursor scan ID/charge");
        check(p.mass <= s.scans[p.scan].parentNeutralMass, "precursor exceeds scan mass");
        previous = p.mass;
    }
}
void writeSnapshot(const std::string &path, const Snapshot &s) {
    validate(s);
    check(s.buildIdentity == buildIdentity(), "build identity mismatch");
    check(!std::filesystem::exists(path), "output already exists");
    Writer w;
    for (char c : std::string("SMVHSP01")) w.bytes.push_back(c);
    w.u32(snapshotVersion); w.text(s.buildIdentity); w.text(s.configText); w.text(s.inputFile); w.text(s.inputSuffix);
    w.number(s.minObservedMz); w.number(s.maxObservedMz); w.number(s.maxScanMass);
    w.integer(s.maxPrecursorCharge); w.integer(s.intensityClassCount); w.u64(s.scans.size());
    for (const auto &v : s.scans) {
        w.integer(v.id); w.integer(v.parentScanId); w.integer(v.parentCharge);
        w.number(v.parentMz); w.number(v.parentNeutralMass); w.number(v.parentMass);
        w.u32(v.highRes1); w.u32(v.highRes2); w.u32(v.skip); w.text(v.retentionTime); w.text(v.scanType);
        w.array(v.parentCharges, [&](auto x){ w.integer(x); }); w.array(v.parentMzs, [&](auto x){ w.number(x); });
        w.number(v.mzLower); w.number(v.mzUpper); w.number(v.sumIntensity); w.number(v.maxIntensity); w.integer(v.totalPeakBins);
        w.array(v.intensityCounts, [&](auto x){ w.integer(x); });
        w.array(v.peaks, [&](auto x){ w.number(x); }); w.array(v.classes, [&](auto x){ w.u32(x); });
        w.integer(v.lowestMass); w.integer(v.highestMass); w.array(v.massHub, [&](auto x){ w.integer(x); });
    }
    w.u64(s.precursors.size());
    for (const auto &v : s.precursors) { w.number(v.mass); w.integer(v.charge); w.u64(v.scan); }
    w.u32(checksum(w.bytes, w.bytes.size()));
    std::ofstream f(path, std::ios::binary);
    f.exceptions(std::ios::badbit | std::ios::failbit);
    f.write(reinterpret_cast<const char *>(w.bytes.data()), w.bytes.size()); f.close();
}
Snapshot readSnapshot(const std::string &path, const std::string &configText) {
    auto size = std::filesystem::file_size(path);
    check(size >= 16 && size <= (std::uint64_t(8) << 30), "invalid file size (limit 8 GiB)");
    Bytes bytes(static_cast<std::size_t>(size));
    std::ifstream f(path, std::ios::binary); f.exceptions(std::ios::badbit | std::ios::failbit);
    f.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    Reader trailer{bytes, bytes.size()-4, bytes.size()};
    check(checksum(bytes, bytes.size()-4) == trailer.u32(), "CRC32 mismatch");
    Reader r{bytes, 0, bytes.size()-4};
    for (char c : std::string("SMVHSP01")) check(r.word(1) == static_cast<unsigned char>(c), "bad magic");
    check(r.u32() == snapshotVersion, "unsupported format version");
    Snapshot s; s.buildIdentity = r.text(); check(s.buildIdentity == buildIdentity(), "incompatible code/build version");
    s.configText = r.text(); check(s.configText == configText, "configuration differs from snapshot");
    s.inputFile = r.text(); s.inputSuffix = r.text();
    s.minObservedMz = r.number(); s.maxObservedMz = r.number(); s.maxScanMass = r.number();
    s.maxPrecursorCharge = r.integer(); s.intensityClassCount = r.integer();
    auto n = r.count(160); s.scans.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        ScanInput v; v.id = r.integer(); v.parentScanId = r.integer(); v.parentCharge = r.integer();
        v.parentMz = r.number(); v.parentNeutralMass = r.number(); v.parentMass = r.number();
        v.highRes1 = r.boolean(); v.highRes2 = r.boolean(); v.skip = r.boolean(); v.retentionTime = r.text(); v.scanType = r.text();
        v.parentCharges = r.array<std::int32_t>(4, [&]{ return r.integer(); }); v.parentMzs = r.array<double>(8, [&]{ return r.number(); });
        v.mzLower = r.number(); v.mzUpper = r.number(); v.sumIntensity = r.number(); v.maxIntensity = r.number(); v.totalPeakBins = r.integer();
        v.intensityCounts = r.array<std::int32_t>(4, [&]{ return r.integer(); }); v.peaks = r.array<double>(8, [&]{ return r.number(); });
        v.classes = r.array<std::uint8_t>(4, [&]{ auto x = r.u32(); check(x <= 255, "invalid class byte"); return x; });
        v.lowestMass = r.integer(); v.highestMass = r.integer(); v.massHub = r.array<std::int32_t>(4, [&]{ return r.integer(); });
        s.scans.push_back(std::move(v));
    }
    n = r.count(20); s.precursors.reserve(n);
    for (std::size_t i = 0; i < n; ++i) { PrecursorInput v; v.mass = r.number(); v.charge = r.integer(); v.scan = r.u64(); s.precursors.push_back(v); }
    check(r.pos == r.limit, "trailing data"); validate(s); return s;
}
}
