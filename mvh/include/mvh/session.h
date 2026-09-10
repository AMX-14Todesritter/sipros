#pragma once
#include "mvh/snapshot.h"
#include <memory>
#include <mutex>

class MS2ScanVector;
namespace mvh {
struct Preparation {
    double spectrumLoadSeconds = 0, preprocessSeconds = 0, snapshotLoadSeconds = 0;
};
struct SearchResult {
    double statePrepareSeconds = 0, searchSeconds = 0, exportSeconds = 0;
    std::uint64_t scans = 0, precursors = 0, skipped = 0, retained = 0;
};
// One live session per process: legacy configuration and MVH tables are global.
// The caller loads a Regular configuration before constructing this session and
// must not change it until destruction. FASTA may be overridden before search.
class SearchSession {
public:
    SearchSession(const std::string &configPath, const std::string &outputDirectory);
    ~SearchSession();
    SearchSession(const SearchSession &) = delete;
    SearchSession &operator=(const SearchSession &) = delete;
    Preparation prepareSpectra(const std::string &input);
    Preparation loadSnapshot(const std::string &path);
    double saveSnapshot(const std::string &path) const;
    SearchResult search(const std::string &outputPath);
    const std::string &inputFile() const { return inputFile_; }
private:
    Snapshot capture() const;
    void restore(const Snapshot &snapshot);
    static std::mutex sessionMutex_;
    std::unique_lock<std::mutex> lock_;
    std::unique_ptr<MS2ScanVector> scans_;
    std::string configPath_, configText_, outputDirectory_, inputFile_, suffix_;
    double minMz_ = 0, maxMz_ = 0, maxMass_ = 0;
    int maxCharge_ = 0, maxBins_ = 0;
    bool ready_ = false;
};
}
