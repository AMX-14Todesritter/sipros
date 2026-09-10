#include "mvh/session.h"
#include "ms2scanvector.h"
#include <stdexcept>
#include <unordered_map>

namespace mvh {
std::mutex SearchSession::sessionMutex_;
namespace {
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
}
SearchSession::SearchSession(const std::string &configPath, const std::string &outputDirectory)
    : lock_(sessionMutex_, std::try_to_lock), configPath_(configPath), outputDirectory_(outputDirectory) {
    require(lock_.owns_lock(), "Only one MVH SearchSession may be alive per process");
    require(ProNovoConfig::getSearchType() == "Regular", "Snapshot search requires Search_Type = Regular");
    configText_ = readText(configPath);
}
SearchSession::~SearchSession() {
    // Search normally destroys this table itself; also handles prepare-only use.
    MVH::destroyLnTable();
}
Preparation SearchSession::prepareSpectra(const std::string &input) {
    require(!scans_, "Session input is already prepared");
    Preparation t;
    inputFile_ = input;
    scans_.reset(new MS2ScanVector(input, outputDirectory_, configPath_, true));
    double begin = omp_get_wtime();
    ProNovoConfig::minObservedMz = std::numeric_limits<double>::max();
    ProNovoConfig::maxObservedMz = 0; ProNovoConfig::dMaxMS2ScanMass = 0; ProNovoConfig::iMaxPercusorCharge = 0;
    require(scans_->loadMassData(), "Could not load spectra");
    t.spectrumLoadSeconds = omp_get_wtime()-begin;
    require(!scans_->vpAllMS2Scans.empty() && !scans_->vpPrecursorMasses.empty(), "No searchable scans/precursors were loaded");
    begin = omp_get_wtime();
    scans_->preProcessAllMs2Mvh();
    // Every search starts from the same state and rebuilds its own lnTable.
    MVH::destroyLnTable();
    minMz_ = ProNovoConfig::minObservedMz; maxMz_ = ProNovoConfig::maxObservedMz;
    maxMass_ = ProNovoConfig::dMaxMS2ScanMass; maxCharge_ = ProNovoConfig::iMaxPercusorCharge;
    suffix_ = ProNovoConfig::getSetFileNameSuffix();
    for (const auto *scan : scans_->vpAllMS2Scans) maxBins_ = std::max(maxBins_, scan->totalPeakBins);
    ready_ = true;
    t.preprocessSeconds = omp_get_wtime()-begin;
    return t;
}
Snapshot SearchSession::capture() const {
    require(ready_, "Session is not prepared");
    Snapshot s;
    s.buildIdentity = buildIdentity(); s.configText = configText_; s.inputFile = inputFile_; s.inputSuffix = suffix_;
    s.minObservedMz = minMz_; s.maxObservedMz = maxMz_; s.maxScanMass = maxMass_;
    s.maxPrecursorCharge = maxCharge_; s.intensityClassCount = ProNovoConfig::NumIntensityClasses;
    std::unordered_map<const MS2Scan *, std::uint64_t> ids;
    for (const auto *scan : scans_->vpAllMS2Scans) {
        ids.emplace(scan, s.scans.size());
        require(scan->pPeakList && scan->intenClassCounts, "Missing preprocessed peak data");
        ScanInput v;
        v.id = scan->iScanId; v.parentScanId = scan->iParentScanID; v.parentCharge = scan->iParentChargeState;
        v.parentMz = scan->dParentMZ; v.parentNeutralMass = scan->dParentNeutralMass; v.parentMass = scan->dParentMass;
        v.highRes1 = scan->isMS1HighRes; v.highRes2 = scan->isMS2HighRes; v.skip = scan->bSkip;
        v.retentionTime = scan->sRTime; v.scanType = scan->sScanType;
        v.parentCharges.assign(scan->iParentChargeStates.begin(), scan->iParentChargeStates.end()); v.parentMzs = scan->dParentMZs;
        v.mzLower = scan->mzLowerBound; v.mzUpper = scan->mzUpperBound;
        v.sumIntensity = scan->dSumIntensity; v.maxIntensity = scan->dMaxIntensity;
        v.totalPeakBins = scan->totalPeakBins;
        v.intensityCounts.assign(scan->intenClassCounts->begin(), scan->intenClassCounts->end());
        const auto &peaks = *scan->pPeakList;
        v.peaks = peaks.pPeaks; v.classes.assign(peaks.pClasses.begin(), peaks.pClasses.end());
        // Empty legacy PeakList leaves its bucket scalars uninitialized; canonical
        // zeros represent the empty index without reading those scalars.
        if (!v.peaks.empty()) {
            v.lowestMass = peaks.iLowestMass; v.highestMass = peaks.iHighestMass;
            v.massHub.assign(peaks.pMassHub.begin(), peaks.pMassHub.end());
        }
        s.scans.push_back(std::move(v));
    }
    require(scans_->vpPrecursorMasses.size() == scans_->vAllPrecursorMassChargeMS2ScanPtrTuples.size(), "Precursor index size mismatch");
    for (std::size_t i = 0; i < scans_->vpPrecursorMasses.size(); ++i) {
        const auto &p = scans_->vAllPrecursorMassChargeMS2ScanPtrTuples[i];
        require(scans_->vpPrecursorMasses[i] == std::get<0>(p), "Precursor query array mismatch");
        s.precursors.push_back({std::get<0>(p), std::get<1>(p), ids.at(std::get<2>(p))});
    }
    return s;
}
double SearchSession::saveSnapshot(const std::string &path) const {
    double begin = omp_get_wtime(); writeSnapshot(path, capture()); return omp_get_wtime()-begin;
}
void SearchSession::restore(const Snapshot &s) {
    require(s.intensityClassCount == ProNovoConfig::NumIntensityClasses, "Intensity configuration mismatch");
    inputFile_ = s.inputFile; suffix_ = s.inputSuffix;
    minMz_ = s.minObservedMz; maxMz_ = s.maxObservedMz; maxMass_ = s.maxScanMass; maxCharge_ = s.maxPrecursorCharge;
    scans_.reset(new MS2ScanVector(inputFile_, outputDirectory_, configPath_, true));
    for (const auto &v : s.scans) {
        auto scan = std::make_unique<MS2Scan>();
        scan->sFT2Filename = inputFile_; scan->iScanId = v.id; scan->iParentScanID = v.parentScanId; scan->iParentChargeState = v.parentCharge;
        scan->dParentMZ = v.parentMz; scan->dParentNeutralMass = v.parentNeutralMass; scan->dParentMass = v.parentMass;
        scan->isMS1HighRes = v.highRes1; scan->isMS2HighRes = v.highRes2; scan->bSkip = v.skip;
        scan->sRTime = v.retentionTime; scan->sScanType = v.scanType;
        scan->iParentChargeStates.assign(v.parentCharges.begin(), v.parentCharges.end()); scan->dParentMZs = v.parentMzs;
        scan->mzLowerBound = v.mzLower; scan->mzUpperBound = v.mzUpper;
        scan->dSumIntensity = v.sumIntensity; scan->dMaxIntensity = v.maxIntensity;
        scan->totalPeakBins = v.totalPeakBins;
        scan->intenClassCounts = new std::vector<int>(v.intensityCounts.begin(), v.intensityCounts.end());
        std::map<double, char> empty;
        scan->pPeakList = new PeakList(&empty);
        auto &peaks = *scan->pPeakList;
        peaks.pPeaks = v.peaks; peaks.pClasses.assign(v.classes.begin(), v.classes.end());
        peaks.pMassHub.assign(v.massHub.begin(), v.massHub.end());
        peaks.iPeakSize = v.peaks.size(); peaks.iLowestMass = v.lowestMass; peaks.iHighestMass = v.highestMass;
        peaks.iMassHubSize = v.massHub.size(); peaks.iMassHubPairSizeMinusOne = v.peaks.empty() ? 0 : v.highestMass-v.lowestMass;
        maxBins_ = std::max(maxBins_, v.totalPeakBins);
        scans_->vpAllMS2Scans.push_back(scan.get()); scan.release();
    }
    for (const auto &p : s.precursors) {
        scans_->vpPrecursorMasses.push_back(p.mass);
        scans_->vAllPrecursorMassChargeMS2ScanPtrTuples.emplace_back(p.mass, p.charge, scans_->vpAllMS2Scans.at(p.scan));
    }
    ready_ = true;
}
Preparation SearchSession::loadSnapshot(const std::string &path) {
    require(!scans_, "Session input is already prepared");
    Preparation t;
    double begin = omp_get_wtime();
    restore(readSnapshot(path, configText_));
    t.snapshotLoadSeconds = omp_get_wtime()-begin;
    return t;
}
SearchResult SearchSession::search(const std::string &outputPath) {
    require(ready_, "Session is not prepared");
    require(!std::filesystem::exists(outputPath), "Search result already exists");
    SearchResult result;
    double begin = omp_get_wtime();
    ProNovoConfig::minObservedMz = minMz_; ProNovoConfig::maxObservedMz = maxMz_;
    ProNovoConfig::dMaxMS2ScanMass = maxMass_; ProNovoConfig::iMaxPercusorCharge = maxCharge_;
    ProNovoConfig::getSetFileNameSuffix() = suffix_;
    ProNovoConfig::dMaxPeptideMass = 0; PeptideUnit::iNumScores = 0;
    for (auto *scan : scans_->vpAllMS2Scans) {
        require(scan->vMassChargePeptidePtrTuples.empty(), "Unfinished candidate associations; session cannot be reused");
        for (auto *peptide : scan->vpWeightSumTopPeptides) delete peptide;
        scan->vpWeightSumTopPeptides.clear();
        scan->vdWeightSumAllScores.clear(); scan->iNumPeptideAssigned = 0;
    }
    require(scans_->_ppdAAforward.empty() && scans_->_ppdAAreverse.empty() && scans_->psequenceIonMasses.empty() && scans_->pSeqs.empty(), "Unfinished MVH workspace");
    MVH::initialLnTable(maxBins_);
    result.statePrepareSeconds = omp_get_wtime()-begin;
    begin = omp_get_wtime();
    // The entire production function, unchanged, including FASTA, PTM and cleanup.
    // A failed search is not retryable: ownership may be partially transferred.
    ready_ = false;
    scans_->searchDatabaseMvh();
    ready_ = true;
    result.searchSeconds = omp_get_wtime()-begin;
    result.scans = scans_->vpAllMS2Scans.size(); result.precursors = scans_->vpPrecursorMasses.size();
    for (const auto *scan : scans_->vpAllMS2Scans) {
        result.skipped += scan->bSkip ? 1 : 0; result.retained += scan->vpWeightSumTopPeptides.size();
    }
    begin = omp_get_wtime(); scans_->writeOutputMvh(outputPath); result.exportSeconds = omp_get_wtime()-begin;
    return result;
}
}
