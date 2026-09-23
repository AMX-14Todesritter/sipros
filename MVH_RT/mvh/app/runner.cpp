#include "runner.h"
#include "mvh_scan_vector.h"
#include <iomanip>
#include <stdexcept>

#ifdef MVH_ENABLE_RT
#include "rt_support.h"
#include "rt_match_observer.h"
#include <cmath>
#include <exception>
#include <memory>
#include <mutex>
#include <unordered_map>
#endif

namespace {
std::size_t writePsms(const std::string &path, const std::string &input,
                     const std::vector<MS2Scan *> &scans) {
    std::ofstream out(path);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(17)
        << "input_file\tscan_index\tscan_id\tprecursor_mass\tprecursor_charge\tpeptide\t"
           "original_peptide\tprotein_names\tcalculated_mass\tmvh_score\tmvh_rank\n";
    std::size_t count = 0;
    for (std::size_t i=0; i<scans.size(); ++i) {
        const auto *scan = scans[i];
        for (std::size_t rank=0; rank<scan->vpWeightSumTopPeptides.size(); ++rank) {
            const auto *p = scan->vpWeightSumTopPeptides[rank];
            out << input << '\t' << i << '\t' << scan->iScanId << '\t'
                << p->dMeasuredParentMass << '\t' << p->iMeasuredParentCharge << '\t'
                << p->sIdentifiedPeptide << '\t' << p->sOriginalPeptide << '\t'
                << p->sProteinNames << '\t' << p->dCalculatedParentMass << '\t'
                << p->vdScores[2] << '\t' << rank+1 << '\n';
            ++count;
        }
    }
    out.close();
    return count;
}
}

void mvh_app::run(const std::string &input, const std::string &config,
                  const std::string &fasta, const std::string &output, int threads) {
    omp_set_num_threads(threads);
    const double begin = omp_get_wtime();
    if (!ProNovoConfig::setFilename(config)) throw std::runtime_error("Cannot load config");
    if (ProNovoConfig::getSearchType() != "Regular")
        throw std::runtime_error("Only Search_Type = Regular is supported");
    ProNovoConfig::setFASTAfilename(fasta);
    // Create category/run parents while preserving exclusive creation of the run itself.
    const auto parent = std::filesystem::path(output).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    if (!std::filesystem::create_directory(output))
        throw std::runtime_error("Output directory must not already exist");
    std::filesystem::copy_file(config, std::filesystem::path(output)/"input_config.cfg");
    MvhScanVector spectra(input, output, config, true);
    if (!spectra.loadMassData()) throw std::runtime_error("Cannot load spectra");
    const double loaded = omp_get_wtime();
    const auto &scans = spectra.vpAllMS2Scans;
    if (scans.empty()) throw std::runtime_error("No scans loaded");
    spectra.preProcessAllMs2Mvh();
    const double prepared = omp_get_wtime();

    #ifdef MVH_ENABLE_RT
    // Create one shared OptiX context and pipeline.
    OptixObjects rtObjects;
    initializeOptix(rtObjects);
    createPipeline(rtObjects, MVH_RT_PTX_PATH);

    // Keep the array aligned with spectra.vpAllMS2Scans.
    // Skipped scans have a null resource entry.
    std::vector<std::unique_ptr<ScanRtResources>> rtScans(scans.size());

    std::size_t builtScans = 0;

    for (std::size_t i = 0; i < scans.size(); ++i) {
        const auto* scan = scans[i];

        if (scan->bSkip)
            continue;

        if (!scan->pPeakList)
            throw std::runtime_error("Missing preprocessed peak list");

        const auto& mzValues = scan->pPeakList->pPeaks;
        const auto& classes = scan->pPeakList->pClasses;

        if (mzValues.size() != classes.size())
            throw std::runtime_error("Peak/class size mismatch");

        if (mzValues.empty())
            continue;

        rtScans[i] = std::make_unique<ScanRtResources>(
            rtObjects.context,
            mzValues,
            classes,
            scan->iScanId
        );

        ++builtScans;
    }

    std::cout << "RT resources ready: "
            << builtScans << " scans\n";

    std::unordered_map<MS2Scan*, ScanRtResources*> rtByScan;
    for (std::size_t i = 0; i < scans.size(); ++i) {
        if (rtScans[i]) rtByScan.emplace(scans[i], rtScans[i].get());
    }

    // Serialize GPU submissions in this correctness-only implementation.
    std::mutex rtMutex;
    std::exception_ptr rtFailure;
    std::size_t verifiedCandidates = 0;
    std::size_t verifiedIons = 0;
    // Declared last so captures are cleared before their resources are destroyed.
    mvh_rt::ObserverScope observerScope;
    mvh_rt::matchObserver =
        [&](MS2Scan* scan, const std::vector<double>& ions) {
            std::lock_guard<std::mutex> lock(rtMutex);
            if (rtFailure) return;
            try {
                const auto found = rtByScan.find(scan);
                if (found == rtByScan.end())
                    throw std::runtime_error("Missing RT scan resource");

                // CUDA's current device is per host thread, including OpenMP workers.
                checkCuda(cudaSetDevice(0));
                const double tolerance = ProNovoConfig::getMassAccuracyFragmentIon();
                const auto rays = generateRays(ions.data(), ions.size(), tolerance);
                const auto results = traceRays(rtObjects, *found->second, rays);
                if (results.size() != ions.size())
                    throw std::runtime_error("RT result size mismatch");

                const auto& mzValues = scan->pPeakList->pPeaks;
                const auto& classes = scan->pPeakList->pClasses;
                for (std::size_t j = 0; j < ions.size(); ++j) {
                    const double mz = ions[j];
                    // Out-of-range ions are excluded, not counted as unmatched.
                    if (mz < scan->mzLowerBound || mz > scan->mzUpperBound) continue;

                    int expectedIndex = -1;
                    double bestError = tolerance;
                    for (std::size_t k = 0; k < mzValues.size(); ++k) {
                        const double error = std::abs(mzValues[k] - mz);
                        // Strict tolerance and first-encountered tie semantics.
                        if (error < bestError) {
                            bestError = error;
                            expectedIndex = static_cast<int>(k);
                        }
                    }
                    const char expectedClass = expectedIndex < 0 ? 0 : classes[expectedIndex];
                    const char cpuClass = scan->pPeakList->findNear(mz, tolerance);
                    const bool cpuMatched = cpuClass != scan->pPeakList->end() && cpuClass > 0;
                    if ((cpuMatched ? cpuClass : 0) != expectedClass)
                        throw std::runtime_error("CPU reference/bucket lookup mismatch");

                    if (results[j].primitiveId != expectedIndex) {
                        std::ostringstream detail;
                        detail << std::setprecision(17)
                               << "RT mismatch: scan=" << scan->iScanId
                               << " ion=" << j << " mz=" << mz
                               << " tolerance=" << tolerance
                               << " expected=" << expectedIndex
                               << " actual=" << results[j].primitiveId
                               << " rt_t=" << results[j].distance;
                        throw std::runtime_error(detail.str());
                    }
                    if (expectedIndex >= 0 &&
                        found->second->peaks().at(expectedIndex).intensityClass != expectedClass)
                        throw std::runtime_error("RT peak metadata class mismatch");
                    ++verifiedIons;
                }
                ++verifiedCandidates;
            } catch (...) {
                // Exceptions must not escape the OpenMP search region.
                rtFailure = std::current_exception();
            }
        };
    #endif

    const double searchBegin = omp_get_wtime();

    // Matching and scoring still use the original CPU implementation.
    spectra.searchDatabaseMvh();
    const double searched = omp_get_wtime();
#ifdef MVH_ENABLE_RT
    mvh_rt::matchObserver = {};
    if (rtFailure) std::rethrow_exception(rtFailure);
    std::cout << "RT verification passed: candidates=" << verifiedCandidates
              << " ions=" << verifiedIons << '\n';
#endif
     const auto count = writePsms((std::filesystem::path(output)/"mvh_psms.tsv").string(), input, scans);
    const double exported = omp_get_wtime();
    std::size_t skipped=0;
    for (const auto *scan : scans) if (scan->bSkip) ++skipped;
    std::ofstream report(std::filesystem::path(output)/"run_summary.tsv");
    report.exceptions(std::ios::failbit | std::ios::badbit);
    report << std::setprecision(17) << "metric\tvalue\n"
           << "omp_max_threads\t" << omp_get_max_threads() << '\n'
           << "peptide_batch_size\t" << PEPTIDE_ARRAY_SIZE << '\n'
           << "config_and_load_seconds\t" << loaded-begin << '\n'
           << "preprocess_seconds\t" << prepared-loaded << '\n'
           << "rt_setup_seconds\t" << searchBegin-prepared << '\n'
           << "search_seconds\t" << searched-searchBegin << '\n'
           << "export_seconds\t" << exported-searched << '\n'
           << "scan_count\t" << scans.size() << '\n'
           << "skipped_scan_count\t" << skipped << '\n'
           << "retained_psm_count\t" << count << '\n';
#ifdef MVH_ENABLE_RT
    report << "rt_verified_candidates\t" << verifiedCandidates << '\n'
           << "rt_verified_ions\t" << verifiedIons << '\n';
#endif
    report.close();
    std::cout << "MVH results: " << output << "\nSearch seconds: " << searched-searchBegin << '\n';
}
