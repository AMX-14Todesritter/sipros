#include "runner.h"
#include "engine.h"
#include "profiling.h"
#include "peptide_generation.h"
#include "mvh_scan_vector.h"
#include <iomanip>
#include <stdexcept>

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
#ifdef MVH_ENABLE_FLOW_COUNTERS
    mvh_flow::reset();
#endif
    MVH_PROFILE_SCOPE("mvh/run");
    MVH_PROFILE_BEGIN(loadRange, "mvh/run/config_and_load");
    mvh_cuda::MatchBackendScope matchResources;
    omp_set_num_threads(1); // Interface retains -t; CUDA replaces CPU parallel work.
    std::cout << "Requested CPU threads: " << threads << "; CUDA execution uses no OpenMP compute loops\n";
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
    mvh_cuda::startScoreImpact(output);
    std::filesystem::copy_file(config, std::filesystem::path(output)/"input_config.cfg");
    MvhScanVector spectra(input, output, config, true);
    if (!spectra.loadMassData()) throw std::runtime_error("Cannot load spectra");
    const double loaded = omp_get_wtime();
    MVH_PROFILE_END(loadRange);
    const auto &scans = spectra.vpAllMS2Scans;
    if (scans.empty()) throw std::runtime_error("No scans loaded");
    MVH_PROFILE_BEGIN(preprocessRange, "mvh/run/preprocess_scans");
    spectra.preProcessAllMs2Mvh();
    MVH_PROFILE_END(preprocessRange);
    const double prepared = omp_get_wtime();
    spectra.searchDatabaseMvh();
    const double searched = omp_get_wtime();
    MVH_PROFILE_BEGIN(exportRange, "mvh/run/export_psms");
    const auto count = writePsms((std::filesystem::path(output)/"mvh_psms.tsv").string(), input, scans);
    const double exported = omp_get_wtime();
    MVH_PROFILE_END(exportRange);
    std::size_t skipped=0;
    for (const auto *scan : scans) if (scan->bSkip) ++skipped;
    std::ofstream report(std::filesystem::path(output)/"run_summary.tsv");
    report.exceptions(std::ios::failbit | std::ios::badbit);
    report << std::setprecision(17) << "metric\tvalue\n"
           << "match_backend\t" << mvh_cuda::matchBackendName() << '\n'
           << "backend\tcuda\n"
           << "result_restoration\t" << mvh_cuda::resultRestorationName() << '\n'
           << "peptide_generation\t" << mvh_cuda::peptideGenerationName() << '\n'
           << "spectrum_cache\t" << (mvh_cuda::spectrumDeviceCache() ? "device" : "host") << '\n'
           << "omp_max_threads\t" << omp_get_max_threads() << '\n'
           << "peptide_batch_size\t" << mvh_cuda::peptideBatchSize() << '\n'
           << "config_and_load_seconds\t" << loaded-begin << '\n'
           << "preprocess_seconds\t" << prepared-loaded << '\n'
           << "search_seconds\t" << searched-prepared << '\n'
           << "export_seconds\t" << exported-searched << '\n'
           << "scan_count\t" << scans.size() << '\n'
           << "skipped_scan_count\t" << skipped << '\n'
           << "retained_psm_count\t" << count << '\n';
    if (mvh_cuda::matchBackendName()=="rt-custom")
        report << "rt_scan_group_size\t" << mvh_cuda::rtScanGroupSize() << '\n'
               << "rt_scan_grouping\tmin-neutral-mass\nrt_class_priority\t3,2,1\n"
               << "rt_workspace_mib\t" << mvh_cuda::rtWorkspaceMiB() << '\n';
    report.close();
#ifdef MVH_ENABLE_FLOW_COUNTERS
    mvh_flow::write(output,searched-prepared,count);
#endif
    std::cout << "MVH results: " << output << "\nSearch seconds: " << searched-prepared << '\n';
}
