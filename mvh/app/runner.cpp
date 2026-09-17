#include "runner.h"
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
    omp_set_num_threads(threads);
    const double begin = omp_get_wtime();
    if (!ProNovoConfig::setFilename(config)) throw std::runtime_error("Cannot load config");
    if (ProNovoConfig::getSearchType() != "Regular")
        throw std::runtime_error("Only Search_Type = Regular is supported");
    ProNovoConfig::setFASTAfilename(fasta);
    if (!std::filesystem::create_directory(output))
        throw std::runtime_error("Output directory must not exist; its parent must exist");
    std::filesystem::copy_file(config, std::filesystem::path(output)/"input_config.cfg");
    MvhScanVector spectra(input, output, config, true);
    if (!spectra.loadMassData()) throw std::runtime_error("Cannot load spectra");
    const double loaded = omp_get_wtime();
    const auto &scans = spectra.vpAllMS2Scans;
    if (scans.empty()) throw std::runtime_error("No scans loaded");
    spectra.preProcessAllMs2Mvh();
    const double prepared = omp_get_wtime();
    spectra.searchDatabaseMvh();
    const double searched = omp_get_wtime();
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
           << "search_seconds\t" << searched-prepared << '\n'
           << "export_seconds\t" << exported-searched << '\n'
           << "scan_count\t" << scans.size() << '\n'
           << "skipped_scan_count\t" << skipped << '\n'
           << "retained_psm_count\t" << count << '\n';
    report.close();
    std::cout << "MVH results: " << output << "\nSearch seconds: " << searched-prepared << '\n';
}
