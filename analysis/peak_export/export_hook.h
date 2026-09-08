#pragma once
// Used only in the isolated export build. The runner forces one OpenMP thread.
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <string>

namespace PeakExport {
inline bool selected = false;
inline int scanId = -1;
inline std::string protein, originalPeptide;
inline double peptideMass = 0;

inline std::ofstream output(const char *name) {
    const char *dir = std::getenv("SIPROS_PEAK_EXPORT_DIR");
    if (!dir) { std::cerr << "Missing export directory\n"; std::exit(2); }
    std::ofstream out(std::filesystem::path(dir) / name);
    if (!out) { std::cerr << "Cannot write " << name << '\n'; std::exit(2); }
    out << std::setprecision(17);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    return out;
}

inline void raw(MS2Scan *scan) {
    if (selected) return;
    const char *requested = std::getenv("SIPROS_PEAK_EXPORT_SCAN");
    if (requested && scan->iScanId != std::stoi(requested)) return;
    if (!requested && scan->vdMZ.size() < static_cast<size_t>(ProNovoConfig::minIntensityClassCount)) return;
    selected = true;
    scanId = scan->iScanId;
    auto out = output("observed_unsorted.tsv");
    out << "scan_id\tinput_peak_index\tmz\tintensity\tcharge\n";
    for (size_t i = 0; i < scan->vdMZ.size(); ++i)
        out << scanId << '\t' << i << '\t' << scan->vdMZ.at(i) << '\t'
            << scan->vdIntensity.at(i) << '\t' << scan->viCharge.at(i) << '\n';
    out.close();
}

inline void candidate(Peptide *peptide) {
    protein = peptide->getProteinName();
    originalPeptide = peptide->getPeptideSeq();
    peptideMass = peptide->getPeptideMass();
}

inline void theoretical(MS2Scan *scan, const std::string &sequence, int charge,
                        const std::vector<double> *ions) {
    if (!selected || scan->iScanId != scanId) return;
    auto out = output("theoretical_unsorted.tsv");
    out << "scan_id\tfragment_index\ttheoretical_mz\tin_spectrum_range\n";
    for (size_t i = 0; i < ions->size(); ++i) {
        double mz = ions->at(i);
        out << scanId << '\t' << i << '\t' << mz << '\t'
            << (mz >= scan->mzLowerBound && mz <= scan->mzUpperBound) << '\n';
    }
    out.close();
    auto retained = output("observed_mvh.tsv");
    retained << "scan_id\tretained_peak_index\tmz\tmvh_class\n";
    for (int i = 0; i < scan->pPeakList->size(); ++i)
        retained << scanId << '\t' << i << '\t' << scan->pPeakList->pPeaks.at(i)
                 << '\t' << static_cast<int>(scan->pPeakList->pClasses.at(i)) << '\n';
    retained.close();
    auto meta = output("candidate.tsv");
    meta << "scan_id\tprecursor_charge\tpeptide\tscoring_sequence\tprotein_source\tpeptide_mass\tfragment_tolerance\n"
         << scanId << '\t' << charge << '\t' << originalPeptide << '\t' << sequence
         << '\t' << protein << '\t' << peptideMass << '\t'
         << ProNovoConfig::getMassAccuracyFragmentIon() << '\n';
    meta.close();
    std::cout << "Exported one scan and its first scored candidate; stopping before MVH matching.\n";
    std::exit(0);
}
}
