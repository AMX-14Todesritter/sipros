// Export a real intermediate state by calling the ORIGINAL MVH functions.
// This program never modifies the Sipros sources or runs a database search.
#include "mvh_scan_vector.h"
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

std::ofstream outputFile(const fs::path &path, bool binary = false) {
    std::ofstream out(path, binary ? std::ios::binary : std::ios::out);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(17);
    return out;
}

std::vector<std::string> splitTabs(const std::string &line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) fields.push_back(field);
    return fields;
}

std::string jsonString(const std::string &value) {
    std::ostringstream out;
    out << '"';
    for (char c : value) {
        if (c == '"' || c == '\\') out << '\\';
        if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else out << c;
    }
    return out.str() + '"';
}

int main(int argc, char **argv) {
    try {
        require(argc == 5, "Usage: export_mvh_sample input.FT2 search.cfg selected_psm.tsv NEW_OUTPUT_DIR");
        const fs::path input = fs::absolute(argv[1]), config = fs::absolute(argv[2]);
        const fs::path psmPath = fs::absolute(argv[3]), output = fs::absolute(argv[4]);
        require(!fs::exists(output), "Output directory must not exist");
        std::ifstream psm(psmPath);
        std::string header, row;
        require(bool(std::getline(psm, header)) && bool(std::getline(psm, row)), "Missing PSM record");
        const auto names = splitTabs(header), values = splitTabs(row);
        require(names.size() == values.size(), "Invalid PSM columns");
        std::map<std::string, std::string> selected;
        for (size_t i = 0; i < names.size(); ++i) selected[names[i]] = values[i];
        const int scanId = std::stoi(selected.at("scan_id"));
        const int precursorCharge = std::stoi(selected.at("precursor_charge"));
        const double referenceScore = std::stod(selected.at("mvh_score"));
        require(precursorCharge == 2, "This teaching fixture labels the charge-2 b/y series only");
        require(ProNovoConfig::setFilename(config.string()), "Cannot load config");
        require(ProNovoConfig::getSearchType() == "Regular", "Expected Regular search");
        omp_set_num_threads(1);
        fs::create_directories(output);
        fs::copy_file(config, output / "search.cfg");
        fs::copy_file(psmPath, output / "selected_psm.tsv");

        // Read the FULL input, not an isolated scan: MVH uses global m/z bounds.
        MvhScanVector spectra(input.string(), output.string(), config.string(), false);
        require(spectra.loadMassData(), "Cannot load input spectra");
        MS2Scan *scan = nullptr;
        size_t scanIndex = 0;
        for (size_t i = 0; i < spectra.vpAllMS2Scans.size(); ++i) {
            if (spectra.vpAllMS2Scans[i]->iScanId == scanId) {
                require(scan == nullptr, "Scan ID is not unique");
                scan = spectra.vpAllMS2Scans[i];
                scanIndex = i;
            }
        }
        require(scan != nullptr, "Selected scan missing");
        {
            auto out = outputFile(output / "raw_peaks.tsv");
            out << "peak_index\tmz\tintensity\tcharge\n";
            for (size_t i = 0; i < scan->vdMZ.size(); ++i)
                out << i << '\t' << scan->vdMZ[i] << '\t' << scan->vdIntensity[i]
                    << '\t' << scan->viCharge[i] << '\n';
        }
        const bool rawSorted = std::is_sorted(scan->vdMZ.begin(), scan->vdMZ.end());
        // Same prefix as MS2Scan::preprocessMvh(), stopping before new PeakList.
        scan->sortPeakList();
        scan->peakData = new std::map<double, char>();
        scan->intenClassCounts = new std::vector<int>();
        std::multimap<double, double> intensityWorkspace;
        require(MVH::Preprocess(scan, &intensityWorkspace), "Preprocess failed");
        require(!scan->bSkip && scan->pPeakList == nullptr, "Not at the requested pre-bucket state");
        {
            auto out = outputFile(output / "experimental_peaks.tsv");
            out << "peak_index\tmz\tintensity_class\n";
            size_t index = 0;
            for (const auto &peak : *scan->peakData)
                out << index++ << '\t' << peak.first << '\t' << int(peak.second) << '\n';
        }
        {
            auto out = outputFile(output / "scan_state.json");
            out << "{\n  \"schema_version\": 1,\n  \"scan_id\": " << scanId
                << ",\n  \"scan_index\": " << scanIndex
                << ",\n  \"input_file\": " << jsonString(input.string())
                << ",\n  \"stage\": \"after MVH::Preprocess, before PeakList construction\""
                << ",\n  \"pPeakList_is_null\": true,\n  \"integer_bucket_index_present\": false"
                << ",\n  \"mz_sorted\": true,\n  \"raw_mz_sorted\": " << (rawSorted ? "true" : "false")
                << ",\n  \"raw_peak_count\": " << scan->vdMZ.size()
                << ",\n  \"retained_peak_count\": " << scan->peakData->size()
                << ",\n  \"retention_time_minutes\": " << std::stod(scan->sRTime)
                << ",\n  \"preprocessing_parent_mz\": " << scan->dParentMZ
                << ",\n  \"preprocessing_parent_charge\": " << scan->iParentChargeState
                << ",\n  \"preprocessing_parent_neutral_mass\": " << scan->dParentNeutralMass
                << ",\n  \"matched_precursor_charge\": " << precursorCharge
                << ",\n  \"matched_precursor_neutral_mass\": " << selected.at("precursor_mass")
                << ",\n  \"mzLowerBound\": " << scan->mzLowerBound
                << ",\n  \"mzUpperBound\": " << scan->mzUpperBound
                << ",\n  \"fragment_tolerance\": " << ProNovoConfig::getMassAccuracyFragmentIon()
                << ",\n  \"minimum_matched_fragments\": " << ProNovoConfig::MinMatchedFragments
                << ",\n  \"bSkip\": false,\n  \"totalPeakBins\": " << scan->totalPeakBins
                << ",\n  \"intenClassCounts\": [";
            for (size_t i = 0; i < scan->intenClassCounts->size(); ++i) {
                if (i) out << ", ";
                out << scan->intenClassCounts->at(i);
            }
            out << "]\n}\n";
        }

        Peptide peptide;
        peptide.sPeptide = selected.at("peptide");
        peptide.preprocessingMVH();
        std::vector<double> ions, forward, reverse;
        std::vector<char> residues;
        require(MVH::CalculateSequenceIons(peptide.sNeutralLossPeptide, precursorCharge,
                    MVH::bUseSmartPlusThreeModel, &ions, &forward, &reverse, &residues),
                "Original theoretical ion generation failed");
        require(MVH::fragmentTypes[FragmentType_B] && MVH::fragmentTypes[FragmentType_Y]
                && ions.size() == 2 * residues.size() - 1, "Unexpected ion-label layout");
        {
            auto out = outputFile(output / "theoretical_ions.tsv");
            out << "ion_index\tion_label\tfragment_charge\tmz\tin_global_mz_range\n";
            for (size_t i = 0; i < ions.size(); ++i) {
                const bool fullY = i + 1 == ions.size();
                const std::string label = fullY ? "y" + std::to_string(residues.size())
                    : std::string(i % 2 ? "y" : "b") + std::to_string(i / 2 + 1);
                out << i << '\t' << label << "\t1\t" << ions[i] << '\t'
                    << (ions[i] >= scan->mzLowerBound && ions[i] <= scan->mzUpperBound) << '\n';
            }
        }
        {
            auto out = outputFile(output / "peptide_state.json");
            out << "{\n  \"peptide\": " << jsonString(peptide.sPeptide)
                << ",\n  \"original_peptide\": " << jsonString(selected.at("original_peptide"))
                << ",\n  \"neutral_loss_peptide\": " << jsonString(peptide.sNeutralLossPeptide)
                << ",\n  \"protein_names\": " << jsonString(selected.at("protein_names"))
                << ",\n  \"calculated_neutral_mass\": " << selected.at("calculated_mass")
                << ",\n  \"precursor_charge\": " << precursorCharge
                << ",\n  \"ion_count\": " << ions.size()
                << ",\n  \"smart_charge_model\": " << (MVH::bUseSmartPlusThreeModel ? "true" : "false")
                << ",\n  \"ordering\": \"original CalculateSequenceIons order, not m/z-sorted\"\n}\n";
        }

        // Validation starts AFTER both requested snapshots have been written.
        // Building buckets here must not be confused with the exported state.
        scan->pPeakList = new PeakList(scan->peakData);
        MVH::initialLnTable(scan->totalPeakBins);
        double score = 0;
        require(MVH::ScoreSequenceVsSpectrum(peptide.sNeutralLossPeptide, precursorCharge,
                    scan, &ions, &forward, &reverse, score, &residues), "Reference scoring failed");
        require(score == referenceScore, "Snapshot score differs from the historical PSM");
        {
            auto out = outputFile(output / "ln_factorial_table.bin", true);
            for (int i = 0; i <= scan->totalPeakBins; ++i) {
                const double value = (*MVH::lnTable)[i];
                out.write(reinterpret_cast<const char *>(&value), sizeof(value));
            }
        }
        int matched = 0;
        std::vector<int> key(ProNovoConfig::NumIntensityClasses + 1, 0);
        {
            auto out = outputFile(output / "reference_matches.tsv");
            out << "ion_index\ttheoretical_mz\texperimental_peak_index\texperimental_mz\terror\tclass\n";
            for (size_t i = 0; i < ions.size(); ++i) {
                if (ions[i] < scan->mzLowerBound || ions[i] > scan->mzUpperBound) continue;
                int nearest = -1;
                double best = std::numeric_limits<double>::infinity();
                for (size_t j = 0; j < scan->pPeakList->pPeaks.size(); ++j) {
                    const double error = std::abs(ions[i] - scan->pPeakList->pPeaks[j]);
                    if (error < best) { best = error; nearest = j; }
                }
                if (!(best < ProNovoConfig::getMassAccuracyFragmentIon())) nearest = -1;
                const int cls = nearest < 0 ? 0 : scan->pPeakList->pClasses[nearest];
                const char original = scan->pPeakList->findNear(ions[i], ProNovoConfig::getMassAccuracyFragmentIon());
                require(cls == (original == scan->pPeakList->end() ? 0 : int(original)), "Linear and bucket lookup differ");
                ++key[cls ? cls - 1 : ProNovoConfig::NumIntensityClasses];
                out << i << '\t' << ions[i] << '\t' << nearest << '\t';
                if (nearest >= 0) {
                    ++matched;
                    out << scan->pPeakList->pPeaks[nearest] << '\t'
                        << scan->pPeakList->pPeaks[nearest] - ions[i] << '\t' << cls << '\n';
                } else out << "\t\t0\n";
            }
        }
        {
            auto out = outputFile(output / "validation.json");
            out << "{\n  \"score_exactly_matches_historical_psm\": true,\n  \"mvh_score\": " << score
                << ",\n  \"matched_ions\": " << matched << ",\n  \"mvhKey\": [";
            for (size_t i = 0; i < key.size(); ++i) { if (i) out << ", "; out << key[i]; }
            out << "],\n  \"ln_table_encoding\": \"little-endian IEEE754 float64, index 0..totalPeakBins\"\n}\n";
        }
        MVH::destroyLnTable();
        std::cout << "PASS: scan=" << scanId << " peaks=" << scan->peakData->size()
                  << " ions=" << ions.size() << " matches=" << matched
                  << " score=" << std::setprecision(17) << score << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
