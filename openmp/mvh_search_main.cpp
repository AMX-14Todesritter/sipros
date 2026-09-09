#include "ms2scanvector.h"
#include <cctype>
#include <iomanip>
#include <stdexcept>

namespace fs = std::filesystem;

static void usage()
{
    std::cout << "Usage: sipros_mvh_search -f input.ft2|input.mzML -c search.cfg\n"
                 "       -fasta proteins.fasta -o NEW_OUTPUT_DIRECTORY [-t threads]\n"
                 "Regular search only. Preserves production MVH search and batch size.\n"
                 "Outputs mvh_psms.tsv and run_summary.tsv; no WDP/Xcorr or FDR filtering.\n";
}

int main(int argc, char **argv)
{
    try {
        std::string input, config, fasta, output;
        int threads = 0;
        for (int i = 1; i < argc; ++i) {
            std::string option = argv[i];
            if (option == "-h" || option == "--help") { usage(); return 0; }
            if (option != "-f" && option != "-c" && option != "-fasta" && option != "-o" && option != "-t")
                throw std::runtime_error("Unknown option: " + option);
            if (++i >= argc) throw std::runtime_error("Missing value for " + option);
            std::string value = argv[i];
            if (option == "-f") input = value;
            else if (option == "-c") config = value;
            else if (option == "-fasta") fasta = value;
            else if (option == "-o") output = value;
            else {
                size_t used = 0;
                threads = std::stoi(value, &used);
                if (used != value.size() || threads < 1)
                    throw std::runtime_error("Thread count must be a positive integer");
            }
        }
        if (input.empty() || config.empty() || fasta.empty() || output.empty()) {
            usage();
            throw std::runtime_error("-f, -c, -fasta and -o are required");
        }
        for (auto *path : {&input, &config, &fasta}) {
            *path = fs::absolute(*path).string();
            if (!fs::is_regular_file(*path) || fs::file_size(*path) == 0)
                throw std::runtime_error("Missing or empty input file: " + *path);
        }
        std::string suffix = fs::path(input).extension().string();
        for (char &c : suffix) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (suffix != ".ft2" && suffix != ".mzml")
            throw std::runtime_error("Input must be FT2 or mzML, not vendor RAW");
        std::ifstream database(fasta);
        if (database.peek() != '>') throw std::runtime_error("FASTA must start with a > header");
        output = fs::absolute(output).string();
        if (fs::exists(output)) throw std::runtime_error("Output directory already exists: " + output);
        if (threads > 0) omp_set_num_threads(threads);

        const double prepareBegin = omp_get_wtime();
        if (!ProNovoConfig::setFilename(config)) throw std::runtime_error("Could not load config: " + config);
        if (ProNovoConfig::getSearchType() != "Regular")
            throw std::runtime_error("This entry supports only Search_Type = Regular");
        ProNovoConfig::setFASTAfilename(fasta);
        fs::create_directories(output);
        fs::copy_file(config, fs::path(output) / "input_config.cfg");
        MS2ScanVector scans(input, output, config, true);
        if (!scans.loadMassData()) throw std::runtime_error("Could not load spectra: " + input);
        const double loadSeconds = omp_get_wtime() - prepareBegin;
        auto summary = scans.startProcessingMvhOnly((fs::path(output) / "mvh_psms.tsv").string());

        std::ofstream report(fs::path(output) / "run_summary.tsv");
        if (!report) throw std::runtime_error("Cannot open run_summary.tsv");
        report.exceptions(std::ios::failbit | std::ios::badbit);
        report << std::setprecision(17) << "metric\tvalue\n"
               << "input_file\t" << input << '\n' << "config_file\t" << config << '\n'
               << "fasta_file\t" << fasta << '\n'
               << "omp_max_threads\t" << omp_get_max_threads() << '\n'
               << "peptide_batch_size\t" << PEPTIDE_ARRAY_SIZE << '\n'
               << "config_and_load_seconds\t" << loadSeconds << '\n'
               << "preprocess_seconds\t" << summary.preprocessSeconds << '\n'
               << "prepare_seconds\t" << loadSeconds + summary.preprocessSeconds << '\n'
               << "search_seconds\t" << summary.searchSeconds << '\n'
               << "scan_count\t" << summary.scanCount << '\n'
               << "precursor_entry_count\t" << summary.precursorCount << '\n'
               << "skipped_scan_count\t" << summary.skippedScanCount << '\n'
               << "retained_psm_count\t" << summary.retainedPsmCount << '\n';
        report.close();
        std::cout << "MVH-only results: " << output << "\nSearch wall time: "
                  << summary.searchSeconds << " seconds\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "sipros_mvh_search: " << error.what() << '\n';
        return 1;
    }
}
