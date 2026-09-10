#include "mvh/session.h"
#include "ms2scanvector.h"
#include <cctype>
#include <iomanip>
#include <stdexcept>

namespace fs = std::filesystem;
namespace {
void usage() {
    std::cout << "sipros_mvh_snapshot (-f input.ft2|input.mzML | --snapshot input.mvh)\n"
                 "  -c config.cfg -o NEW_DIRECTORY [-fasta database.fasta] [-t threads]\n"
                 "  [--repeat N] [--prepare-only]\n"
                 "Raw input writes preprocessed.mvh; --prepare-only needs no FASTA.\n"
                 "Search runs keep separate run_1/, run_2/, ... TSVs. Regular only.\n";
}
int positive(const std::string &text) {
    std::size_t used = 0; int n = std::stoi(text, &used);
    if (used != text.size() || n < 1) throw std::runtime_error("Expected a positive integer: " + text);
    return n;
}
std::string inputPath(const std::string &name) {
    auto p = fs::absolute(name);
    if (!fs::is_regular_file(p) || fs::file_size(p) == 0) throw std::runtime_error("Missing/empty input: " + p.string());
    return p.string();
}
std::ofstream report(const fs::path &path) {
    std::ofstream out(path); out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(17) << "metric\tvalue\n"; return out;
}
}
int main(int argc, char **argv) {
    try {
        std::string input, snapshot, config, fasta, output;
        int threads = 0, repeat = 1; bool prepareOnly = false;
        for (int i = 1; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--help" || flag == "-h") { usage(); return 0; }
            if (flag == "--prepare-only") { prepareOnly = true; continue; }
            if (flag != "-f" && flag != "--snapshot" && flag != "-c" && flag != "-fasta" && flag != "-o" && flag != "-t" && flag != "--repeat")
                throw std::runtime_error("Unknown option: " + flag);
            if (++i == argc) throw std::runtime_error("Missing value: " + flag);
            std::string value = argv[i];
            if (flag == "-f") input = value;
            else if (flag == "--snapshot") snapshot = value;
            else if (flag == "-c") config = value;
            else if (flag == "-fasta") fasta = value;
            else if (flag == "-o") output = value;
            else if (flag == "-t") threads = positive(value);
            else repeat = positive(value);
        }
        if (input.empty() == snapshot.empty() || config.empty() || output.empty())
            throw std::runtime_error("Exactly one of -f/--snapshot, plus -c and -o, is required");
        if (prepareOnly && (!snapshot.empty() || repeat != 1)) throw std::runtime_error("--prepare-only requires raw input and no repeats");
        config = inputPath(config);
        if (!input.empty()) {
            input = inputPath(input);
            auto suffix = fs::path(input).extension().string();
            for (char &c : suffix) c = std::tolower(static_cast<unsigned char>(c));
            if (suffix != ".ft2" && suffix != ".mzml") throw std::runtime_error("Input must be FT2 or mzML");
        } else snapshot = inputPath(snapshot);
        if (!prepareOnly) {
            if (fasta.empty()) throw std::runtime_error("-fasta is required for search");
            fasta = inputPath(fasta);
            std::ifstream f(fasta);
            if (f.peek() != '>') throw std::runtime_error("FASTA must begin with >");
        }
        output = fs::absolute(output).string();
        if (fs::exists(output)) throw std::runtime_error("Output directory already exists: " + output);
        if (threads) omp_set_num_threads(threads);
        double begin = omp_get_wtime();
        if (!ProNovoConfig::setFilename(config)) throw std::runtime_error("Could not load config");
        if (ProNovoConfig::getSearchType() != "Regular") throw std::runtime_error("Only Search_Type = Regular is supported");
        if (!prepareOnly) ProNovoConfig::setFASTAfilename(fasta);
        double configSeconds = omp_get_wtime()-begin;
        if (!fs::create_directories(output)) throw std::runtime_error("Output directory was not newly created");
        fs::copy_file(config, fs::path(output)/"input_config.cfg");
        mvh::SearchSession session(config, output);
        auto prep = input.empty() ? session.loadSnapshot(snapshot) : session.prepareSpectra(input);
        double saveSeconds = 0;
        if (!input.empty()) saveSeconds = session.saveSnapshot((fs::path(output)/"preprocessed.mvh").string());
        auto p = report(fs::path(output)/"preparation.tsv");
        p << "input_file\t" << session.inputFile() << '\n' << "config_file\t" << config << '\n'
          << "fasta_file\t" << fasta << '\n' << "input_mode\t" << (input.empty() ? "snapshot" : "spectra") << '\n'
          << "snapshot_file\t" << (input.empty() ? snapshot : (fs::path(output)/"preprocessed.mvh").string()) << '\n'
          << "snapshot_version\t" << mvh::snapshotVersion << '\n' << "build_identity\t" << mvh::buildIdentity() << '\n'
          << "omp_max_threads\t" << omp_get_max_threads() << '\n'
          << "config_seconds\t" << configSeconds << '\n'
          << "spectrum_load_seconds\t" << prep.spectrumLoadSeconds << '\n'
          << "preprocess_seconds\t" << prep.preprocessSeconds << '\n'
          << "snapshot_load_seconds\t" << prep.snapshotLoadSeconds << '\n'
          << "snapshot_save_seconds\t" << saveSeconds << '\n';
        p.close();
        if (prepareOnly) { std::cout << "Snapshot prepared: " << output << '\n'; return 0; }
        for (int i = 1; i <= repeat; ++i) {
            auto dir = fs::path(output)/("run_"+std::to_string(i));
            if (!fs::create_directory(dir)) throw std::runtime_error("Run directory already exists");
            auto result = session.search((dir/"mvh_psms.tsv").string());
            auto r = report(dir/"run_summary.tsv");
            r << "repeat\t" << i << '\n' << "omp_max_threads\t" << omp_get_max_threads() << '\n'
              << "peptide_batch_size\t" << PEPTIDE_ARRAY_SIZE << '\n'
              << "state_prepare_seconds\t" << result.statePrepareSeconds << '\n'
              << "search_seconds\t" << result.searchSeconds << '\n'
              << "export_seconds\t" << result.exportSeconds << '\n'
              << "scan_count\t" << result.scans << '\n' << "precursor_entry_count\t" << result.precursors << '\n'
              << "skipped_scan_count\t" << result.skipped << '\n' << "retained_psm_count\t" << result.retained << '\n';
            r.close();
            std::cout << "Run " << i << ": search=" << result.searchSeconds << " seconds, retained=" << result.retained << '\n';
        }
        return 0;
    } catch (const std::exception &e) { std::cerr << "sipros_mvh_snapshot: " << e.what() << '\n'; return 1; }
}
