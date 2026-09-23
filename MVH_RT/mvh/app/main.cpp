#include "runner.h"
#include "output_paths.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <cctype>

int main(int argc, char **argv) {
    try {
        std::map<std::string,std::string> options;
        for (int i=1; i<argc; ++i) {
            std::string key=argv[i];
            if (key=="--help" || key=="-h") {
                std::cout << "Usage: sipros_mvh -f spectra.ft2|spectra.mzML -c search.cfg "
                             "-fasta proteins.fasta [-o NEW_DIRECTORY] [-t threads]\n";
                std::cout << "Default output: output/search/mvh_rt/<input>_<UTC timestamp>/\n"
                             "SIPROS_OUTPUT_ROOT overrides the output root.\n";
                return 0;
            }
            if (key!="-f" && key!="-c" && key!="-fasta" && key!="-o" && key!="-t")
                throw std::runtime_error("Unknown option: "+key);
            if (++i==argc || options.count(key)) throw std::runtime_error("Missing/duplicate option: "+key);
            options[key]=argv[i];
        }
        for (const auto *key : {"-f","-c","-fasta"})
            if (!options.count(key) || options[key].empty()) throw std::runtime_error(std::string("Required: ")+key);
        int threads=1;
        if (options.count("-t")) {
            std::size_t used=0;
            threads=std::stoi(options["-t"], &used);
            if (used!=options["-t"].size() || threads<1) throw std::runtime_error("Invalid thread count");
        }
        for (const auto *key : {"-f","-c","-fasta"}) {
            auto &path=options[key];
            path=std::filesystem::absolute(path).lexically_normal().string();
            if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path)==0)
                throw std::runtime_error("Missing/empty input: "+path);
        }
        std::string ext=std::filesystem::path(options["-f"]).extension().string();
        for (auto &c:ext) c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (ext!=".ft2" && ext!=".mzml") throw std::runtime_error("Expected FT2 or mzML");
        std::ifstream fasta(options["-fasta"]);
        if (fasta.peek()!='>') throw std::runtime_error("FASTA must begin with >");
        if (!options.count("-o")) {
            options["-o"] = sipros_output::searchDirectory("mvh_rt", options["-f"]).string();
        } else if (options["-o"].empty()) {
            throw std::runtime_error("Output directory must not be empty");
        }
        options["-o"]=std::filesystem::absolute(options["-o"]).lexically_normal().string();
        if (std::filesystem::exists(options["-o"])) throw std::runtime_error("Output already exists");
        mvh_app::run(options["-f"],options["-c"],options["-fasta"],options["-o"],threads);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "sipros_mvh: " << e.what() << '\n';
        return 1;
    }
}
