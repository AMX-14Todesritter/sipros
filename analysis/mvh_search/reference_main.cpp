// Only adapt thread selection; run the original CLI and processing path unchanged.
#define main siprosOriginalMain
#include "../../openmp/main.cpp"
#undef main
#include <stdexcept>

int main(int argc, char **argv)
{
    try {
        int kept = 1;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "-t") {
                if (++i == argc) throw std::runtime_error("Missing -t value");
                size_t used = 0;
                std::string value(argv[i]);
                int count = std::stoi(value, &used);
                if (used != value.size() || count < 1) throw std::runtime_error("Invalid -t value");
                omp_set_num_threads(count);
            } else argv[kept++] = argv[i];
        }
        argv[kept] = nullptr;
        return siprosOriginalMain(kept, argv);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
