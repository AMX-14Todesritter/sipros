#pragma once
#include <string>
namespace mvh_app {
// One process, one config/spectrum/FASTA, one complete unmodified MVH search.
void run(const std::string &input, const std::string &config,
         const std::string &fasta, const std::string &output, int threads);
}
