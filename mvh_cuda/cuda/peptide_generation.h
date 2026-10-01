#pragma once
#include <memory>
#include <string>
class Peptide;
namespace mvh_cuda {
void setPeptideGeneration(const std::string& mode);
const std::string& peptideGenerationName();
// Same streaming interface as ProteinDatabase; CPU loading/result adapters wrap
// bounded GPU digestion and PTM pages. Original code remains the oracle.
class SearchPeptideGenerator {
public:
    explicit SearchPeptideGenerator(bool screen);
    ~SearchPeptideGenerator();
    void loadDatabase();
    bool getFirstProtein();
    bool getNextPeptide(Peptide* peptide);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
