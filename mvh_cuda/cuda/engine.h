#pragma once
#include <vector>
#include <tuple>
#include <string>
class MS2Scan;
class Peptide;
namespace mvh_cuda {
void setVerification(bool enabled);
void setScoreImpact(bool enabled);
void startScoreImpact(const std::string &outputDirectory);
void setMatchBackend(const std::string &name);
const std::string &matchBackendName();
struct MatchBackendScope { ~MatchBackendScope(); };
void setPeptideBatchSize(int size);
int peptideBatchSize();
void preProcessAllMs2Mvh(std::vector<MS2Scan *> &scans);
void preprocessingMVH(std::vector<Peptide *> &peptides);
void scorePeptidesMVH(std::vector<MS2Scan *> &scans, const std::vector<Peptide *> &peptides);
void assignPeptides2Scans(const std::vector<Peptide *> &peptides,
                         const std::vector<std::tuple<double, int, MS2Scan *>> &precursors,
                         const std::vector<MS2Scan *> &scans);
}
namespace mvh_cuda { void runContractTests(); }
