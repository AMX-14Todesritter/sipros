#pragma once
#include <vector>
#include "peptide_batch.h"
#include <tuple>
#include <string>
class MS2Scan;
class Peptide;
namespace mvh_cuda {
void setVerification(bool enabled);
bool verificationEnabled();
void setResultRestoration(const std::string& mode);
const char* resultRestorationName();
void beginSearchResults(const std::vector<MS2Scan*>& scans);
void finishSearchResults(const std::vector<MS2Scan*>& scans);
void setScoreImpact(bool enabled);
void startScoreImpact(const std::string &outputDirectory);
void setMatchBackend(const std::string &name);
const std::string &matchBackendName();
struct MatchBackendScope { ~MatchBackendScope(); };
void setPeptideBatchSize(int size);
void setSpectrumDeviceCache(bool enabled);
bool spectrumDeviceCache();
int peptideBatchSize();
void preProcessAllMs2Mvh(std::vector<MS2Scan *> &scans);
// Borrowed until the next pack/reset; production packs once for all stages.
const PeptideBatch& packPeptideBatch(const std::vector<Peptide*>& peptides);
void preprocessingMVH(const PeptideBatch& peptides);
void scorePeptidesMVH(std::vector<MS2Scan*>& scans, const PeptideBatch& inputs,
                      const std::vector<Peptide*>& resultObjects);
void assignPeptides2Scans(const PeptideBatch& peptides,
                         const std::vector<std::tuple<double, int, MS2Scan *>>& precursors,
                         const std::vector<MS2Scan*>& scans);
void preprocessingMVH(std::vector<Peptide *> &peptides);
void scorePeptidesMVH(std::vector<MS2Scan *> &scans, const std::vector<Peptide *> &peptides);
void assignPeptides2Scans(const std::vector<Peptide *> &peptides,
                         const std::vector<std::tuple<double, int, MS2Scan *>> &precursors,
                         const std::vector<MS2Scan *> &scans);
}
namespace mvh_cuda { void runContractTests(); }
