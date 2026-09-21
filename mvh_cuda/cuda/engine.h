#pragma once
#include <vector>
class MS2Scan;
class Peptide;
namespace mvh_cuda {
void setVerification(bool enabled);
void preProcessAllMs2Mvh(std::vector<MS2Scan *> &scans);
void preprocessingMVH(std::vector<Peptide *> &peptides);
void scorePeptidesMVH(std::vector<MS2Scan *> &scans);
}
namespace mvh_cuda { void runContractTests(); }
