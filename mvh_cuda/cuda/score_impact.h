#pragma once

#include "types.cuh"
#include <array>
#include <fstream>

namespace mvh_cuda {

// Candidate occurrences before merge/top filtering, not individual peaks.
enum ImpactCount {
    BothScored, ReferenceOnlyScored, RtOnlyScored, NeitherScored,
    ChangedScore, InvalidResult, ImpactCountSize
};
struct ScoreDifference {
    Candidate candidate;
    Result reference;
    Result observed;
};
struct ScoreImpactBatch {
    unsigned long long candidates = 0;
    std::array<unsigned long long, ImpactCountSize> counts{};
    std::vector<ScoreDifference> differences;
};

ScoreImpactBatch collectScoreImpact(const Candidate *candidates, int size,
                                   const Result *reference, const Result *observed);

// Open only for explicitly requested diagnostic runs in a new output directory.
class ScoreImpactWriter {
public:
    explicit ScoreImpactWriter(const std::string &directory);
    void writeBatch(const ScoreImpactBatch &batch);
    void writeDifference(const ScoreDifference &difference, int scanId,
                         double precursorMass, const std::string &peptide);
private:
    unsigned long long batchIndex = 0;
    std::ofstream summary;
    std::ofstream differences;
};

} // namespace mvh_cuda
