#include "score_impact.h"
#include <cub/cub.cuh>
#include <filesystem>
#include <iomanip>

namespace mvh_cuda {
namespace {

struct KeepScoreDifference {
    const Result *reference;
    const Result *observed;
    __device__ bool operator()(int index) const {
        const auto &left = reference[index];
        const auto &right = observed[index];
        return left.status != right.status ||
               (left.status == ResultScored && right.status == ResultScored &&
                left.score != right.score);
    }
};

__global__ void countScoreImpact(const Result *reference, const Result *observed,
                                 int size, unsigned long long *counts) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    const auto &left = reference[index];
    const auto &right = observed[index];
    const bool leftScored = left.status == ResultScored;
    const bool rightScored = right.status == ResultScored;
    const int category = leftScored ? (rightScored ? BothScored : ReferenceOnlyScored)
                                    : (rightScored ? RtOnlyScored : NeitherScored);
    atomicAdd(counts + category, 1ULL);
    if (leftScored && rightScored && left.score != right.score)
        atomicAdd(counts + ChangedScore, 1ULL);
    if (left.status < 0 || right.status < 0) atomicAdd(counts + InvalidResult, 1ULL);
}

__global__ void gatherScoreDifferences(const int *indices, int size,
                                      const Candidate *candidates,
                                      const Result *reference, const Result *observed,
                                      ScoreDifference *output) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    const int source = indices[index];
    output[index] = {candidates[source], reference[source], observed[source]};
}

} // namespace

ScoreImpactBatch collectScoreImpact(const Candidate *candidates, int size,
                                   const Result *reference, const Result *observed) {
    ScoreImpactBatch output;
    output.candidates = size;
    if (!size) return output;
    Buffer<unsigned long long> counts(ImpactCountSize);
    check(cudaMemset(counts.p, 0, ImpactCountSize * sizeof(unsigned long long)));
    countScoreImpact<<<(size + 127) / 128, 128>>>(reference, observed, size, counts.p);
    Buffer<int> indices(size), selectedCount(1);
    cub::CountingInputIterator<int> input(0);
    const KeepScoreDifference keep{reference, observed};
    size_t scratchBytes = 0;
    check(cub::DeviceSelect::If(nullptr, scratchBytes, input, indices.p,
                              selectedCount.p, size, keep));
    Buffer<unsigned char> scratch(scratchBytes);
    check(cub::DeviceSelect::If(scratch.p, scratchBytes, input, indices.p,
                              selectedCount.p, size, keep));
    int selected = 0;
    check(cudaMemcpy(&selected, selectedCount.p, sizeof(int), cudaMemcpyDeviceToHost));
    Buffer<ScoreDifference> differences(selected);
    if (selected)
        gatherScoreDifferences<<<(selected + 127) / 128, 128>>>(indices.p, selected,
            candidates, reference, observed, differences.p);
    synced();
    std::vector<unsigned long long> hostCounts;
    counts.read(hostCounts);
    std::copy(hostCounts.begin(), hostCounts.end(), output.counts.begin());
    differences.read(output.differences);
    return output;
}

ScoreImpactWriter::ScoreImpactWriter(const std::string &directory) {
    summary.exceptions(std::ios::failbit | std::ios::badbit);
    differences.exceptions(std::ios::failbit | std::ios::badbit);
    summary.open(std::filesystem::path(directory) / "score_impact_batches.tsv");
    differences.open(std::filesystem::path(directory) / "candidate_score_changes.tsv");
    summary << "batch\tcandidates\tboth_scored\tcuda_only_scored\trt_only_scored\t"
               "neither_scored\tscore_changed\tinvalid_results\n";
    differences << std::setprecision(17)
        << "batch\tscan_index\tscan_id\tpeptide_id\tprecursor_index\tprecursor_mass\t"
           "precursor_charge\tpeptide\tcuda_status\trt_status\tcuda_score\trt_score\tscore_delta\n";
}

void ScoreImpactWriter::writeBatch(const ScoreImpactBatch &batch) {
    ++batchIndex;
    summary << batchIndex << '\t' << batch.candidates;
    for (const auto value : batch.counts) summary << '\t' << value;
    summary << '\n';
    summary.flush();
}

void ScoreImpactWriter::writeDifference(const ScoreDifference &difference, int scanId,
                                       double precursorMass, const std::string &peptide) {
    const auto &candidate = difference.candidate;
    const auto &left = difference.reference;
    const auto &right = difference.observed;
    differences << batchIndex << '\t' << candidate.scanId << '\t' << scanId << '\t'
        << candidate.peptideId << '\t' << candidate.precursorId << '\t' << precursorMass << '\t'
        << candidate.charge << '\t' << peptide << '\t' << left.status << '\t' << right.status << '\t';
    if (left.status == ResultScored) differences << left.score;
    differences << '\t';
    if (right.status == ResultScored) differences << right.score;
    differences << '\t';
    if (left.status == ResultScored && right.status == ResultScored)
        differences << right.score - left.score;
    differences << '\n';
}

} // namespace mvh_cuda
