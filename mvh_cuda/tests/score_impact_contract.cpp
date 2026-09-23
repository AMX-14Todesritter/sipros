#include "score_impact.h"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace mvh_cuda;
    try {
        const std::vector<Candidate> candidates{{0,0,0,2},{1,0,0,2},{2,0,0,2},{3,0,0,2},{4,0,0,2}};
        // Same score with a changed matched count must not be a score error.
        const std::vector<Result> reference{{10,ResultScored,8,6},{10,ResultScored,8,6},
            {10,ResultScored,8,6},{0,ResultInsufficient,8,3},{0,ResultSkipped,0,0}};
        const std::vector<Result> observed{{9,ResultScored,8,5},{10,ResultScored,8,5},
            {0,ResultInsufficient,8,3},{2,ResultScored,8,5},{0,ResultSkipped,0,0}};
        Buffer<Candidate> ids(candidates);
        Buffer<Result> left(reference), right(observed);
        const auto impact = collectScoreImpact(ids.p, candidates.size(), left.p, right.p);
        const std::array<unsigned long long, ImpactCountSize> expected{2,1,1,1,1,0};
        if (impact.counts != expected || impact.candidates != 5 || impact.differences.size() != 3)
            throw std::runtime_error("score-impact counters or selection mismatch");
        for (int index = 0; index < 3; ++index)
            if (impact.differences[index].candidate.peptideId != (index == 0 ? 0 : index + 1))
                throw std::runtime_error("score-impact differences lost candidate identity/order");
        const auto empty = collectScoreImpact(nullptr, 0, nullptr, nullptr);
        if (empty.candidates || !empty.differences.empty())
            throw std::runtime_error("empty score-impact batch failed");
        std::cout << "PASS: score deltas, eligibility transitions, unchanged score despite match-count change, empty batch\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
