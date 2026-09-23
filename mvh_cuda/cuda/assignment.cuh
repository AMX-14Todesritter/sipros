#pragma once
#include "types.cuh"

namespace mvh_cuda {
// Inclusive endpoints match the CPU mass filter. Binary search changes the
// lookup cost, not which precursor entries or duplicate hypotheses are retained.
__device__ MassRange GetRangeFromMass(const Precursor *precursors, int size,
                                     double lower, double upper) {
    int first = 0, last = size;
    while (first < last) {
        const int middle = first + (last - first) / 2;
        if (precursors[middle].mass < lower) first = middle + 1;
        else last = middle;
    }
    const int begin = first;
    last = size;
    while (first < last) {
        const int middle = first + (last - first) / 2;
        if (precursors[middle].mass <= upper) first = middle + 1;
        else last = middle;
    }
    return begin < first ? MassRange{begin, first - 1} : MassRange{-1, -1};
}

__global__ void GetAllRangeFromMass(const double *masses, int size,
                                   const Precursor *precursors, int precursorCount,
                                   const MassWindow *windows, int windowCount,
                                   MassRange *ranges, int *rangeCounts,
                                   uint64_t *associationCounts) {
    const int peptide = blockIdx.x * blockDim.x + threadIdx.x;
    if (peptide >= size) return;
    MassRange previous{-1, -1};
    int count = 0;
    uint64_t associations = 0;
    auto *output = ranges + uint64_t(peptide) * windowCount;
    for (int w = 0; w < windowCount; ++w) {
        const auto range = GetRangeFromMass(precursors, precursorCount,
            __dadd_rn(masses[peptide], windows[w].lower),
            __dadd_rn(masses[peptide], windows[w].upper));
        if (range.first < 0) continue;
        if (previous.first < 0) previous = range;
        // Preserve the original strict comparison, including its treatment of
        // windows sharing exactly one precursor entry. Do not deduplicate here.
        else if (previous.last > range.first) previous.last = range.last;
        else {
            output[count++] = previous;
            associations += previous.last - previous.first + 1;
            previous = range;
        }
    }
    if (previous.first >= 0) {
        output[count++] = previous;
        associations += previous.last - previous.first + 1;
    }
    rangeCounts[peptide] = count;
    associationCounts[peptide] = associations;
}

__global__ void assignPeptides2Scans(const MassRange *ranges, const int *rangeCounts,
                                    const uint64_t *offsets, int size, int windowCount,
                                    const Precursor *precursors, Candidate *candidates,
                                    int *scanKeys) {
    const int peptide = blockIdx.x * blockDim.x + threadIdx.x;
    if (peptide >= size) return;
    uint64_t output = offsets[peptide];
    for (int w = 0; w < rangeCounts[peptide]; ++w) {
        const auto range = ranges[uint64_t(peptide) * windowCount + w];
        for (int p = range.first; p <= range.last; ++p, ++output) {
            const auto precursor = precursors[p];
            candidates[output] = {peptide, p, precursor.scanId, precursor.charge};
            scanKeys[output] = precursor.scanId;
        }
    }
}

__global__ void setCandidateRanges(Scan *scans, const Candidate *candidates, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    const int scan = candidates[index].scanId;
    if (!index || candidates[index - 1].scanId != scan)
        scans[scan].candidateOffset = index;
    if (index == size - 1 || candidates[index + 1].scanId != scan) {
        // Find the beginning independently: no cross-block synchronization is
        // assumed between the two writes to this scan's metadata.
        int lower = 0, upper = index;
        while (lower < upper) {
            const int middle = lower + (upper - lower) / 2;
            if (candidates[middle].scanId < scan) lower = middle + 1;
            else upper = middle;
        }
        scans[scan].candidates = index - lower + 1;
    }
}
}
