#pragma once
#include "scoring.cuh"

namespace mvh_cuda {
// Cache once per (peptide object, precursor charge) within a bounded batch.
// Object IDs intentionally avoid changing sequence/protein duplicate semantics.
struct CountIons {
    uint64_t count = 0;
    __device__ void add(double) { ++count; }
};
struct StoreIons {
    double *output;
    uint64_t count = 0;
    __device__ void add(double mz) { output[count++] = mz; }
};
__global__ void markTheoreticalKeys(const Candidate *candidates, int size,
                                    int stride, int *active) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && candidates[i].charge < stride) atomicExch(active + uint64_t(candidates[i].peptideId) * stride
                           + candidates[i].charge, 1);
}
__global__ void countTheoreticalIons(const PeptideInput *peptides, const char *texts,
                                    int size, int stride, int *active,
                                    uint64_t *counts, Config config) {
    const int key = blockIdx.x * blockDim.x + threadIdx.x;
    if (key >= size) return;
    counts[key] = 0;
    if (!active[key]) return;
    CountIons ions;
    if (!CalculateSequenceIons(texts + peptides[key / stride].text,
                               key % stride, config, ions)) {
        active[key] = -1;
        return;
    }
    counts[key] = ions.count;
}
__global__ void generateTheoreticalIons(const PeptideInput *peptides, const char *texts,
                                       int size, int stride, const int *active,
                                       const uint64_t *offsets, double *output,
                                       Config config) {
    const int key = blockIdx.x * blockDim.x + threadIdx.x;
    if (key >= size || active[key] <= 0) return;
    StoreIons ions{output + offsets[key]};
    CalculateSequenceIons(texts + peptides[key / stride].text,
                          key % stride, config, ions);
}
}
