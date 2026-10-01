#pragma once
#include <cstdint>
#include <array>
namespace mvh_cuda {
constexpr int GenerationMaxLength = 128;
constexpr int GenerationSites = GenerationMaxLength + 2;
struct GenerationProtein { uint64_t offset; int length; };
struct GenerationBase { int protein, begin, length; };
struct GenerationAtoms { int count[6]; };
struct GenerationMod { double mass; char symbol; };
struct GenerationConfig {
    int minimum, maximum, missed, maxPtm, removeM;
    int after[256], before[256], modBegin[256], modCount[256];
    int atoms[256][6], termini[6];
};
struct GenerationResult { double mass; int base; };
// Host floating-point compatibility boundary: atom counting is on the GPU;
// the existing fast-math CPU evaluation order is preserved for precursor mass.
double generationMass(const GenerationAtoms& atoms);
void resetGenerationMass();
}
