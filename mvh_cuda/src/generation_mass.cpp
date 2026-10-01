#include "generation_types.h"
#include "averagine.h"
#include <memory>
namespace mvh_cuda {
namespace { std::unique_ptr<averagine> model; }
void resetGenerationMass() { model.reset(); }
double generationMass(const GenerationAtoms& atoms) {
    if (!model) model = std::make_unique<averagine>();
    auto& a = *model;
    std::copy(atoms.count, atoms.count+6, a.pepAtomCounts.begin());
    std::vector<double> weights = {*a.C13Abundance, *a.H2Abundance, *a.O17Abundance + *a.O18Abundance,
                                  *a.N15Abundance, *a.PfakeAbundance, *a.S33Abundance + *a.S34Abundance + *a.S36Abundance};
    for (size_t i=0;i<6;++i) weights[i] *= a.pepAtomCounts[i];
    double oxygen = a.weighted_mean({*a.O17Mass-*a.O16Mass, *a.O18Mass-*a.O17Mass},
                                   {*a.O17Abundance,*a.O18Abundance});
    double sulfur = a.weighted_mean({*a.S33Mass-*a.S32Mass,*a.S34Mass-*a.S33Mass,(*a.S36Mass-*a.S34Mass)/2.0},
                                   {*a.S33Abundance,*a.S34Abundance,*a.S36Abundance*2});
    double neutron = a.weighted_mean({*a.C13Mass-*a.C12Mass,*a.H2Mass-*a.H1Mass,oxygen,
                                     *a.N15Mass-*a.N14Mass,*a.PfakeMass-*a.PfakeLowMass,sulfur}, weights);
    double base = a.pepAtomCounts[0]*(*a.C12Mass) + a.pepAtomCounts[1]*(*a.H1Mass) +
                  a.pepAtomCounts[2]*(*a.O16Mass) + a.pepAtomCounts[3]*(*a.N14Mass) +
                  a.pepAtomCounts[4]*(*a.PfakeLowMass) + a.pepAtomCounts[5]*(*a.S32Mass);
    return a.estimatePrecursorMassbyNP(base,a.pepAtomCounts[a.SIPatomIX],neutron);
}
}
