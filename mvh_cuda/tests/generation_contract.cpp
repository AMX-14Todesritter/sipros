#include "engine.h"
#include "peptide_generation.h"
#include "proNovoConfig.h"
#include "peptide.h"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=3 || !ProNovoConfig::setFilename(argv[1]))return 2;
        ProNovoConfig::setFASTAfilename(argv[2]);
        mvh_cuda::setVerification(true);
        mvh_cuda::setPeptideGeneration("cuda");
        mvh_cuda::SearchPeptideGenerator generator(false);
        generator.loadDatabase();generator.getFirstProtein();
        Peptide peptide;uint64_t count=0;
        while(generator.getNextPeptide(&peptide))++count;
        std::cout<<"PASS: generated="<<count<<"; exact CPU sequence/mass/order/metadata parity\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
