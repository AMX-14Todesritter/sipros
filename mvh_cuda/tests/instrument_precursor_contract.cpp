#include "mvh_scan_vector.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc != 3 || !ProNovoConfig::setFilename(argv[1])) return 2;
        const auto path = std::filesystem::path(argv[2]) / "instrument_precursor_contract.FT2";
        struct Cleanup { std::filesystem::path p; ~Cleanup(){std::filesystem::remove(p);} } cleanup{path};
        auto write = [&](const std::string &s){std::ofstream f(path); f << s;};
        write("S\t100\t435.853546\t12\nZ\t3\t9999\t20\t100\nZ\t9\t2000\n100\t200\n"
              "S\t101\t404.697021\t12\nZ\t2\t9999\t4\t100\n110\t300\n");
        {
            MvhScanVector scans(path.string(), argv[2], argv[1], false);
            if (!scans.loadMassData() || scans.vpAllMS2Scans.size()!=2) throw std::runtime_error("Scan count");
            for (int i=0;i<2;++i) {
                auto *s=scans.vpAllMS2Scans[i]; const int z=i==0?3:2;
                const double mz=i==0?435.853546:404.697021;
                if (s->iParentChargeState!=z || s->dParentMZ!=mz ||
                    std::abs(s->dParentNeutralMass-z*(mz-ProNovoConfig::getProtonMass()))>1e-9 ||
                    !s->iParentChargeStates.empty() || !s->dParentMZs.empty())
                    throw std::runtime_error("Instrument precursor differs");
            }
        }
        for (const auto &zline : {std::string(""), std::string("Z\t0\t999\t3\t435\n"), std::string("Z\tbad\t999\n")}) {
            write("S\t102\t435.853546\t12\n"+zline+"100\t200\n");
            bool rejected=false;
            try {MvhScanVector scans(path.string(),argv[2],argv[1],false); scans.loadMassData();}
            catch(const std::runtime_error&){rejected=true;}
            if(!rejected) throw std::runtime_error("Invalid/missing primary charge accepted");
        }
        std::cout << "PASS instrument S m/z + first Z charge, extras ignored, invalid charge rejected\n";
        return 0;
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
