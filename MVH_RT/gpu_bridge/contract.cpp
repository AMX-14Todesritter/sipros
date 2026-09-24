#include "bridge.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc,char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "triangle";
    const bool custom = mode == "custom";
    const auto geometry = custom ? mvh_rt_gpu::GeometryKind::Spheres :
        mode == "instanced" ? mvh_rt_gpu::GeometryKind::InstancedTriangles : mvh_rt_gpu::GeometryKind::Triangles;
    using namespace mvh_cuda;
    try {
        Config cfg{}; cfg.classes=3; cfg.minMatched=1; cfg.fragmentTolerance=0.01;
        Scan scan{}; scan.peaks=1; scan.lower=0; scan.upper=1000;
        scan.counts[0]=1; scan.counts[3]=99; scan.totalBins=100;
        std::vector<Scan> hostScans{scan};
        std::vector<double> hostPeaks{100.075691};
        std::vector<int> hostClasses{1};
        Buffer<Scan> scans(hostScans);
        Buffer<double> peaks(hostPeaks);
        Buffer<int> classes(hostClasses);
        Buffer<Candidate> candidates(std::vector<Candidate>{{0,0,0,1},{1,0,0,1},{2,0,0,1}});
        Buffer<PeptideInput> peptides(std::vector<PeptideInput>{{0,0},{0,1},{0,2}});
        Buffer<char> texts(std::vector<char>{'[','A','A',']',0});
        Buffer<uint64_t> offsets(std::vector<uint64_t>{0,0,1,1,2,2,3});
        Buffer<int> valid(std::vector<int>{0,1,0,1,0,1});
        Buffer<double> ions(std::vector<double>{100.07569046688,100.074691,100.076691});
        std::vector<double> table(101); for(int i=1;i<=100;++i)table[i]=table[i-1]+std::log(double(i));
        Buffer<double> lnTable(table);
        Buffer<Result> results(3);
        mvh_rt_gpu::Params p{};
        p.scans=scans.p;p.candidates=candidates.p;p.peptides=peptides.p;p.texts=texts.p;
        p.peaks=peaks.p;p.classes=classes.p;p.lnTable=lnTable.p;p.results=results.p;
        p.cfg=cfg;p.ionOffsets=offsets.p;p.ionValid=valid.p;p.cachedIons=ions.p;
        p.size=3;p.chargeStride=2;
        for(int repeat=0;repeat<2;++repeat) {
            mvh_rt_gpu::prepare(hostScans,scans.p,peaks.p,classes.p,peaks.n,geometry,cfg);
            mvh_rt_gpu::launch(p); synced();
            std::vector<Result> got;results.read(got);
            if(got[0].predicted!=1 || got[0].matched!=(custom ? 1 : 0) ||
               got[0].status!=(custom ? ResultScored : ResultInsufficient))
                throw std::runtime_error("Float-zero query did not follow the selected backend contract");
            for(int i=1;i<3;++i)
                if(got[i].predicted!=1 || got[i].matched!=1 || got[i].status!=ResultScored)
                    throw std::runtime_error("Nonzero query failed");
        }
        // Exercise the uncached device theory path using the same score function.
        cfg.minLength=2;cfg.bIon=1;cfg.yIon=1;cfg.proton=1;cfg.yWater=18;
        cfg.mass['A']=71;cfg.fragmentTolerance=0.01;
        mvh_rt_gpu::reset();hostPeaks={72.001};
        check(cudaMemcpy(peaks.p,hostPeaks.data(),sizeof(double),cudaMemcpyHostToDevice));
        mvh_rt_gpu::prepare(hostScans,scans.p,peaks.p,classes.p,peaks.n,geometry,cfg);
        p.cfg=cfg;p.size=1;p.chargeStride=0;p.ionOffsets=nullptr;p.ionValid=nullptr;p.cachedIons=nullptr;
        mvh_rt_gpu::launch(p);synced();
        std::vector<Result> got;results.read(got);
        if(got[0].predicted!=3 || got[0].matched!=1 || got[0].status!=ResultScored)
            throw std::runtime_error("Uncached theory/RT scoring failed");
        mvh_rt_gpu::reset();
        std::cout << "PASS: cached/direct theory, reused GAS, backend-specific zero-distance behavior, nonzero matches\n";
        return 0;
    } catch(const std::exception &e) {
        mvh_rt_gpu::reset(); std::cerr<<e.what()<<'\n';return 1;
    }
}
