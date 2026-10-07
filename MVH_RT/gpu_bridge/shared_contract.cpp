#include "bridge.h"
#include "shared_theory.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

std::vector<mvh_cuda::Precursor> precursorMetadata(const std::vector<mvh_cuda::Scan>& scans) {
    std::vector<mvh_cuda::Precursor> result;
    for(size_t i=0;i<scans.size();++i) {
        result.push_back({1000.0+double(scans.size()-i),int(i),2});
        result.push_back({2000.0+double(i),int(i),3});
    }
    return result;
}

void check(int k) {
    using namespace mvh_cuda;
    mvh_rt_gpu::reset(); mvh_rt_gpu::setScanGroupSize(k);
    Config cfg{};cfg.classes=3;cfg.minMatched=1;cfg.fragmentTolerance=0.125;
    std::vector<Scan> hs(35);
    std::vector<double> hp;
    std::vector<int> hc;
    // Same query can resolve scans at different class stages. Irrelevant scans
    // have class-3 exact hits, and must not affect related scans' outcomes.
    for (int s=0;s<35;++s) {
        auto &scan=hs[s];scan.peakOffset=hp.size();scan.lower=0;scan.upper=1000;
        scan.totalBins=1000;scan.counts[0]=32;scan.counts[1]=64;scan.counts[2]=128;scan.counts[3]=776;
        std::vector<double> p;std::vector<int> c;
        switch(s%7) {
        case 0: p={100.0,100.0625,100.03125};c={1,3,2};break;
        case 1: p={100.0,100.0625};c={0,2};break;
        case 2: p={100.0625};c={1};break;
        case 3: p={100.125,100.25};c={3,2};break;
        case 4: p={99.96875,100.03125,100.03125};c={3,3,3};break;
        case 5: p={100.0};c={3};break;
        case 6: p={100.0};c={3};scan.skip=1;break;
        }
        if (s==8) scan.lower=100.5; // Candidate outside its valid m/z range.
        scan.peaks=p.size();hp.insert(hp.end(),p.begin(),p.end());hc.insert(hc.end(),c.begin(),c.end());
    }
    std::vector<Candidate> candidates;
    for (int s=34;s>=0;--s) if(s%7!=5) candidates.push_back({0,s,s,1});
    candidates.push_back({0,99,0,1}); // Repeated precursor hypothesis.
    candidates.push_back({1,100,1,1}); // Same group, distinct theory.
    Buffer<Scan> scans(hs);Buffer<double> peaks(hp);Buffer<int> classes(hc);
    Buffer<Candidate> dc(candidates);Buffer<PeptideInput> peptides(std::vector<PeptideInput>{{0,0},{0,1}});
    Buffer<uint64_t> offsets(std::vector<uint64_t>{0,0,2,2,3});
    Buffer<int> valid(std::vector<int>{0,1,0,1});
    // Two identical ions may both use the same experimental peak.
    Buffer<double> ions(std::vector<double>{100.0,100.0,200.0});
    std::vector<double> table(1001);for(int i=1;i<=1000;++i)table[i]=table[i-1]+std::log(double(i));
    Buffer<double> lt(table);Buffer<Result> results(candidates.size());
    Buffer<unsigned> selected(candidates.size()*2);
    mvh_rt_gpu::Params p{};p.scans=scans.p;p.peaks=peaks.p;p.classes=classes.p;
    p.candidates=dc.p;p.peptides=peptides.p;p.ionOffsets=offsets.p;p.ionValid=valid.p;
    p.selectedPeaks=selected.p;p.selectedPeakStride=2;
    p.cachedIons=ions.p;p.chargeStride=2;p.results=results.p;p.lnTable=lt.p;p.cfg=cfg;p.size=candidates.size();
    for(int repeat=0;repeat<2;++repeat) {
        mvh_rt_gpu::prepare(hs,scans.p,peaks.p,classes.p,hp.size(),mvh_rt_gpu::GeometryKind::Spheres, cfg, precursorMetadata(hs));
        mvh_rt_gpu::launch(p);synced();std::vector<Result> got;results.read(got);
        std::vector<unsigned> winners;selected.read(winners);
        for(size_t i=0;i<candidates.size();++i) {
            const auto &c=candidates[i];const auto &scan=hs[c.scanId];Result expected{};
            if(!scan.skip) {
                int keys[4]={};int predicted=0,matched=0;
                for(int n=0;n<(c.peptideId?1:2);++n) {
                    double q=c.peptideId?200.0:100.0;
                    if(q<scan.lower || q>scan.upper)continue;
                    ++predicted;int best=0;unsigned bestPeak=~0u;double distance=cfg.fragmentTolerance;
                    for(int j=0;j<scan.peaks;++j) {
                        size_t peak=scan.peakOffset+j;double d=std::abs(q-hp[peak]);int cls=hc[peak];
                        if(cls>0 && d<cfg.fragmentTolerance && (cls>best || (cls==best && (d<distance || (d==distance && peak<bestPeak))))) {
                            best=cls;distance=d;bestPeak=peak;
                        }
                    }
                    if(winners[i*2+n]!=bestPeak)throw std::runtime_error("Wrong selected peak identity");
                    if(best){++keys[best-1];++matched;}else ++keys[3];
                }
                expected.predicted=predicted;expected.matched=matched;expected.status=ResultInsufficient;
                if(matched>=cfg.minMatched) {
                    double value=0;for(int cls=0;cls<4;++cls) value+=(table[scan.counts[cls]]-table[scan.counts[cls]-keys[cls]])-table[keys[cls]];
                    value-=(table[scan.totalBins]-table[scan.totalBins-predicted])-table[predicted];
                    expected.score=-value;expected.status=ResultScored;
                }
            }
            const auto &r=got[i];
            if(r.status!=expected.status || r.predicted!=expected.predicted || r.matched!=expected.matched || std::abs(r.score-expected.score)>1e-12)
                throw std::runtime_error("Shared any-hit differs from exhaustive reference: K="+std::to_string(k)+" scan="+std::to_string(c.scanId));
        }
    }
    mvh_rt_gpu::reset();
}
// More than one launch: distinct peptide objects must retain global candidate
// mappings, including an invalid final object in the second launch.
void checkLaunchBoundary() {
    using namespace mvh_cuda;
    mvh_rt_gpu::reset();mvh_rt_gpu::setScanGroupSize(8);
    constexpr int count=65537;
    Config cfg{};cfg.classes=3;cfg.minMatched=1;cfg.fragmentTolerance=0.125;
    Scan scan{};scan.peaks=1;scan.lower=0;scan.upper=1000;scan.totalBins=1000;
    scan.counts[0]=1;scan.counts[3]=999;
    std::vector<Scan> hs{scan};Buffer<Scan> scans(hs);
    Buffer<double> peaks(std::vector<double>{100.0});Buffer<int> classes(std::vector<int>{1});
    std::vector<Candidate> cs;std::vector<PeptideInput> ps;
    std::vector<uint64_t> os;std::vector<int> vs;
    for(int i=0;i<count;++i) {
        cs.push_back({i,i,0,i==count-1?0:1});ps.push_back({0,i});
        os.push_back(i);os.push_back(i);vs.push_back(0);vs.push_back(i==count-1?0:1);
    }
    os.push_back(count);
    Buffer<Candidate> candidates(cs);Buffer<PeptideInput> peptides(ps);
    Buffer<uint64_t> offsets(os);Buffer<int> valid(vs);
    Buffer<double> ions(std::vector<double>(count,100.0));
    std::vector<double> table(1001);for(int i=1;i<=1000;++i)table[i]=table[i-1]+std::log(double(i));
    Buffer<double> lt(table);Buffer<Result> results(count);
    mvh_rt_gpu::Params p{};p.scans=scans.p;p.peaks=peaks.p;p.classes=classes.p;p.candidates=candidates.p;
    p.peptides=peptides.p;p.ionOffsets=offsets.p;p.ionValid=valid.p;p.cachedIons=ions.p;
    p.chargeStride=2;p.results=results.p;p.lnTable=lt.p;p.cfg=cfg;p.size=count;
    mvh_rt_gpu::prepare(hs,scans.p,peaks.p,classes.p,1,mvh_rt_gpu::GeometryKind::Spheres, cfg, precursorMetadata(hs));
    mvh_rt_gpu::launch(p);synced();std::vector<Result> got;results.read(got);
    for(int i=0;i<count;++i)
        if(i==count-1 ? got[i].status!=-1 :
           (got[i].status!=ResultScored || got[i].predicted!=1 || got[i].matched!=1))
            throw std::runtime_error("Global result mapping changed at OptiX launch boundary");
    mvh_rt_gpu::reset();
}
void checkMassGroups() {
    using namespace mvh_cuda;
    std::vector<Scan> scans(5);
    Buffer<double> peaks(0);Buffer<int> classes(0);
    std::vector<Precursor> ps{{500,0,2},{300,1,2},{100,2,2},{300,3,2},{900,4,2},{50,4,3}};
    auto grouped=mvh_rt_gpu::prepareGroupedGeometry(scans,peaks.p,classes.p,ps,2,0.01);
    std::vector<int> got;grouped->scanGroups.read(got);
    if(got!=std::vector<int>({2,1,0,1,0}))throw std::runtime_error("Precursor mass grouping is not stable/minimum-based");
}
void checkHitBudget() {
    using namespace mvh_cuda;
    mvh_rt_gpu::reset();mvh_rt_gpu::setScanGroupSize(1);mvh_rt_gpu::setWorkspaceMiB(16);
    Config cfg{};cfg.classes=3;cfg.minMatched=1;cfg.fragmentTolerance=0.01;
    Scan s{};s.peaks=10000;s.lower=0;s.upper=1000;s.totalBins=100000;s.counts[0]=10000;s.counts[3]=90000;
    std::vector<Scan> hs{s};Buffer<Scan> scans(hs);
    Buffer<double> peaks(std::vector<double>(s.peaks,100.0));Buffer<int> classes(std::vector<int>(s.peaks,1));
    Buffer<Candidate> cs(std::vector<Candidate>{{0,0,0,1}});Buffer<PeptideInput> peptides(std::vector<PeptideInput>{{0,0}});
    Buffer<uint64_t> offsets(std::vector<uint64_t>{0,0,200});Buffer<int> valid(std::vector<int>{0,1});
    Buffer<double> ions(std::vector<double>(200,100.0));Buffer<Result> results(1);
    std::vector<double> table(100001);for(int i=1;i<=100000;++i)table[i]=table[i-1]+std::log(double(i));Buffer<double> lt(table);
    mvh_rt_gpu::Params p{};p.scans=scans.p;p.peaks=peaks.p;p.classes=classes.p;p.candidates=cs.p;p.peptides=peptides.p;
    p.ionOffsets=offsets.p;p.ionValid=valid.p;p.cachedIons=ions.p;p.chargeStride=2;p.results=results.p;p.lnTable=lt.p;p.cfg=cfg;p.size=1;
    mvh_rt_gpu::prepare(hs,scans.p,peaks.p,classes.p,peaks.n,mvh_rt_gpu::GeometryKind::Spheres,cfg,precursorMetadata(hs));
    bool rejected=false;
    try{mvh_rt_gpu::launch(p);}catch(const std::runtime_error &e){rejected=std::string(e.what()).find("hit list exceeds")!=std::string::npos;}
    if(!rejected)throw std::runtime_error("Oversized hit list was not rejected");
    mvh_rt_gpu::setWorkspaceMiB(64);mvh_rt_gpu::launch(p);synced();std::vector<Result> got;results.read(got);
    if(got[0].status!=ResultScored||got[0].matched!=200||got[0].predicted!=200)throw std::runtime_error("Hit budget retry lost matches");
    mvh_rt_gpu::reset();mvh_rt_gpu::setWorkspaceMiB(512);
}
int main() {
    try {for(int k:{1,2,8,32,128})check(k);checkLaunchBoundary();checkMassGroups();checkHitBudget();std::cout<<"PASS: shared scans, unrelated hits, class priority, strict boundary, duplicated ions/candidates, skip/ranges and reuse\n";return 0;}
    catch(const std::exception &e){mvh_rt_gpu::reset();std::cerr<<e.what()<<'\n';return 1;}
}
