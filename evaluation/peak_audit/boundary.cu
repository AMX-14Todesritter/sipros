#include "bridge.h"
#include "scoring.cuh"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
struct Test {const char* name;std::vector<double> peaks;std::vector<int> classes;double query;};
int main(int argc,char**argv){
 try{
  if(argc!=2)throw std::runtime_error("Expected new output directory");std::filesystem::path out(argv[1]);std::filesystem::create_directories(out);
  std::ofstream report(out/"boundary_results.tsv"),examples(out/"boundary_examples.tsv"),casesOut(out/"boundary_cases.tsv");
  report<<"tolerance\tbackend\tqueries\toracle_hits\toracle_misses\tpeak_identity_mismatches\tclass_mismatches\tactual_hits\tactual_misses\n";
  examples<<"tolerance\tbackend\tcase\tquery\toracle_peak\tactual_peak\toracle_class\tactual_class\toracle_mz\tactual_mz\n";
  casesOut<<"tolerance\tcase\tquery\tpeak_id\tmz\tclass\n";examples<<std::setprecision(17);casesOut<<std::setprecision(17);report<<std::setprecision(17);
  for(double tol:{0.125,0.01}){
   double q=100.0,edge=q+tol;std::vector<Test> tests{
    {"exact",{q},{1},q},{"positive_boundary",{q},{2},edge},{"negative_boundary",{q},{2},q-tol},
    {"just_outside",{q},{3},std::nextafter(edge,std::numeric_limits<double>::infinity())},
    {"just_inside",{q},{3},std::nextafter(edge,-std::numeric_limits<double>::infinity())},
    {"equal_distance_tie",{q-tol*.5,q+tol*.5},{1,2},q},
    {"duplicate_mz",{q,q},{1,3},q},{"multiple_valid",{q-tol*.75,q+tol*.25},{3,1},q},
    {"no_hit",{q},{1},q+2*tol},{"empty",{},{},q},{"close_peaks",{q,q+tol*.001},{2,1},q+tol*.0009},
    {"class_zero",{q},{0},q},{"large_mz",{8192.123456},{2},8192.123456+tol*.5},
    {"integer_boundary",{q-tol*.25},{2},q+tol*.25},{"float_sensitive",{100.075691},{2},100.07569046688}};
   using namespace mvh_cuda;Config cfg{};cfg.classes=3;cfg.minMatched=1;cfg.fragmentTolerance=tol;
   std::vector<Scan> hs;std::vector<double> hp,ions;std::vector<int> hc,valid;std::vector<uint64_t> offsets;std::vector<Candidate> candidates;std::vector<PeptideInput> peptides;
   for(size_t i=0;i<tests.size();++i){const auto&t=tests[i];Scan scan{};scan.peakOffset=hp.size();scan.peaks=t.peaks.size();scan.lower=0;scan.upper=10000;scan.totalBins=1000;
    scan.lowest=t.peaks.empty()?0:int(t.peaks.front());scan.highest=t.peaks.empty()?0:int(t.peaks.back());
    for(int c=0;c<3;++c)scan.counts[c]=100;scan.counts[3]=700;
    hs.push_back(scan);hp.insert(hp.end(),t.peaks.begin(),t.peaks.end());hc.insert(hc.end(),t.classes.begin(),t.classes.end());
    ions.push_back(t.query);candidates.push_back({int(i),0,int(i),1});peptides.push_back({0,int(i)});offsets.push_back(i);offsets.push_back(i);valid.push_back(0);valid.push_back(1);
    for(size_t j=0;j<t.peaks.size();++j)casesOut<<tol<<'\t'<<t.name<<'\t'<<t.query<<'\t'<<j<<'\t'<<t.peaks[j]<<'\t'<<t.classes[j]<<'\n';
   }offsets.push_back(ions.size());
   std::vector<short> hub;
   for(auto&scan:hs){scan.hubOffset=hub.size();for(int bin=scan.lowest;bin<=scan.highest;++bin){int first=-1,last=-1;for(int j=0;j<scan.peaks;++j)if(int(hp[scan.peakOffset+j])==bin){if(first<0)first=j;last=j+1;}hub.push_back(first);hub.push_back(last);}}
   std::vector<double> table(1001);for(int i=1;i<=1000;++i)table[i]=table[i-1]+std::log(double(i));
   Buffer<Scan> ds(hs);Buffer<double> dp(hp),di(ions),dt(table);Buffer<int> dc(hc),dv(valid);Buffer<uint64_t> dor(offsets);Buffer<short> dh(hub);
   Buffer<Candidate> dcan(candidates);Buffer<PeptideInput> dpep(peptides);Buffer<char> text(std::vector<char>{'[','A',']',0});Buffer<Result> result(tests.size());Buffer<PeakAuditState> audit(1);
   cfg.peakAudit=audit.p;
   for(const std::string backend:{"cuda","rt-instanced","rt-custom"}){
    check(cudaMemset(audit.p,0,sizeof(PeakAuditState)));
    if(backend=="cuda")ScoreSequenceVsSpectrum<<<1,128>>>(ds.p,dcan.p,tests.size(),dpep.p,text.p,dp.p,dc.p,dh.p,dt.p,result.p,cfg,dor.p,dv.p,di.p,2);
    else {mvh_rt_gpu::reset();mvh_rt_gpu::prepare(hs,ds.p,dp.p,dc.p,hp.size(),backend=="rt-custom"?mvh_rt_gpu::GeometryKind::Spheres:mvh_rt_gpu::GeometryKind::InstancedTriangles,cfg);
     mvh_rt_gpu::Params p{};p.scans=ds.p;p.candidates=dcan.p;p.size=tests.size();p.peptides=dpep.p;p.texts=text.p;p.peaks=dp.p;p.classes=dc.p;p.hub=dh.p;p.lnTable=dt.p;p.results=result.p;p.cfg=cfg;p.ionOffsets=dor.p;p.ionValid=dv.p;p.cachedIons=di.p;p.chargeStride=2;mvh_rt_gpu::launch(p);}
    synced();std::vector<PeakAuditState> read;audit.read(read);unsigned long long sums[7]{};for(auto&shard:read[0].shards)for(int k=0;k<7;++k)sums[k]+=shard[k];
    report<<tol<<'\t'<<backend;for(auto n:sums)report<<'\t'<<n;report<<'\n';
    for(unsigned long long j=0;j<std::min(read[0].exampleCount,128ULL);++j){auto&e=read[0].examples[j];examples<<tol<<'\t'<<backend<<'\t'<<tests[e.scanIndex].name<<'\t'<<e.mz<<'\t'<<e.expectedPeak<<'\t'<<e.actualPeak<<'\t'<<e.expectedClass<<'\t'<<e.actualClass<<'\t'<<e.expectedMz<<'\t'<<e.actualMz<<'\n';}
    if(backend=="cuda"&&(sums[3]||sums[4]))throw std::runtime_error("CUDA boundary results differ from independent double oracle");
   }mvh_rt_gpu::reset();
  }return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
