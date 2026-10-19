#include "../matching/fusion.cuh"
#include "../cuda/scoring.cuh"
#include <iostream>
#include <cmath>
using namespace mvh_cuda;
int main(){try{
 Config cfg{};cfg.classes=2;cfg.minLength=2;cfg.minMatched=2;cfg.bIon=cfg.yIon=1;
 cfg.proton=1;cfg.yWater=18;cfg.fragmentTolerance=.01;cfg.mass['A']=100;
 Scan s{};s.peaks=3;s.lowest=101;s.highest=219;s.lower=0;s.upper=300;s.totalBins=100;
 s.counts[0]=10;s.counts[1]=20;s.counts[2]=70;
 std::vector<double> peaks{101,119,219},table(101);std::vector<int> classes{1,0,2};
 for(int i=1;i<=100;++i)table[i]=table[i-1]+std::log(double(i));
 std::vector<short> hub(2*(219-101+1),-1);for(int i=0;i<3;++i){int b=int(peaks[i])-101;hub[2*b]=i;hub[2*b+1]=i+1;}
 Candidate c{0,0,0,2};PeptideInput p{};std::vector<char> text{'[','A','A',']',0};
 Buffer<Scan> ds(1);Buffer<Candidate> dg(std::vector<Candidate>{c});Buffer<PeptideInput> dp(std::vector<PeptideInput>{p});
 Buffer<char> dt(text);Buffer<double> dpeaks(peaks),dl(table);Buffer<int> dc(classes);Buffer<short> dh(hub);Buffer<Result> dr(1);
 auto require=[](bool ok){if(!ok)throw std::runtime_error("fusion scoring contract failed");};
 // Regression: a built-in sphere may hit just outside the strict double window.
 {Scan boundary{};boundary.peaks=1;double peak=608.84259;double mz=608.83258996688005;
  require(mvh_matching::validateRtPeak(0,mz,boundary,&peak,.01)==-1);
  require(mvh_matching::validateRtPeak(0,peak,boundary,&peak,.01)==0);
  require(mvh_matching::validateRtPeak(-1,mz,boundary,&peak,.01)==-1);}
 for(int scenario=0;scenario<5;++scenario){
  auto scan=s;auto candidate=c;auto config=cfg;
  if(scenario==1)candidate.charge=0;if(scenario==2)scan.skip=1;if(scenario==3)config.minMatched=3;if(scenario==4)scan.peaks=0;
  ds.upload(std::vector<Scan>{scan});dg.upload(std::vector<Candidate>{candidate});
  auto out=mvh_matching::evaluateFusion<2>(scan,candidate,&p,text.data(),classes.data(),table.data(),config,nullptr,0,0,1,
   mvh_matching::BinLookup{scan,peaks.data(),hub.data(),config.fragmentTolerance});
  ScoreSequenceVsSpectrum<<<1,1>>>(ds.p,dg.p,1,dp.p,dt.p,dpeaks.p,dc.p,dh.p,dl.p,dr.p,config,nullptr,nullptr,nullptr,0);synced();std::vector<Result> original;dr.read(original);
  require(original[0].status==out.result.status && original[0].predicted==out.result.predicted && original[0].matched==out.result.matched && std::abs(original[0].score-out.result.score)<1e-12);
  if(scenario==0){double expected=-std::log((10.*20.*70.)/(100.*99.*98./6.));require(out.summary.peakHits==3 && out.summary.bins[0]==1 && out.summary.bins[1]==1 && out.summary.bins[2]==1 && out.result.matched==2 && out.result.predicted==3 && out.result.status==ResultScored && std::abs(out.result.score-expected)<1e-12);}
  if(scenario==1)require(out.result.status==-1);
  if(scenario==2)require(out.result.status==ResultSkipped);
  if(scenario==3)require(out.result.status==ResultInsufficient && out.result.score==0);
  if(scenario==4)require(out.result.matched==0 && out.summary.bins[2]==3);
 }
 std::cout<<"FUSION ORIGINAL SCORING CONTRACT PASSED\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
