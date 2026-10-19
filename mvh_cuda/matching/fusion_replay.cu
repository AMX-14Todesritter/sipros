#include "fusion.cuh"
#include "bridge.h"
#include <omp.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <regex>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <memory>
#include <cstring>
#include <limits>
using namespace mvh_cuda;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
using mvh_matching::FusionOutput;
struct Map {
 int fd;void* p=nullptr;size_t bytes;
 Map(fs::path path){bytes=fs::file_size(path);fd=open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("Open failed: "+path.string());
  if(bytes){p=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);if(p==MAP_FAILED)throw std::runtime_error("mmap failed");}}
 ~Map(){if(p && p!=MAP_FAILED)munmap(p,bytes);close(fd);}
 template<class T>const T* data()const{return static_cast<const T*>(p);}
 template<class T>size_t count()const{if(bytes%sizeof(T))throw std::runtime_error("Invalid capture element size");return bytes/sizeof(T);}
};
std::string read(fs::path p){std::ifstream f(p);if(!f)throw std::runtime_error("Missing "+p.string());return {(std::istreambuf_iterator<char>(f)),{}};}
uint64_t field(const std::string& s,const char* key){std::smatch m;if(!std::regex_search(s,m,std::regex(std::string("\"")+key+"\"\\s*:\\s*([0-9]+)")))throw std::runtime_error(std::string("Missing field ")+key);return std::stoull(m[1]);}
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
double median(std::vector<double> v){std::sort(v.begin(),v.end());return v.size()%2?v[v.size()/2]:(v[v.size()/2-1]+v[v.size()/2])/2;}
template<class T>void upload(Buffer<T>& b,const T* p,size_t n){b.resize(std::max(n,size_t(1)));if(n)check(cudaMemcpy(b.p,p,n*sizeof(T),cudaMemcpyHostToDevice));}
struct Events{
 cudaEvent_t a,b;Events(){check(cudaEventCreate(&a));check(cudaEventCreate(&b));}~Events(){cudaEventDestroy(a);cudaEventDestroy(b);}
 template<class F>double time(F f){check(cudaEventRecord(a));f();check(cudaEventRecord(b));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));return ms*.001;}
};
struct Device {
 Buffer<Scan> scans;Buffer<double> peaks,table,mz;Buffer<int> classes;Buffer<short> hub;
 Buffer<PeptideInput> peptides;Buffer<char> texts;Buffer<Candidate> groups;Buffer<uint64_t> offsets;
 Buffer<unsigned char> validity;Buffer<FusionOutput> cudaOut,rtOut;
 ~Device(){mvh_rt_gpu::reset();}
};
__constant__ Config fusionConfig;
template<int Mode>__global__ void fusionKernel(const Scan* scans,const Candidate* groups,int n,
 const PeptideInput* peptides,const char* texts,const double* peaks,const int* classes,const short* hub,
 const double* table,const double* mz,const uint64_t* offsets,uint64_t base,
 const unsigned char* valid,FusionOutput* output){
 const Config& cfg=fusionConfig;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;auto c=groups[i];const auto& s=scans[c.scanId];
 output[i]=mvh_matching::evaluateFusion<Mode>(s,c,peptides,texts,classes,table,cfg,mz,
  offsets[i]-base,offsets[i+1]-base,valid[i],mvh_matching::BinLookup{s,peaks,hub,cfg.fragmentTolerance});
}
template<int Mode>void launch(Device& d,int n,uint64_t base){
 fusionKernel<Mode><<<(n+127)/128,128>>>(d.scans.p,d.groups.p,n,d.peptides.p,d.texts.p,d.peaks.p,
  d.classes.p,d.hub.p,d.table.p,d.mz.p,d.offsets.p,base,d.validity.p,d.cudaOut.p);check(cudaGetLastError());
}
struct AuditSink{
 const Scan& scan;const double* mz;uint64_t pos,end;uint64_t mismatches=0;double maxError=0;
 __host__ __device__ void add(double value){
  if(value<scan.lower || value>scan.upper)return;
  if(pos>=end){++mismatches;++pos;return;}
  const double err=fabs(value-mz[pos]);if(err>maxError)maxError=err;
#ifdef __CUDA_ARCH__
  if(__double_as_longlong(value)!=__double_as_longlong(mz[pos]))++mismatches;
#else
  if(std::memcmp(&value,mz+pos,8))++mismatches;
#endif
  ++pos;
 }
};
bool equal(const FusionOutput& a,const FusionOutput& b,double& maxScoreError){
 bool same=a.summary.peakHits==b.summary.peakHits;
 for(int k=0;k<=MaxClasses;++k)same&=a.summary.bins[k]==b.summary.bins[k];
 same&=a.result.status==b.result.status && a.result.predicted==b.result.predicted && a.result.matched==b.result.matched;
 const double err=std::abs(a.result.score-b.result.score);maxScoreError=std::max(maxScoreError,err);
 return same && std::isfinite(a.result.score) && std::isfinite(b.result.score) && err<=1e-10*std::max(1.,std::abs(a.result.score));
}
uint64_t hashSummary(uint64_t h,const FusionOutput& a){h=(h^a.summary.peakHits)*1099511628211ULL;for(auto v:a.summary.bins)h=(h^v)*1099511628211ULL;return h;}
int main(int argc,char** argv){try{
 fs::path input,output;int threads=8,repeats=5,warmup=1,chunk=1000000,maxChunks=0;std::string backends="cpu,cuda,rt";bool rtPostcheck=true;
 for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::invalid_argument("Missing value");std::string k=argv[i],v=argv[i+1];
  if(k=="--capture")input=v;else if(k=="--output")output=v;else if(k=="--threads")threads=std::stoi(v);
  else if(k=="--repeats")repeats=std::stoi(v);else if(k=="--warmup")warmup=std::stoi(v);
  else if(k=="--chunk-groups")chunk=std::stoi(v);else if(k=="--max-chunks")maxChunks=std::stoi(v);
  else if(k=="--rt-postcheck"){if(v!="on" && v!="off")throw std::invalid_argument("Require --rt-postcheck on|off");rtPostcheck=v=="on";}
  else if(k=="--backends")backends=v;else throw std::invalid_argument("Unknown option: "+k);}
 if(input.empty() || output.empty() || threads<1 || repeats<1 || warmup<0 || chunk<1 || maxChunks<0)throw std::invalid_argument("Invalid replay options");
 bool selected[3]={false,false,false};std::stringstream bs(backends);std::string token;
 while(std::getline(bs,token,',')){if(token=="cpu")selected[0]=true;else if(token=="cuda")selected[1]=true;else if(token=="rt")selected[2]=true;else throw std::invalid_argument("Unknown backend");}
 if(!selected[0]&&!selected[1]&&!selected[2])throw std::invalid_argument("No backends");
 // Validate native-layout extension before opening the output or initializing CUDA.
 auto abi=read(input/"fusion_abi.json");if(field(abi,"version")!=1 || field(abi,"scan_bytes")!=sizeof(Scan) || field(abi,"config_bytes")!=sizeof(Config) || field(abi,"peptide_bytes")!=sizeof(PeptideInput))throw std::runtime_error("Fusion ABI mismatch");
 auto meta=read(input/"COMPLETE.json");
 const uint16_t endian=1;if(*reinterpret_cast<const unsigned char*>(&endian)!=1)throw std::runtime_error("Little endian required");
 Map sm(input/"fusion_scans.bin"),pm(input/"peaks.f64"),cm(input/"classes.i32"),hm(input/"hub.i16"),pepm(input/"fusion_peptides.bin"),tm(input/"fusion_texts.bin"),lm(input/"fusion_ln.f64"),cfgm(input/"fusion_config.bin"),gm(input/"candidates.i32x4"),om(input/"query_offsets.u64"),qm(input/"query_mz.f64"),vm(input/"validity.u8");
 const size_t scanN=sm.count<Scan>(),peakN=pm.count<double>(),pepN=pepm.count<PeptideInput>(),groupN=gm.count<Candidate>(),queryN=qm.count<double>(),tableN=lm.count<double>();
 if(!scanN || !groupN || cfgm.bytes!=sizeof(Config) || cm.count<int>()!=peakN || om.count<uint64_t>()!=groupN+1 || vm.bytes!=groupN)throw std::runtime_error("Capture size mismatch");
 if(field(meta,"scan_count")!=scanN || field(meta,"candidate_count")!=groupN || field(meta,"query_count")!=queryN || field(meta,"peptide_count")!=pepN || field(meta,"peak_count")!=peakN || field(meta,"hub_count")!=hm.count<short>())throw std::runtime_error("Metadata/count mismatch");
 const auto* scans=sm.data<Scan>();const auto* groups=gm.data<Candidate>();const auto* offsets=om.data<uint64_t>();const auto* mz=qm.data<double>();const auto* valid=vm.data<unsigned char>();const auto* peptides=pepm.data<PeptideInput>();
 const Config cfg=*cfgm.data<Config>();if(cfg.classes<1 || cfg.classes>MaxClasses || !std::isfinite(cfg.fragmentTolerance) || cfg.fragmentTolerance<=0 || offsets[0] || offsets[groupN]!=queryN)throw std::runtime_error("Invalid capture configuration");
 for(size_t i=0;i<pepN;++i){auto start=peptides[i].text;if(start>=tm.bytes || !std::memchr(tm.data<char>()+start,0,std::min(size_t(MaxText),tm.bytes-start)))throw std::runtime_error("Invalid peptide text");}
 for(size_t i=0;i<scanN;++i){const auto& s=scans[i];if(s.peaks<0 || s.peakOffset>peakN || size_t(s.peaks)>peakN-s.peakOffset || s.totalBins<0 || size_t(s.totalBins)>=tableN)throw std::runtime_error("Invalid scan");
  if(s.peaks && (s.highest<s.lowest || s.hubOffset>hm.count<short>() || uint64_t(s.highest-s.lowest+1)*2>hm.count<short>()-s.hubOffset))throw std::runtime_error("Invalid hub");
  for(int k=0;k<=cfg.classes;++k)if(s.counts[k]<0 || size_t(s.counts[k])>=tableN)throw std::runtime_error("Invalid MVH population");}
 for(size_t i=0;i<peakN;++i)if(cm.data<int>()[i]<0 || cm.data<int>()[i]>cfg.classes)throw std::runtime_error("Invalid class");
 for(size_t i=0;i<groupN;++i)if(groups[i].scanId<0 || size_t(groups[i].scanId)>=scanN || groups[i].peptideId<0 || size_t(groups[i].peptideId)>=pepN || offsets[i]>offsets[i+1] || valid[i]>2)throw std::runtime_error("Invalid candidate");
 const size_t limit=maxChunks?std::min(groupN,size_t(maxChunks)*chunk):groupN;
 fs::create_directories(output.parent_path());if(!fs::create_directory(output))throw std::runtime_error("Output must be new");
 omp_set_dynamic(0);omp_set_num_threads(threads);int actualThreads=0;
 #pragma omp parallel
 {if(omp_get_thread_num()==0)actualThreads=omp_get_num_threads();}
 std::ofstream samples(output/"samples.tsv"),report(output/"comparison.tsv");samples<<std::setprecision(15)<<"mode\tchunk\tbackend\trepeat\tseconds\n";report<<std::setprecision(15)<<"mode\tbackend\tgroups\tqueries\tthreads\tmedian_seconds\tmin_seconds\tmax_seconds\tsummary_checksum\n";
 const char* names[]={"cpu","cuda","rt"};const char* modes[]={"precomputed","generate_match","generate_match_mvh"};
 std::unique_ptr<Device> d;double commonUpload=0;
 if(selected[1] || selected[2]){auto t=Clock::now();d=std::make_unique<Device>();upload(d->scans,scans,scanN);upload(d->peaks,pm.data<double>(),peakN);upload(d->classes,cm.data<int>(),peakN);upload(d->hub,hm.data<short>(),hm.count<short>());upload(d->peptides,peptides,pepN);upload(d->texts,tm.data<char>(),tm.bytes);upload(d->table,lm.data<double>(),tableN);check(cudaMemcpyToSymbol(fusionConfig,&cfg,sizeof(cfg)));synced();commonUpload=seconds(t);}
 uint64_t auditMismatches=0;double auditMaxError=0;auto auditStart=Clock::now();
 // Independently verify host ion generation against the captured GPU query stream.
 #pragma omp parallel for schedule(static) reduction(+:auditMismatches) reduction(max:auditMaxError)
 for(size_t i=0;i<limit;++i){const auto c=groups[i];const auto& s=scans[c.scanId];if(s.skip){if(offsets[i]!=offsets[i+1] || valid[i]!=2)++auditMismatches;continue;}
  AuditSink sink{s,mz,offsets[i],offsets[i+1]};bool ok=CalculateSequenceIons(tm.data<char>()+peptides[c.peptideId].text,c.charge,cfg,sink);
  auditMismatches+=sink.mismatches+(sink.pos!=sink.end)+(ok!=(valid[i]==1));auditMaxError=std::max(auditMaxError,sink.maxError);}
 const double auditSeconds=seconds(auditStart);
 {std::ofstream f(output/"ion_audit.json");f<<std::setprecision(17)<<"{\"query_or_validity_mismatches\":"<<auditMismatches<<",\"max_mz_error\":"<<auditMaxError<<",\"seconds_excluded_from_timing\":"<<auditSeconds<<"}\n";}
 if(auditMismatches)throw std::runtime_error("Host ion generation differs from captured GPU queries; see ion_audit.json");
 std::ofstream mismatchFile(output/"result_mismatches.tsv");
 mismatchFile<<std::setprecision(17)<<"mode\tcandidate\tbackend\tcpu_peak_hits\tobserved_peak_hits\tcpu_matched\tobserved_matched\tcpu_status\tobserved_status\tcpu_score\tobserved_score\n";
 uint64_t rtMismatches[3]{},totalMismatches=0,rtChecksums[3]{};
 uint64_t checksumByMode[3]{};double buildTimes[3]{},uploadTimes[3]{},maxScoreError=0;uint64_t comparisons=0;
 for(int mode=0;mode<3;++mode){
  if(selected[2]){mvh_rt_gpu::reset();auto t=Clock::now();std::vector<Scan> host(scans,scans+scanN);mvh_rt_gpu::prepare(host,d->scans.p,d->peaks.p,d->classes.p,peakN,mvh_rt_gpu::GeometryKind::Spheres,cfg,false,mode+(rtPostcheck?0:3));synced();buildTimes[mode]=seconds(t);}
  std::vector<double> totals[3]={std::vector<double>(repeats),std::vector<double>(repeats),std::vector<double>(repeats)};
  uint64_t checksums[3]={1469598103934665603ULL,1469598103934665603ULL,1469598103934665603ULL};
  for(size_t first=0,ci=0;first<limit;first+=chunk,++ci){const int n=int(std::min(size_t(chunk),limit-first));const uint64_t base=offsets[first],nq=offsets[first+n]-base;
   std::vector<FusionOutput> cpu(n),a,b;auto runCpu=[&]{
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i){auto c=groups[first+i];const auto& s=scans[c.scanId];auto lookup=mvh_matching::BinLookup{s,pm.data<double>(),hm.data<short>(),cfg.fragmentTolerance};
     if(mode==0)cpu[i]=mvh_matching::evaluateFusion<0>(s,c,peptides,tm.data<char>(),cm.data<int>(),lm.data<double>(),cfg,mz,offsets[first+i],offsets[first+i+1],valid[first+i],lookup);
     else if(mode==1)cpu[i]=mvh_matching::evaluateFusion<1>(s,c,peptides,tm.data<char>(),cm.data<int>(),lm.data<double>(),cfg,nullptr,0,0,valid[first+i],lookup);
     else cpu[i]=mvh_matching::evaluateFusion<2>(s,c,peptides,tm.data<char>(),cm.data<int>(),lm.data<double>(),cfg,nullptr,0,0,valid[first+i],lookup);
    }};
   if(d){auto t=Clock::now();upload(d->groups,groups+first,n);upload(d->offsets,offsets+first,n+1);upload(d->validity,valid+first,n);if(mode==0)upload(d->mz,mz+base,nq);d->cudaOut.resize(n);d->rtOut.resize(n);
    if(selected[2]){mvh_rt_gpu::Params p{};p.scans=d->scans.p;p.candidates=d->groups.p;p.peptides=d->peptides.p;p.texts=d->texts.p;p.peaks=d->peaks.p;p.classes=d->classes.p;p.hub=d->hub.p;p.lnTable=d->table.p;p.cfg=cfg;p.size=n;p.fusionOutputs=d->rtOut.p;p.fusionValidity=d->validity.p;p.matchOffsets=d->offsets.p;p.matchMz=d->mz.p;p.matchBase=base;mvh_rt_gpu::stageFusion(p);}synced();uploadTimes[mode]+=seconds(t);}
   auto gpu=[&](int backend){if(backend==2)mvh_rt_gpu::launchStagedMatching();else if(mode==0)launch<0>(*d,n,base);else if(mode==1)launch<1>(*d,n,base);else launch<2>(*d,n,base);};
   for(int w=0;w<warmup;++w)for(int k=0;k<3;++k)if(selected[k]){if(k==0)runCpu();else{gpu(k);synced();}}
   std::unique_ptr<Events> events;if(d)events=std::make_unique<Events>();
   for(int r=0;r<repeats;++r)for(int k=0;k<3;++k){const int backend=(k+r+ci)%3;if(!selected[backend])continue;double t;
    if(backend==0){auto start=Clock::now();runCpu();t=seconds(start);}else t=events->time([&]{gpu(backend);});
    totals[backend][r]+=t;samples<<modes[mode]<<'\t'<<ci<<'\t'<<names[backend]<<'\t'<<r<<'\t'<<t<<'\n';}
   if(!selected[0])runCpu();if(selected[1])d->cudaOut.read(a);if(selected[2])d->rtOut.read(b);
   for(int i=0;i<n;++i){
    checksums[0]=hashSummary(checksums[0],cpu[i]);
    for(int backend=1;backend<3;++backend)if(selected[backend]){
     const auto& v=backend==1?a[i]:b[i];++comparisons;
     if(!equal(cpu[i],v,maxScoreError)){
      if(backend!=2 || rtPostcheck)throw std::runtime_error(std::string("Result mismatch mode=")+modes[mode]+" backend="+names[backend]+" candidate="+std::to_string(first+i));
      ++rtMismatches[mode];++totalMismatches;
      if(rtMismatches[mode]<=20)mismatchFile<<modes[mode]<<'\t'<<first+i<<'\t'<<names[backend]<<'\t'<<cpu[i].summary.peakHits<<'\t'<<v.summary.peakHits<<'\t'<<cpu[i].result.matched<<'\t'<<v.result.matched<<'\t'<<cpu[i].result.status<<'\t'<<v.result.status<<'\t'<<cpu[i].result.score<<'\t'<<v.result.score<<'\n';
     }
     checksums[backend]=hashSummary(checksums[backend],v);
    }
   }
   std::cout<<modes[mode]<<" chunk="<<ci<<" candidates="<<first+n<<'/'<<limit<<" checked rt_mismatches="<<rtMismatches[mode]<<std::endl;
  }
  checksumByMode[mode]=checksums[0];rtChecksums[mode]=checksums[2];
  for(int k=0;k<3;++k)if(selected[k])report<<modes[mode]<<'\t'<<names[k]<<'\t'<<limit<<'\t'<<offsets[limit]<<'\t'<<(k==0?actualThreads:0)<<'\t'<<median(totals[k])<<'\t'<<*std::min_element(totals[k].begin(),totals[k].end())<<'\t'<<*std::max_element(totals[k].begin(),totals[k].end())<<'\t'<<checksums[k]<<'\n';
  report.flush();
 }
 if(checksumByMode[0]!=checksumByMode[1] || checksumByMode[1]!=checksumByMode[2])throw std::runtime_error("Matching summaries differ across fusion modes");
 std::ofstream result(output/"result.json");result<<std::setprecision(17)<<"{\"complete\":true,\"groups\":"<<limit<<",\"queries\":"<<offsets[limit]<<",\"threads_requested\":"<<threads<<",\"threads_actual\":"<<actualThreads<<",\"repeats\":"<<repeats<<",\"warmups_per_chunk\":"<<warmup<<",\"chunk_groups\":"<<chunk<<",\"rt_postcheck\":"<<(rtPostcheck?"true":"false")<<",\"results_equivalent\":"<<(totalMismatches?"false":"true")<<",\"rt_equivalent_performance_comparison_valid\":"<<(totalMismatches?"false":"true")<<",\"rt_mismatches_by_mode\":["<<rtMismatches[0]<<','<<rtMismatches[1]<<','<<rtMismatches[2]<<"],\"rt_same_summary_across_modes\":"<<((rtChecksums[0]==rtChecksums[1] && rtChecksums[1]==rtChecksums[2])?"true":"false")<<",\"result_mismatches\":"<<totalMismatches<<",\"validated_gpu_candidates\":"<<comparisons<<",\"max_score_error\":"<<maxScoreError<<",\"ion_query_mismatches\":"<<auditMismatches<<",\"common_upload_seconds\":"<<commonUpload<<",\"rt_build_seconds\":["<<buildTimes[0]<<','<<buildTimes[1]<<','<<buildTimes[2]<<"],\"chunk_upload_seconds\":["<<uploadTimes[0]<<','<<uploadTimes[1]<<','<<uploadTimes[2]<<"],\"same_summary_across_modes\":true,\"cpu_is_controlled_shared_algorithm_not_original_sipros\":true}\n";
 std::cout<<"FUSION REPLAY COMPLETE"<<std::endl;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
