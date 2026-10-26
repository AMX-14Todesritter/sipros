// Matching-only experiment: one candidate/thread, one class/query. No scoring.
#define MVH_OPTIX_DEVICE
#include "scoring.cuh"
#include "bridge.h"
#include <omp.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
using namespace mvh_cuda;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
#ifdef MVH_MATCHING_POSTCHECK
constexpr bool Postcheck=true;
#else
constexpr bool Postcheck=false;
#endif
struct Mapping {
 int fd=-1;void* ptr=nullptr;size_t bytes=0;
 explicit Mapping(fs::path p){bytes=fs::file_size(p);fd=open(p.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("Cannot open "+p.string());
  if(bytes){ptr=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);if(ptr==MAP_FAILED){close(fd);fd=-1;throw std::runtime_error("mmap failed");}}}
 ~Mapping(){if(ptr && ptr!=MAP_FAILED)munmap(ptr,bytes);if(fd>=0)close(fd);}
 template<class T>const T* data()const{return static_cast<const T*>(ptr);}
 void require(uint64_t n,size_t width)const{if(n>size_t(-1)/width || bytes!=n*width)throw std::runtime_error("Capture size mismatch");}
};
std::string textFile(fs::path p){std::ifstream f(p);if(!f)throw std::runtime_error("Missing "+p.string());return {(std::istreambuf_iterator<char>(f)),{}};}
std::string field(const std::string& s,const char* key){std::smatch m;if(!std::regex_search(s,m,std::regex(std::string("\"")+key+"\"\\s*:\\s*([0-9.eE+-]+)")))throw std::runtime_error(std::string("Missing field ")+key);return m[1];}
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
double median(std::vector<double> v){std::sort(v.begin(),v.end());return v.size()%2?v[v.size()/2]:(v[v.size()/2-1]+v[v.size()/2])/2;}
template<class T>void upload(Buffer<T>& d,const T* p,size_t n){d.resize(std::max(n,size_t(1)));if(n)check(cudaMemcpy(d.p,p,n*sizeof(T),cudaMemcpyHostToDevice));}
struct Events {
 cudaEvent_t start,end;
 Events(){check(cudaEventCreate(&start));check(cudaEventCreate(&end));}
 ~Events(){cudaEventDestroy(start);cudaEventDestroy(end);}
 template<class F>double measure(F f){check(cudaEventRecord(start));f();check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,start,end));return ms*.001;}
};
struct RtReset {~RtReset(){mvh_rt_gpu::reset();}};
// Uses the production CUDA matcher verbatim, including strict tolerance and class 0.
__global__ void matchClasses(const Scan* scans,const Candidate* candidates,int n,
 const uint64_t* offsets,uint64_t base,const double* mz,const double* peaks,
 const int* classes,const short* hub,double tolerance,int* output){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const Scan& scan=scans[candidates[i].scanId];
 for(uint64_t j=offsets[i]-base;j<offsets[i+1]-base;++j)
  output[j]=findNear(mz[j],tolerance,scan,peaks,classes,hub);
}
// Independent, untimed oracle: binary search sorted peaks, preserving first-on-tie.
int oracle(double mz,const Scan& s,const double* peaks,double tolerance){
 if(!s.peaks)return -1;
 const double* begin=peaks+s.peakOffset;const double* end=begin+s.peaks;
 const int right=int(std::lower_bound(begin,end,mz)-begin);
 int best=-1;double distance=tolerance;
 for(int i:{right,right-1})if(i>=0 && i<s.peaks){const double d=std::abs(mz-begin[i]);
  if(d<tolerance && (d<distance || (d==distance && (best<0 || i<best)))){best=i;distance=d;}}
 if(best>=0)best=int(std::lower_bound(begin,end,begin[best])-begin);
 return best;
}
int main(int argc,char** argv){try{
 fs::path input,output;int repeats=7,warmup=2,threads=8,maxChunks=0;uint64_t chunk=1000000;
 for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::invalid_argument("Missing option value");std::string k=argv[i],v=argv[i+1];
  if(k=="--capture")input=v;else if(k=="--output")output=v;else if(k=="--repeats")repeats=std::stoi(v);
  else if(k=="--warmup")warmup=std::stoi(v);else if(k=="--chunk-groups"){if(v.empty() || v[0]=='-')throw std::invalid_argument("Invalid chunk");chunk=std::stoull(v);}
  else if(k=="--threads")threads=std::stoi(v);else if(k=="--max-chunks")maxChunks=std::stoi(v);else throw std::invalid_argument("Unknown option: "+k);}
 if(input.empty() || output.empty() || repeats<1 || warmup<0 || threads<1 || maxChunks<0 || chunk>uint64_t(INT32_MAX))throw std::invalid_argument("Invalid replay configuration");
 const std::string meta=textFile(input/"COMPLETE.json");if(field(meta,"format_version")!="1" || meta.find("little")==std::string::npos)throw std::runtime_error("Require little-endian capture v1");
 const uint16_t endian=1;if(*reinterpret_cast<const unsigned char*>(&endian)!=1)throw std::runtime_error("Little-endian host required");
 const uint64_t scansN=std::stoull(field(meta,"scan_count")),peaksN=std::stoull(field(meta,"peak_count")),hubN=std::stoull(field(meta,"hub_count")),groupsN=std::stoull(field(meta,"candidate_count")),queriesN=std::stoull(field(meta,"query_count"));
 const double tolerance=std::stod(field(meta,"fragment_tolerance_da"));
 if(!scansN || scansN>INT32_MAX || !groupsN || groupsN>INT32_MAX || !std::isfinite(tolerance) || tolerance<=0 || tolerance>=.5)throw std::runtime_error("Invalid capture counts/tolerance");
 if(!chunk)chunk=groupsN;const uint64_t limit=maxChunks?std::min(groupsN,uint64_t(maxChunks)*chunk):groupsN;
 struct ScanRecord{uint64_t peak,hub;int32_t count,low,high,skip;double lower,upper;};static_assert(sizeof(ScanRecord)==48 && sizeof(Candidate)==16);
 Mapping sm(input/"scans.bin"),pm(input/"peaks.f64"),cm(input/"classes.i32"),hm(input/"hub.i16"),gm(input/"candidates.i32x4"),om(input/"query_offsets.u64"),qm(input/"query_mz.f64"),vm(input/"validity.u8");
 sm.require(scansN,48);pm.require(peaksN,8);cm.require(peaksN,4);hm.require(hubN,2);gm.require(groupsN,16);om.require(groupsN+1,8);qm.require(queriesN,8);vm.require(groupsN,1);
 const auto* candidates=gm.data<Candidate>();const auto* offsets=om.data<uint64_t>();const auto* mz=qm.data<double>();const auto* peaks=pm.data<double>();const auto* classes=cm.data<int>();const auto* hub=hm.data<short>();
 if(offsets[0] || offsets[groupsN]!=queriesN)throw std::runtime_error("Invalid query offsets");
 std::vector<Scan> scans(scansN);int classCount=1;
 for(uint64_t i=0;i<scansN;++i){auto r=sm.data<ScanRecord>()[i];auto& s=scans[i];s.peakOffset=r.peak;s.hubOffset=r.hub;s.peaks=r.count;s.lowest=r.low;s.highest=r.high;s.skip=r.skip;s.lower=r.lower;s.upper=r.upper;
  if(r.count<0 || r.peak>peaksN || uint64_t(r.count)>peaksN-r.peak || r.skip<0 || r.skip>1 || !std::isfinite(r.lower) || !std::isfinite(r.upper) || r.lower>r.upper)throw std::runtime_error("Invalid scan");
  if(r.count){if(r.low<0 || r.high<r.low || r.hub>hubN || uint64_t(r.high-r.low+1)*2>hubN-r.hub)throw std::runtime_error("Invalid hub span");
   for(int b=0;b<=r.high-r.low;++b){const int first=hub[r.hub+2*b],last=hub[r.hub+2*b+1];if(first!=-1 && (first<0 || last<first || last>r.count))throw std::runtime_error("Invalid hub entry");}}
  for(int j=0;j<r.count;++j){const auto p=r.peak+j;if(!std::isfinite(peaks[p]) || (j && peaks[p]<peaks[p-1]) || classes[p]<0 || classes[p]>MaxClasses)throw std::runtime_error("Invalid peak/class order");classCount=std::max(classCount,classes[p]);}
 }
 for(uint64_t i=0;i<groupsN;++i){if(candidates[i].scanId<0 || uint64_t(candidates[i].scanId)>=scansN || offsets[i]>offsets[i+1] || vm.data<unsigned char>()[i]>2)throw std::runtime_error("Invalid candidate");
  const auto& s=scans[candidates[i].scanId];if(s.skip && offsets[i]!=offsets[i+1])throw std::runtime_error("Skipped scan has queries");
  for(uint64_t q=offsets[i];q<offsets[i+1];++q)if(!std::isfinite(mz[q]) || mz[q]<s.lower || mz[q]>s.upper)throw std::runtime_error("Invalid in-range query");}
 fs::create_directories(output.parent_path());if(!fs::create_directory(output))throw std::runtime_error("Output must be new");
 omp_set_dynamic(0);omp_set_num_threads(threads);
 Config cfg{};cfg.classes=classCount;cfg.fragmentTolerance=tolerance;
 auto t=Clock::now();Buffer<Scan> ds(scans);Buffer<double> dp;Buffer<int> dc;Buffer<short> dh;
 upload(dp,peaks,peaksN);upload(dc,classes,peaksN);upload(dh,hub,hubN);synced();double commonUpload=elapsed(t);
 RtReset cleanup;t=Clock::now();mvh_rt_gpu::prepare(scans,ds.p,dp.p,dc.p,peaksN,mvh_rt_gpu::GeometryKind::Spheres,cfg);synced();double rtBuild=elapsed(t);
 Buffer<Candidate> dg;Buffer<uint64_t> dOffsets;Buffer<double> dMz;Buffer<int> deviceOutput;
 std::ofstream samples(output/"samples.tsv"),mismatch(output/"mismatch_examples.tsv");samples<<std::setprecision(15)<<"chunk\tfirst_candidate\tcandidates\tqueries\tbackend\trepeat\tseconds\n";
 mismatch<<std::setprecision(17)<<"query\tcandidate\tscan\tmz\tcpu_peak\tcpu_class\trt_class\n";
 std::vector<double> times[2]={std::vector<double>(repeats),std::vector<double>(repeats)};
 uint64_t differences=0,positive[2]{},checksums[2]={1469598103934665603ULL,1469598103934665603ULL};double uploads=0,validation=0;size_t minFree=size_t(-1),totalMemory=0;
 for(uint64_t first=0,ci=0;first<limit;first+=chunk,++ci){const int n=int(std::min(chunk,limit-first));const uint64_t base=offsets[first],nq=offsets[first+n]-base;
  t=Clock::now();upload(dg,candidates+first,n);upload(dOffsets,offsets+first,n+1);upload(dMz,mz+base,nq);deviceOutput.resize(std::max(nq,uint64_t(1)));
  mvh_rt_gpu::Params p{};p.scans=ds.p;p.candidates=dg.p;p.peaks=dp.p;p.classes=dc.p;p.cfg=cfg;p.size=n;p.matchOffsets=dOffsets.p;p.matchMz=dMz.p;p.matchBase=base;p.matchClasses=deviceOutput.p;mvh_rt_gpu::stageMatching(p);synced();uploads+=elapsed(t);
  size_t available=0;check(cudaMemGetInfo(&available,&totalMemory));minFree=std::min(minFree,available);
  auto launch=[&](int backend){if(backend)mvh_rt_gpu::launchStagedMatching();else{matchClasses<<<(n+127)/128,128>>>(ds.p,dg.p,n,dOffsets.p,base,dMz.p,dp.p,dc.p,dh.p,tolerance,deviceOutput.p);check(cudaGetLastError());}};
  for(int w=0;w<warmup;++w)for(int k=0;k<2;++k){launch((w+ci+k)%2);synced();}
  Events events;
  for(int r=0;r<repeats;++r)for(int k=0;k<2;++k){int backend=(r+ci+k)%2;double seconds=events.measure([&]{launch(backend);});times[backend][r]+=seconds;samples<<ci<<'\t'<<first<<'\t'<<n<<'\t'<<nq<<'\t'<<(backend?"rt":"cuda")<<'\t'<<r<<'\t'<<seconds<<'\n';}
  t=Clock::now();std::vector<int> reference(nq),observed(nq);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<n;++i){const auto& s=scans[candidates[first+i].scanId];for(uint64_t q=offsets[first+i]-base;q<offsets[first+i+1]-base;++q){const int peak=oracle(mz[base+q],s,peaks,tolerance);reference[q]=peak<0?0:classes[s.peakOffset+peak];}}
  // Re-run outside timing, sharing one output allocation between both backends.
  launch(0);synced();if(nq)check(cudaMemcpy(observed.data(),deviceOutput.p,nq*4,cudaMemcpyDeviceToHost));
  if(reference!=observed)throw std::runtime_error("Production CUDA differs from independent sorted-peak oracle");
  for(int v:observed){positive[0]+=v>0;checksums[0]=(checksums[0]^uint64_t(v))*1099511628211ULL;}
  launch(1);synced();if(nq)check(cudaMemcpy(observed.data(),deviceOutput.p,nq*4,cudaMemcpyDeviceToHost));
  for(int i=0;i<n;++i){const auto& s=scans[candidates[first+i].scanId];for(uint64_t q=offsets[first+i]-base;q<offsets[first+i+1]-base;++q){int v=observed[q];if(v<0 || v>classCount)throw std::runtime_error("RT returned invalid class");positive[1]+=v>0;checksums[1]=(checksums[1]^uint64_t(v))*1099511628211ULL;
   if(v!=reference[q]){if(differences<30){int peak=oracle(mz[base+q],s,peaks,tolerance);mismatch<<base+q<<'\t'<<first+i<<'\t'<<candidates[first+i].scanId<<'\t'<<mz[base+q]<<'\t'<<peak<<'\t'<<reference[q]<<'\t'<<v<<'\n';}++differences;}}}
  validation+=elapsed(t);std::cout<<"CHUNK "<<ci<<" candidates="<<first+n<<'/'<<limit<<" queries="<<offsets[first+n]<<" class_mismatches="<<differences<<std::endl;
 }
 std::ofstream report(output/"comparison.tsv");report<<std::setprecision(17)<<"backend\tcandidates\tqueries\tmedian_matching_seconds\tmin_seconds\tmax_seconds\tpositive_class_queries\tclass_checksum\n";
 for(int k=0;k<2;++k)report<<(k?"rt":"cuda")<<'\t'<<limit<<'\t'<<offsets[limit]<<'\t'<<median(times[k])<<'\t'<<*std::min_element(times[k].begin(),times[k].end())<<'\t'<<*std::max_element(times[k].begin(),times[k].end())<<'\t'<<positive[k]<<'\t'<<checksums[k]<<'\n';
 std::ofstream result(output/"result.json");result<<std::setprecision(17)<<"{\"complete\":true,\"scope\":\"resident matching and per-query class output only\",\"rt_postcheck\":"<<(Postcheck?"true":"false")<<",\"results_equivalent\":"<<(differences?"false":"true")<<",\"class_mismatches\":"<<differences<<",\"cuda_oracle_mismatches\":0,\"candidates\":"<<limit<<",\"queries\":"<<offsets[limit]<<",\"scan_count\":"<<scansN<<",\"peak_count\":"<<peaksN<<",\"repeats\":"<<repeats<<",\"warmup_per_chunk\":"<<warmup<<",\"chunk_groups\":"<<chunk<<",\"chunk_count\":"<<(limit+chunk-1)/chunk<<",\"cuda_over_rt_matching\":"<<median(times[0])/median(times[1])<<",\"common_upload_seconds\":"<<commonUpload<<",\"rt_geometry_and_build_seconds\":"<<rtBuild<<",\"query_upload_seconds\":"<<uploads<<",\"validation_seconds\":"<<validation<<",\"minimum_free_vram_at_chunk_boundaries_bytes\":"<<minFree<<",\"total_vram_bytes\":"<<totalMemory<<",\"output_bytes_per_query\":4}\n";
 std::cout<<"MATCHING ONLY COMPLETE class_mismatches="<<differences<<std::endl;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
