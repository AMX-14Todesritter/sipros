#include "matching.cuh"
#include "bridge.h"
#include "profiling.h"
#include <cuda_profiler_api.h>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>

using namespace mvh_cuda;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
#ifdef MVH_MATCHING_DIAGNOSTICS
constexpr bool DiagnosticBuild=true;
void addStatistics(mvh_matching::Statistics& a,const mvh_matching::Statistics& b) {
    a.queries+=b.queries;a.hits+=b.hits;a.buckets+=b.buckets;
    a.emptyBuckets+=b.emptyBuckets;a.peakChecks+=b.peakChecks;
    a.emptyScans+=b.emptyScans;a.outsideRanges+=b.outsideRanges;
    a.onlyEmptyBuckets+=b.onlyEmptyBuckets;a.traces+=b.traces;a.noHandles+=b.noHandles;
    a.maxChecks=std::max(a.maxChecks,b.maxChecks);
    for(int i=0;i<9;++i)a.checkHistogram[i]+=b.checkHistogram[i];
}
#else
constexpr bool DiagnosticBuild=false;
#endif
#if defined(MVH_MATCHING_DIAGNOSTICS) || defined(MVH_MATCHING_NVTX)
constexpr bool ProfilingBuild=true;
#else
constexpr bool ProfilingBuild=false;
#endif
#ifdef MVH_MATCHING_NVTX
constexpr bool NvtxEnabled=true;
#else
constexpr bool NvtxEnabled=false;
#endif
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
struct Mapping {
    int fd=-1;void* ptr=nullptr;size_t bytes=0;
    explicit Mapping(const fs::path& p) {
        bytes=fs::file_size(p);fd=open(p.c_str(),O_RDONLY);
        if(fd<0) throw std::runtime_error("Cannot open "+p.string());
        if(bytes) {ptr=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);
            if(ptr==MAP_FAILED) {close(fd);fd=-1;throw std::runtime_error("mmap failed");}}
    }
    ~Mapping(){if(ptr && ptr!=MAP_FAILED)munmap(ptr,bytes);if(fd>=0)close(fd);}
    template<class T> const T* data() const {return static_cast<const T*>(ptr);}
    void requireBytes(uint64_t count,size_t width) const {
        if(count>std::numeric_limits<size_t>::max()/width || bytes!=count*width)
            throw std::runtime_error("Capture size mismatch");
    }
};
struct Events {
    cudaEvent_t start,end;
    Events(){check(cudaEventCreate(&start));check(cudaEventCreate(&end));}
    ~Events(){cudaEventDestroy(start);cudaEventDestroy(end);}
    template<class F> double time(F launch) {
        check(cudaEventRecord(start));launch();check(cudaEventRecord(end));
        check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,start,end));
        return double(ms)*0.001;
    }
};
struct RtReset {~RtReset(){mvh_rt_gpu::reset();}};
double median(std::vector<double> v) {
    std::sort(v.begin(),v.end());return v.size()%2?v[v.size()/2]:(v[v.size()/2-1]+v[v.size()/2])/2;
}
template<class T> void upload(Buffer<T>& buffer,const T* values,size_t n) {
    buffer.resize(std::max(n,size_t(1)));
    if(n)check(cudaMemcpy(buffer.p,values,n*sizeof(T),cudaMemcpyHostToDevice));
}
int main(int argc,char** argv) {
    try {
        fs::path input,output;int repeats=7,warmup=2,chunk=1000000,replicas=1;
        bool aggregate=false,profileCapture=false;int maxChunks=0;
        for(int i=1;i<argc;++i) {
            std::string k=argv[i];if(i+1>=argc)throw std::invalid_argument("Missing option value");
            std::string v=argv[++i];
            if(k=="--capture")input=v;else if(k=="--output")output=v;
            else if(k=="--repeats")repeats=std::stoi(v);
            else if(k=="--warmup")warmup=std::stoi(v);
            else if(k=="--chunk-groups")chunk=std::stoi(v);
            else if(k=="--output-mode") {if(v!="queries" && v!="candidate")throw std::invalid_argument("Bad output mode");aggregate=v=="candidate";}
            else if(k=="--max-chunks")maxChunks=std::stoi(v);
            else if(k=="--profile-capture")profileCapture=std::stoi(v)!=0;
            else if(k=="--replicas")replicas=std::stoi(v);
            else throw std::invalid_argument("Unknown option: "+k);
        }
        if(input.empty() || output.empty() || repeats<1 || warmup<0 || chunk<1 || replicas<1 || replicas>32 || replicas>chunk)
            throw std::invalid_argument("Require --capture DIR --output NEW_DIR [--repeats N --warmup N --chunk-groups N]");
        if(maxChunks<0 || (profileCapture && (!NvtxEnabled || maxChunks!=1)))throw std::invalid_argument("Profile capture requires NVTX target and --max-chunks 1");
        mvh_matching::ProfileRange replayRange("matching/replay");
        fs::create_directories(output.parent_path());
        if(!fs::create_directory(output))throw std::runtime_error("Output must be new");
        std::ifstream mf(input/"COMPLETE.json");
        std::string meta((std::istreambuf_iterator<char>(mf)),{});
        auto value=[&](const std::string& key) {
            std::smatch match;
            if(!std::regex_search(meta,match,std::regex("\\\""+key+"\\\"\\s*:\\s*([0-9.eE+-]+)")))
                throw std::runtime_error("Missing capture field: "+key);
            return match[1].str();
        };
        if(value("format_version")!="1" || meta.find("little")==std::string::npos)
            throw std::runtime_error("Unsupported capture format");
        const uint16_t endian=1;
        if(*reinterpret_cast<const unsigned char*>(&endian)!=1)throw std::runtime_error("Little endian required");
        const uint64_t scansN=std::stoull(value("scan_count")),peaksN=std::stoull(value("peak_count"));
        const uint64_t hubN=std::stoull(value("hub_count")),groupsN=std::stoull(value("candidate_count"));
        const uint64_t queriesN=std::stoull(value("query_count"));
        const double tolerance=std::stod(value("fragment_tolerance_da"));
        if(!scansN || scansN>size_t(std::numeric_limits<int>::max())/replicas || !groupsN)
            throw std::runtime_error("Invalid capture counts");
        struct ScanRecord {uint64_t peak,hub;int32_t count,low,high,skip;double lower,upper;};
        static_assert(sizeof(ScanRecord)==48 && sizeof(Candidate)==16);
        Mapping sm(input/"scans.bin"),pm(input/"peaks.f64"),cm(input/"classes.i32"),hm(input/"hub.i16");
        Mapping gm(input/"candidates.i32x4"),om(input/"query_offsets.u64"),qm(input/"query_mz.f64");
        sm.requireBytes(scansN,48);pm.requireBytes(peaksN,8);cm.requireBytes(peaksN,4);hm.requireBytes(hubN,2);
        gm.requireBytes(groupsN,16);om.requireBytes(groupsN+1,8);qm.requireBytes(queriesN,8);
        const auto* groups=gm.data<Candidate>();const auto* offsets=om.data<uint64_t>();
        const auto* mz=qm.data<double>();const auto* classes=cm.data<int>();
        if(offsets[0] || offsets[groupsN]!=queriesN)throw std::runtime_error("Invalid group offsets");
        std::vector<Scan> scans(scansN*replicas);
        for(size_t i=0;i<scans.size();++i) {
            const auto s=sm.data<ScanRecord>()[i%scansN];auto& d=scans[i];
            const uint64_t replica=i/scansN;
            d.peakOffset=s.peak+replica*peaksN;d.hubOffset=s.hub+replica*hubN;d.peaks=s.count;d.lowest=s.low;
            d.highest=s.high;d.skip=s.skip;d.lower=s.lower;d.upper=s.upper;
            if(s.count<0 || s.peak>peaksN || uint64_t(s.count)>peaksN-s.peak)
                throw std::runtime_error("Invalid peak span");
            if(s.count && (s.high<s.low || s.hub>hubN || uint64_t(s.high-s.low+1)*2>hubN-s.hub))
                throw std::runtime_error("Invalid hub span");
        }
        for(size_t i=0;i<groupsN;++i)
            if(groups[i].scanId<0 || size_t(groups[i].scanId)>=scansN || offsets[i]>offsets[i+1])
                throw std::runtime_error("Invalid candidate/offset");
        Config cfg{};cfg.fragmentTolerance=tolerance;cfg.classes=1;
        for(size_t i=0;i<peaksN;++i)cfg.classes=std::max(cfg.classes,classes[i]);
        auto start=Clock::now();
        Buffer<Scan> ds(scans);Buffer<double> dp;Buffer<int> dc;Buffer<short> dh;
        dp.resize(peaksN*replicas);dc.resize(peaksN*replicas);dh.resize(hubN*replicas);
        for(int r=0;r<replicas;++r) {
            check(cudaMemcpy(dp.p+r*peaksN,pm.data<double>(),peaksN*8,cudaMemcpyHostToDevice));
            check(cudaMemcpy(dc.p+r*peaksN,classes,peaksN*4,cudaMemcpyHostToDevice));
            check(cudaMemcpy(dh.p+r*hubN,hm.data<short>(),hubN*2,cudaMemcpyHostToDevice));
        }
        synced();const double uploadIndex=elapsed(start);
        Buffer<Candidate> dg;Buffer<uint64_t> dOffsets;Buffer<double> dMz;
        Buffer<int> cudaResult,rtResult;
        Buffer<mvh_matching::CandidateSummary> cudaSummaries,rtSummaries;
#ifdef MVH_MATCHING_DIAGNOSTICS
        Buffer<mvh_matching::Statistics> cudaStatistics,rtStatistics;
        mvh_matching::Statistics statistics[2]{};
#endif
        RtReset cleanup;
        start=Clock::now();
        {mvh_matching::ProfileRange range("matching/rt_build");
         mvh_rt_gpu::prepare(scans,ds.p,dp.p,dc.p,peaksN*replicas,mvh_rt_gpu::GeometryKind::Spheres,cfg,aggregate);
         synced();}
        const double buildRt=elapsed(start);
        std::vector<double> totals[2]={std::vector<double>(repeats),std::vector<double>(repeats)};
        std::ofstream samples(output/"samples.tsv"),examples(output/"mismatch_examples.tsv");
        samples<<std::setprecision(12)<<"chunk\tfirst_group\tgroups\tqueries\tbackend\trepeat\tseconds\n";
        examples<<std::setprecision(17)<<"query_index\tgroup\tscan\tmz\tcuda_peak\trt_peak\tcuda_class\trt_class\n";
        uint64_t mismatches=0,classMismatches=0,hits[2]={0,0},checksums[2]={0,0},seen=0;
        size_t minFree=std::numeric_limits<size_t>::max(),totalMemory=0;
        double uploadQueries=0;Events timer;size_t chunkIndex=0;
        const int sourceChunk=chunk/replicas;
        const uint64_t replayGroups=maxChunks?std::min(groupsN,uint64_t(maxChunks)*sourceChunk):groupsN;
        const uint64_t replayQueries=offsets[replayGroups]*replicas;
        uint64_t summaryMismatches=0;
        for(uint64_t first=0;first<replayGroups;first+=sourceChunk,++chunkIndex) {
            mvh_matching::ProfileRange chunkRange("matching/chunk");
            const int sourceN=int(std::min(uint64_t(sourceChunk),replayGroups-first));
            const int n=sourceN*replicas;
            const uint64_t base=offsets[first]*replicas;
            const uint64_t queryCount=(offsets[first+sourceN]-offsets[first])*replicas;
            std::vector<Candidate> chunkGroups;chunkGroups.reserve(n);
            std::vector<uint64_t> chunkOffsets;chunkOffsets.reserve(n+1);chunkOffsets.push_back(base);
            std::vector<double> chunkMz;chunkMz.reserve(queryCount);
            for(int g=0;g<sourceN;++g) {
                for(int r=0;r<replicas;++r) {
                    Candidate c=groups[first+g];c.scanId+=r*scansN;chunkGroups.push_back(c);
                    chunkMz.insert(chunkMz.end(),mz+offsets[first+g],mz+offsets[first+g+1]);
                    chunkOffsets.push_back(base+chunkMz.size());
                }
            }
            mvh_matching::Statistics* cudaCounters=nullptr; mvh_matching::Statistics* rtCounters=nullptr;
            start=Clock::now(); {mvh_matching::ProfileRange uploadRange("matching/upload");
            upload(dg,chunkGroups.data(),n);upload(dOffsets,chunkOffsets.data(),n+1);
            upload(dMz,chunkMz.data(),queryCount);cudaResult.resize(std::max(queryCount,uint64_t(1)));
            rtResult.resize(aggregate?1:std::max(queryCount,uint64_t(1)));
            if(aggregate){cudaSummaries.resize(n);rtSummaries.resize(n);}

#ifdef MVH_MATCHING_DIAGNOSTICS
            cudaStatistics.resize(n);rtStatistics.resize(n);
            cudaCounters=cudaStatistics.p;rtCounters=rtStatistics.p;
#endif
            mvh_rt_gpu::stageGroupedMatching(ds.p,dg.p,dOffsets.p,dMz.p,base,n,
                aggregate?nullptr:rtResult.p,cfg,rtCounters,aggregate?rtSummaries.p:nullptr,
                aggregate?dc.p:nullptr);
            synced();}uploadQueries+=elapsed(start);
            size_t free=0;check(cudaMemGetInfo(&free,&totalMemory));minFree=std::min(minFree,free);
            auto launch=[&](int b) {
                mvh_matching::ProfileRange range(b==0?(aggregate?"matching/cuda_candidate":"matching/cuda"):(aggregate?"matching/rt_candidate":"matching/rt"));
                if(b==0) {
                    if(aggregate)mvh_matching::launchCudaGroupedSummary(ds.p,dp.p,dh.p,dc.p,cfg.classes,
                        dg.p,dOffsets.p,dMz.p,base,n,tolerance,cudaSummaries.p,cudaCounters);
                    else mvh_matching::launchCudaGrouped(ds.p,dp.p,dh.p,dg.p,dOffsets.p,dMz.p,
                        base,n,tolerance,cudaResult.p,nullptr,cudaCounters);
                }
                else mvh_rt_gpu::launchStagedMatching();
            };
            {mvh_matching::ProfileRange warmupRange("matching/warmup");
             for(int r=0;r<warmup;++r)for(int j=0;j<2;++j){launch((j+chunkIndex+r)%2);synced();}}
            if(profileCapture && chunkIndex==0)check(cudaProfilerStart());
            for(int r=0;r<repeats;++r)for(int j=0;j<2;++j) {
                mvh_matching::ProfileRange measuredRange("matching/measured");
                const int b=(j+chunkIndex+r)%2;const double t=timer.time([&]{launch(b);});
                totals[b][r]+=t;
                samples<<chunkIndex<<'\t'<<first*replicas<<'\t'<<n<<'\t'<<queryCount<<'\t'
                    <<(b==0?"cuda":"rt-custom")<<'\t'<<r<<'\t'<<t<<'\n';
            }
            if(profileCapture && chunkIndex==0)check(cudaProfilerStop());
#ifdef MVH_MATCHING_DIAGNOSTICS
            {mvh_matching::ProfileRange range("matching/diagnostic_download");
             std::vector<mvh_matching::Statistics> data;
             for(int backend=0;backend<2;++backend) {
                if(backend==0)cudaStatistics.read(data);else rtStatistics.read(data);
                mvh_matching::Statistics chunkStats{};
                for(const auto& item:data)addStatistics(chunkStats,item);
                if(chunkStats.queries!=queryCount)throw std::runtime_error("Diagnostic query count mismatch");
                addStatistics(statistics[backend],chunkStats);
             }}
#endif
            mvh_matching::ProfileRange validationRange("matching/validate");
            if(aggregate) {
                // Validation-only reference kernel; excluded from timing/profiler capture.
                {mvh_matching::ProfileRange referenceRange("matching/reference");
                 mvh_matching::launchCudaGrouped(ds.p,dp.p,dh.p,dg.p,dOffsets.p,dMz.p,
                     base,n,tolerance,cudaResult.p);synced();}
                std::vector<int> reference(queryCount);
                if(queryCount)check(cudaMemcpy(reference.data(),cudaResult.p,queryCount*4,cudaMemcpyDeviceToHost));
                std::vector<mvh_matching::CandidateSummary> a,b;cudaSummaries.read(a);rtSummaries.read(b);
                for(int g=0;g<n;++g) {
                    const auto& scan=scans[chunkGroups[g].scanId];mvh_matching::CandidateSummary expected{};
                    if(chunkOffsets[g+1]-chunkOffsets[g]>UINT32_MAX)throw std::runtime_error("Summary count overflow");
                    for(uint64_t q=chunkOffsets[g]-base;q<chunkOffsets[g+1]-base;++q) {
                        const int peak=reference[q];
                        if(peak<-1 || peak>=scan.peaks)throw std::runtime_error("Invalid reference peak");
                        const int cls=peak<0?0:classes[(scan.peakOffset+peak)%peaksN];
                        expected.peakHits+=peak>=0;if(cls>0)++expected.bins[cls-1];else ++expected.bins[cfg.classes];
                    }
                    if(std::memcmp(&expected,&a[g],sizeof(expected)))throw std::runtime_error("CUDA summary/reference mismatch");
                    summaryMismatches+=std::memcmp(&expected,&b[g],sizeof(expected))!=0;
                    for(int backend=0;backend<2;++backend) {
                        const auto& v=backend==0?a[g]:b[g];hits[backend]+=v.peakHits;
                        checksums[backend]=checksums[backend]*1099511628211ULL+v.peakHits;
                        for(auto bin:v.bins)checksums[backend]=checksums[backend]*1099511628211ULL+bin;
                    }
                }
            } else {
            std::vector<int> a(queryCount),b(queryCount);
            if(queryCount) {
                check(cudaMemcpy(a.data(),cudaResult.p,queryCount*4,cudaMemcpyDeviceToHost));
                check(cudaMemcpy(b.data(),rtResult.p,queryCount*4,cudaMemcpyDeviceToHost));
            }
            for(int g=0;g<n;++g) {
                const int scanId=chunkGroups[g].scanId;const auto& scan=scans[scanId];
                for(uint64_t q=chunkOffsets[g]-base;q<chunkOffsets[g+1]-base;++q) {
                    if(a[q]<-1 || a[q]>=scan.peaks || b[q]<-1 || b[q]>=scan.peaks)
                        throw std::runtime_error("Invalid matching result index");
                    hits[0]+=a[q]>=0;hits[1]+=b[q]>=0;
                    checksums[0]=checksums[0]*1099511628211ULL+uint64_t(a[q]+1);
                    checksums[1]=checksums[1]*1099511628211ULL+uint64_t(b[q]+1);
                    const int ca=a[q]<0?0:classes[(scan.peakOffset+a[q])%peaksN];
                    const int cb=b[q]<0?0:classes[(scan.peakOffset+b[q])%peaksN];
                    classMismatches+=ca!=cb;
                    if(a[q]!=b[q]) {
                        if(mismatches<20)examples<<base+q<<'\t'<<first*replicas+g<<'\t'<<scanId<<'\t'
                            <<chunkMz[q]<<'\t'<<a[q]<<'\t'<<b[q]<<'\t'<<ca<<'\t'<<cb<<'\n';
                        ++mismatches;
                    }
                }
            }
            }
            seen+=queryCount;
            std::cout<<"CHUNK "<<chunkIndex<<" groups="<<(first+sourceN)*replicas<<'/'<<groupsN*replicas
                <<" queries="<<seen<<" mismatches="<<mismatches<<std::endl;
        }
        if(seen!=replayQueries)throw std::runtime_error("Incomplete replay");
#ifdef MVH_MATCHING_DIAGNOSTICS
        std::ofstream diagnostic(output/"diagnostics.tsv");diagnostic<<std::setprecision(12);
        diagnostic<<"backend\tmetric\tvalue\n";
        for(int backend=0;backend<2;++backend) {
            const auto& d=statistics[backend];
            if(d.queries!=seen || d.hits!=hits[backend])throw std::runtime_error("Diagnostic/result conservation failed");
            auto row=[&](const char* metric,auto value){diagnostic<<(backend==0?"cuda":"rt-custom")<<'\t'<<metric<<'\t'<<value<<'\n';};
            row("queries",d.queries);row("peak_hits",d.hits);row("misses",d.queries-d.hits);
            if(backend==0) {
                row("bucket_visits",d.buckets);row("empty_bucket_visits",d.emptyBuckets);
                row("queries_touching_only_empty_buckets",d.onlyEmptyBuckets);
                row("empty_scan_queries",d.emptyScans);row("outside_integer_range_queries",d.outsideRanges);
                row("peak_checks",d.peakChecks);row("max_peak_checks_per_query",d.maxChecks);
                row("mean_peak_checks_per_query",d.queries?double(d.peakChecks)/d.queries:0);
                row("zero_peak_check_queries",d.checkHistogram[0]);
                uint64_t sum=0;const char* labels[]={"0","1","2","3-4","5-8","9-16","17-32","33-64","65+"};
                for(int k=0;k<9;++k){sum+=d.checkHistogram[k];row((std::string("checks_hist_")+labels[k]).c_str(),d.checkHistogram[k]);}
                if(sum!=d.queries)throw std::runtime_error("Histogram conservation failed");
            } else {
                row("optix_trace_calls",d.traces);row("no_handle_queries",d.noHandles);
                row("bvh_node_visits","unavailable");
                if(d.traces+d.noHandles!=d.queries)throw std::runtime_error("Trace conservation failed");
            }
        }
        diagnostic.close();
#endif
        std::ofstream report(output/(DiagnosticBuild?"diagnostic_timings_NOT_PERFORMANCE.tsv":ProfilingBuild?"profiling_timings_NOT_PERFORMANCE.tsv":"comparison.tsv"));report<<std::setprecision(12);
        report<<(ProfilingBuild?"backend\tgroups\tqueries\tmedian_PROFILED_seconds":"backend\tgroups\tqueries\tmedian_matching_seconds")<<"\tmin_seconds\tmax_seconds\tqueries_per_second\tpeak_hits\tchecksum\n";
        for(int b=0;b<2;++b)report<<(b==0?"cuda":"rt-custom")<<'\t'<<replayGroups*replicas<<'\t'<<replayQueries<<'\t'
            <<median(totals[b])<<'\t'<<*std::min_element(totals[b].begin(),totals[b].end())<<'\t'
            <<*std::max_element(totals[b].begin(),totals[b].end())<<'\t'<<replayQueries/median(totals[b])
            <<'\t'<<hits[b]<<'\t'<<checksums[b]<<'\n';
        std::ofstream result(output/"result.json");result<<std::setprecision(17)
            <<"{\n\"performance_valid\":"<<(ProfilingBuild?"false":"true")<<",\n\"nvtx_enabled\":"<<(NvtxEnabled?"true":"false")<<",\n\"diagnostic_build\": "<<(DiagnosticBuild?"true":"false")<<",\n\"complete\":true,\n\"scan_count\":"<<scansN*replicas<<",\n\"candidate_count\":"<<groupsN*replicas
            <<",\n\"query_count\":"<<replayQueries<<",\n\"repeats\":"<<repeats<<",\n\"warmup_per_chunk\":"<<warmup
            <<",\n\"chunk_groups\":"<<chunk<<",\n\"fragment_tolerance_da\":"<<tolerance
            <<",\n\"peak_index_mismatches\":"<<(aggregate?std::string("null"):std::to_string(mismatches))<<",\n\"class_mismatches\":"<<(aggregate?std::string("null"):std::to_string(classMismatches))
            <<",\n\"candidate_summary_mismatches\":"<<summaryMismatches
#ifdef MVH_MATCHING_LINEAR
            <<",\n\"cuda_search_method\":\"linear_all_peaks\""
#else
            <<",\n\"cuda_search_method\":\"integer_bins\""
#endif
            <<",\n\"output_mode\":\""<<(aggregate?"candidate":"queries")<<"\""
            <<",\n\"theoretical_output_bytes\":"<<(aggregate?replayGroups*replicas*sizeof(mvh_matching::CandidateSummary):replayQueries*4)
            <<",\n\"replicas\":"<<replicas<<",\n\"peak_count\":"<<peaksN*replicas
#if !defined(MVH_MATCHING_DIAGNOSTICS) && !defined(MVH_MATCHING_NVTX)
            <<",\n\"cuda_over_rt\":"<<median(totals[0])/median(totals[1])
#endif
            <<",\n\"rt_build_seconds\":"<<buildRt<<",\n\"common_index_upload_seconds\":"<<uploadIndex
            <<",\n\"query_upload_seconds\":"<<uploadQueries
            <<",\n\"minimum_free_vram_bytes_at_chunk_boundaries\":"<<minFree
            <<",\n\"total_vram_bytes\":"<<totalMemory<<"\n}\n";
        report.close();samples.close();result.close();examples.close();
        if(ProfilingBuild)std::cout<<"PROFILING COMPLETE; instrumented timings are not performance results"<<std::endl;
        else std::cout<<"REPLAY COMPLETE speedup="<<median(totals[0])/median(totals[1])<<std::endl;
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
