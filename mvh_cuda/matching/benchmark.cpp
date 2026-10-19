#include "../matching/matching.cuh"
#ifdef MATCHING_ENABLE_RT
#include "bridge.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <omp.h>
#include <sstream>

using Clock = std::chrono::steady_clock;
using namespace mvh_cuda;
using mvh_matching::Query;
static double seconds(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now()-t).count();
}
struct Event {
    cudaEvent_t value;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};
template<class F> double gpuTime(F f) {
    Event start, end;
    check(cudaEventRecord(start.value));
    f();
    check(cudaEventRecord(end.value));
    check(cudaEventSynchronize(end.value));
    float ms; check(cudaEventElapsedTime(&ms, start.value, end.value));
    return ms * 1e-3;
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size()%2 ? v[v.size()/2] : (v[v.size()/2-1]+v[v.size()/2])/2;
}
struct RtLifetime {
    ~RtLifetime() {
#ifdef MATCHING_ENABLE_RT
        mvh_rt_gpu::reset();
#endif
    }
};
int main(int argc, char** argv) {
    try {
        std::string peaksPath, queriesPath, output;
        int repeats=11, threads=1;
        double tolerance=0.01;
        bool contract=false;
        for (int i=1; i<argc; ++i) {
            const std::string key=argv[i];
            if (key=="--contract") { contract=true; continue; }
            if (i+1>=argc) throw std::invalid_argument("Missing value: "+key);
            const std::string value=argv[++i];
            if(key=="--peaks") peaksPath=value;
            else if(key=="--queries") queriesPath=value;
            else if(key=="--output") output=value;
            else if(key=="--repeats") repeats=std::stoi(value);
            else if(key=="--threads") threads=std::stoi(value);
            else if(key=="--tolerance") tolerance=std::stod(value);
            else throw std::invalid_argument("Unknown option: "+key);
        }
        if (repeats<1 || threads<1 || !std::isfinite(tolerance) || tolerance<=0)
            throw std::invalid_argument("Invalid repeats, threads or tolerance");
        if ((!peaksPath.empty()) != (!queriesPath.empty()))
            throw std::invalid_argument("Provide both --peaks and --queries");
        if (!contract && output.empty()) throw std::invalid_argument("Specify --output under output/");
        omp_set_num_threads(threads);
        std::vector<std::vector<double>> byScan;
        std::vector<Query> queries;
        if (peaksPath.empty()) {
            byScan.resize(2);
            for (int s=0;s<2;++s)
                for(int i=0;i<512;++i) byScan[s].push_back(100.125+i*0.125+s*0.03125);
            for(int i=0;i<(contract?256:100000);++i) {
                const int s=i%2, p=(i/2)%512;
                queries.push_back({s,byScan[s][p]+(i%3==0?0.003:i%3==1?0.04:0.0)});
            }
        } else {
            auto read=[&](const std::string& path, auto consume) {
                std::ifstream f(path);
                if(!f) throw std::runtime_error("Cannot read "+path);
                std::string line; size_t number=0;
                while(std::getline(f,line)) {
                    ++number;
                    if(line.empty() || line[0]=='#') continue;
                    int scan; double mz; std::string extra;
                    std::istringstream row(line);
                    if(!(row>>scan>>mz) || row>>extra || scan<0 || scan>1000000 ||
                       !std::isfinite(mz) || mz<0 || mz>1000000)
                        throw std::runtime_error("Invalid scan/mz at "+path+":"+std::to_string(number));
                    consume(scan,mz);
                }
            };
            read(peaksPath,[&](int s,double mz) {
                if(byScan.size()<=size_t(s)) byScan.resize(size_t(s)+1);
                byScan[s].push_back(mz);
            });
            read(queriesPath,[&](int s,double mz) { queries.push_back({s,mz}); });
        }
        if(queries.empty() || queries.size()>size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("Empty or oversized query list");
        const auto indexStart=Clock::now();
        std::vector<Scan> scans(byScan.size());
        std::vector<double> peaks;
        std::vector<int> classes;
        std::vector<short> hub;
        for(size_t s=0;s<byScan.size();++s) {
            auto& values=byScan[s];
            if(values.size()>size_t(std::numeric_limits<short>::max()))
                throw std::runtime_error("Scan exceeds current short hub index limit");
            if(!std::is_sorted(values.begin(),values.end()))
                throw std::runtime_error("Peaks must already be sorted; local peak IDs are preserved");
            Scan& scan=scans[s]; scan.peakOffset=peaks.size(); scan.hubOffset=hub.size();
            scan.peaks=values.size(); scan.lower=0; scan.upper=1000001;
            if(values.empty()) { scan.lowest=0; scan.highest=-1; continue; }
            scan.lowest=int(values.front()); scan.highest=int(values.back());
            hub.resize(hub.size()+2*size_t(scan.highest-scan.lowest+1),-1);
            for(size_t i=0;i<values.size();++i) {
                const size_t bucket=scan.hubOffset+2*size_t(int(values[i])-scan.lowest);
                if(hub[bucket]<0) hub[bucket]=short(i);
                hub[bucket+1]=short(i+1);
                peaks.push_back(values[i]); classes.push_back(int(i%4));
            }
        }
        for(auto q:queries)
            if(size_t(q.scanId)>=scans.size()) throw std::runtime_error("Query references absent scan");
        const double cpuBuild=seconds(indexStart);
        std::vector<int> expected(queries.size());
        auto cpu=[&] {
            #pragma omp parallel for schedule(static)
            for(int i=0;i<int(queries.size());++i) {
                const auto q=queries[i];
                expected[i]=mvh_matching::nearestPeak(q.mz,tolerance,scans[q.scanId],peaks.data(),hub.data());
            }
        };
        cpu();
        if(contract) {
            auto oracle=[&](const Scan& scan,double mz,double tol) {
                int best=-1; double distance=tol;
                for(int i=0;i<scan.peaks;++i) {
                    const double d=std::abs(mz-peaks[scan.peakOffset+i]);
                    if(d<distance) { best=i; distance=d; }
                }
                return best;
            };
            for(size_t i=0;i<queries.size();++i)
                if(expected[i]!=oracle(scans[queries[i].scanId],queries[i].mz,tolerance))
                    throw std::runtime_error("CPU lookup differs from exhaustive oracle");
            // Exactly representable strict boundary, equidistant tie and empty scan.
            const double p[]={100.0,100.25}; const short h[]={0,2};
            Scan test{};test.peaks=2;test.lowest=test.highest=100;
            if(mvh_matching::nearestPeak(100.125,0.125,test,p,h)!=-1 ||
               mvh_matching::nearestPeak(100.125,0.25,test,p,h)!=0 ||
               mvh_matching::nearestPeak(100.0,0.01,test,p,h)!=0)
                throw std::runtime_error("Strict boundary/tie/exact match failed");
            test.peaks=0;
            if(mvh_matching::nearestPeak(100,0.01,test,p,h)!=-1)
                throw std::runtime_error("Empty scan contract failed");
        }
        const auto uploadStart=Clock::now();
        Buffer<Scan> ds(scans); Buffer<double> dp(peaks); Buffer<int> dc(classes);
        Buffer<short> dh(hub); Buffer<Query> dq(queries); Buffer<int> result(queries.size());
        synced(); const double upload=seconds(uploadStart);
        // Destroy RT before its referenced device buffers, including on exceptions.
        RtLifetime lifetime;
        Config cfg{}; cfg.classes=3; cfg.fragmentTolerance=tolerance;
        double rtBuild=0;
#ifdef MATCHING_ENABLE_RT
        auto start=Clock::now();
        mvh_rt_gpu::prepare(scans,ds.p,dp.p,dc.p,peaks.size(),mvh_rt_gpu::GeometryKind::Spheres,cfg);
        synced(); rtBuild=seconds(start);
        mvh_rt_gpu::stageMatching(ds.p,dq.p,queries.size(),result.p,cfg);
        synced();
#endif
        std::ofstream file;
        if(!output.empty()) { file.open(output); if(!file) throw std::runtime_error("Cannot write output"); }
        std::ostream& report=output.empty()?std::cout:file;
        report<<std::setprecision(12);
        report<<"backend\tqueries\tmedian_seconds\tmin_seconds\tmax_seconds\tqueries_per_second\tmismatches\tindex_build_seconds\tupload_seconds\tchecksum\n";
        for(const std::string backend : {"cpu","cuda","rt-custom"}) {
#ifndef MATCHING_ENABLE_RT
            if(backend=="rt-custom") continue;
#endif
            auto gpu=[&] {
                if(backend=="cuda") mvh_matching::launchCuda(ds.p,dp.p,dh.p,dq.p,queries.size(),tolerance,result.p);
#ifdef MATCHING_ENABLE_RT
                else mvh_rt_gpu::launchStagedMatching();
#endif
            };
            for(int i=0;i<3;++i) { if(backend=="cpu") cpu(); else {gpu();synced();} }
            std::vector<double> times;
            for(int i=0;i<repeats;++i) {
                if(backend=="cpu") {auto start=Clock::now();cpu();times.push_back(seconds(start));}
                else times.push_back(gpuTime(gpu));
            }
            std::vector<int> observed;
            if(backend=="cpu") observed=expected; else result.read(observed);
            size_t mismatches=0; uint64_t checksum=1469598103934665603ULL;
            for(size_t i=0;i<observed.size();++i) {
                mismatches+=observed[i]!=expected[i]; checksum=(checksum^uint64_t(observed[i]+1))*1099511628211ULL;
            }
            if(contract && mismatches) throw std::runtime_error(backend+" differs on non-boundary contract queries");
            report<<backend<<'\t'<<queries.size()<<'\t'<<median(times)<<'\t'
                  <<*std::min_element(times.begin(),times.end())<<'\t'
                  <<*std::max_element(times.begin(),times.end())<<'\t'
                  <<queries.size()/median(times)<<'\t'<<mismatches<<'\t'
                  <<(backend=="rt-custom"?rtBuild:cpuBuild)<<'\t'
                  <<(backend=="cpu"?0:upload)<<'\t'<<checksum<<'\n';
        }
        std::cout<<"PASS: matching-only replay; threads="<<threads<<" repeats="<<repeats<<'\n';
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
