#pragma once
#ifdef MVH_ENABLE_FLOW_COUNTERS
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <stdexcept>
namespace mvh_flow {
struct Timing { double seconds=0; uint64_t calls=0; };
inline std::map<std::string,uint64_t> totals;
inline std::map<std::string,Timing> timings;
inline std::map<uint64_t,uint64_t> peptideReuseHistogram;
inline void time(const char *name,double seconds) { auto &t=timings[name];t.seconds+=seconds;++t.calls; }
inline void reset() { totals.clear(); timings.clear(); peptideReuseHistogram.clear(); }
inline void write(const std::string &path,double searchSeconds,uint64_t finalPsms) {
    totals["total_final_psm_candidates"]=finalPsms;
    std::ofstream reuse(path+"/peptide_reuse_histogram.tsv");
    reuse.exceptions(std::ios::failbit|std::ios::badbit);
    reuse<<"scan_associations\tpeptide_entries\n";
    for(const auto &row:peptideReuseHistogram) reuse<<row.first<<'\t'<<row.second<<'\n';
    std::ofstream c(path+"/flow_counters.tsv"),t(path+"/flow_timings.tsv");
    c.exceptions(std::ios::failbit|std::ios::badbit);t.exceptions(std::ios::failbit|std::ios::badbit);
    c<<"counter\tvalue\n";for(const auto &p:totals)c<<p.first<<'\t'<<p.second<<'\n';
    t<<std::setprecision(17)<<"stage_name\ttotal_time_seconds\tpercentage_of_search_time\tinvocation_count\taverage_time_per_invocation\ttiming_kind\n";
    for(const auto &p:timings)t<<p.first<<'\t'<<p.second.seconds<<'\t'
        <<(searchSeconds?100*p.second.seconds/searchSeconds:0)<<'\t'<<p.second.calls<<'\t'
        <<p.second.seconds/p.second.calls<<'\t'
        <<(p.first=="gpu/scoring_fused"?"cuda_event":"host_inclusive")<<'\n';
    for(const char *name:{"theoretical_generation_inside_scoring","fragment_lookup_inside_scoring","mvh_arithmetic_inside_scoring"})
        t<<name<<"\tNA\tNA\tNA\tNA\tfused_not_separately_timed\n";
    t<<"XCorr_WDP\t0\t0\t0\tNA\tnot_executed\n";
}
}
#endif
