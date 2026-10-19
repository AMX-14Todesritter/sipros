#pragma once
#include "capture.h"
#include "scoring.cuh"
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace mvh_matching {
namespace {
bool captureEnabled = false, captureDone = false;
std::filesystem::path captureDirectory;
using namespace mvh_cuda;

struct CaptureSink {
    const Scan& scan;
    double* output;
    uint64_t count = 0;
    __device__ void add(double mz) {
        // Exactly the filtering done by IonCounter::add, before lookup.
        if (mz < scan.lower || mz > scan.upper) return;
        if (output) output[count] = mz;
        ++count;
    }
};

__global__ void captureQueries(const Scan* scans, const Candidate* candidates,
    int first, int count, const PeptideInput* peptides, const char* texts,
    Config cfg, const uint64_t* cachedOffsets, const int* cachedValid,
    const double* cachedIons, int stride, const uint64_t* outputOffsets,
    double* output, uint64_t* counts, unsigned char* validity) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Candidate c = candidates[first+i];
    const Scan& scan = scans[c.scanId];
    if (scan.skip) { counts[i]=0; validity[i]=2; return; }
    CaptureSink sink{scan, output ? output+outputOffsets[i] : nullptr};
    bool valid;
    if (stride && c.charge < stride) {
        const uint64_t key=uint64_t(c.peptideId)*stride+c.charge;
        valid=cachedValid[key]>0;
        for(uint64_t j=cachedOffsets[key];valid && j<cachedOffsets[key+1];++j)
            sink.add(cachedIons[j]);
    } else {
        valid=CalculateSequenceIons(texts+peptides[c.peptideId].text,c.charge,cfg,sink);
    }
    counts[i]=sink.count;
    validity[i]=valid ? 1 : 0;
}

template<class T> void writeValues(std::ofstream& out, const std::vector<T>& v) {
    if(!v.empty()) out.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
}
std::ofstream binaryFile(const char* name) {
    std::ofstream f;
    f.exceptions(std::ios::badbit|std::ios::failbit);
    f.open(captureDirectory/name,std::ios::binary);
    return f;
}
template<class T> void dumpDevice(const char* name, const T* values, size_t count) {
    auto f=binaryFile(name);
    std::vector<T> block;
    for(size_t i=0;i<count;) {
        const size_t n=std::min(count-i,size_t(1048576));block.resize(n);
        check(cudaMemcpy(block.data(),values+i,n*sizeof(T),cudaMemcpyDeviceToHost));
        writeValues(f,block);i+=n;
    }
    f.close();
}
}

void setCaptureEnabled(bool enabled) { captureEnabled=enabled; }
void startCapture(const std::string& runDirectory) {
    captureDirectory=std::filesystem::path(runDirectory)/"matching_capture";
    captureDone=false;
}

// Streams a complete scoring batch using bounded temporary query storage.
// This intentionally runs outside all matching performance measurements.
bool captureBatch(const std::vector<mvh_cuda::Scan>& hostScans,
    const mvh_cuda::Scan* scans, const mvh_cuda::Candidate* candidates, int size,
    const mvh_cuda::PeptideInput* peptides, size_t peptideCount, const char* texts,
    const double* peaks, const int* classes, size_t peakCount,
    const short* hub, size_t hubCount, const mvh_cuda::Config& cfg,
    const uint64_t* cachedOffsets, const int* cachedValid,
    const double* cachedIons, int stride, size_t textCount, const double* lnTable, size_t lnCount) {
    using namespace mvh_cuda;
    if(!captureEnabled || captureDone || !size) return false;
    const uint16_t endian=1;
    if(*reinterpret_cast<const unsigned char*>(&endian)!=1)
        throw std::runtime_error("Capture v1 requires a little-endian host");
    if(captureDirectory.empty() || !std::filesystem::create_directory(captureDirectory))
        throw std::runtime_error("Matching capture directory is not new");
    static_assert(sizeof(double)==8 && sizeof(int)==4 && sizeof(short)==2);
    static_assert(sizeof(Candidate)==16);
    // A partial write never creates COMPLETE.json.
    dumpDevice("peaks.f64",peaks,peakCount);
    dumpDevice("classes.i32",classes,peakCount);
    dumpDevice("hub.i16",hub,hubCount);
    // ABI-tagged extension for controlled generation/matching/MVH replays.
    dumpDevice("fusion_peptides.bin",peptides,peptideCount);
    dumpDevice("fusion_texts.bin",texts,textCount);
    dumpDevice("fusion_ln.f64",lnTable,lnCount);
    {auto f=binaryFile("fusion_config.bin");f.write(reinterpret_cast<const char*>(&cfg),sizeof(cfg));}
    {auto f=binaryFile("fusion_scans.bin");writeValues(f,hostScans);}
    {std::ofstream f(captureDirectory/"fusion_abi.json");
     f<<"{\"version\":1,\"scan_bytes\":"<<sizeof(Scan)
      <<",\"config_bytes\":"<<sizeof(Config)<<",\"peptide_bytes\":"<<sizeof(PeptideInput)<<"}\n";}
    auto sf=binaryFile("scans.bin");
    for(const auto& s:hostScans) {
        const uint64_t offsets[]={s.peakOffset,s.hubOffset};
        const int32_t fields[]={s.peaks,s.lowest,s.highest,s.skip};
        const double bounds[]={s.lower,s.upper};
        sf.write(reinterpret_cast<const char*>(offsets),sizeof(offsets));
        sf.write(reinterpret_cast<const char*>(fields),sizeof(fields));
        sf.write(reinterpret_cast<const char*>(bounds),sizeof(bounds));
    }
    sf.close();
    auto cf=binaryFile("candidates.i32x4");
    auto of=binaryFile("query_offsets.u64");
    auto qf=binaryFile("query_mz.f64");
    auto vf=binaryFile("validity.u8");
    uint64_t total=0;of.write(reinterpret_cast<const char*>(&total),sizeof(total));
    constexpr int chunkSize=32768;
    Buffer<uint64_t> counts, deviceOffsets;
    Buffer<unsigned char> flags;
    Buffer<double> queryBuffer;
    std::vector<Candidate> hostCandidates;
    uint64_t validCount=0, invalidCount=0, skippedCount=0;
    for(int first=0;first<size;first+=chunkSize) {
        const int n=std::min(chunkSize,size-first);
        counts.resize(n);flags.resize(n);
        captureQueries<<<(n+127)/128,128>>>(scans,candidates,first,n,peptides,texts,cfg,
            cachedOffsets,cachedValid,cachedIons,stride,nullptr,nullptr,counts.p,flags.p);
        synced();
        std::vector<uint64_t> hostCounts;counts.read(hostCounts);
        std::vector<uint64_t> offsets(n+1,0);
        for(int i=0;i<n;++i) offsets[i+1]=offsets[i]+hostCounts[i];
        deviceOffsets.upload(offsets);queryBuffer.resize(offsets.back());
        captureQueries<<<(n+127)/128,128>>>(scans,candidates,first,n,peptides,texts,cfg,
            cachedOffsets,cachedValid,cachedIons,stride,deviceOffsets.p,queryBuffer.p,
            counts.p,flags.p);
        synced();
        std::vector<uint64_t> written;counts.read(written);
        if(written!=hostCounts) throw std::runtime_error("Capture count/store mismatch");
        std::vector<double> queries;queryBuffer.read(queries);writeValues(qf,queries);
        std::vector<unsigned char> valid;flags.read(valid);writeValues(vf,valid);
        for(auto v:valid) {validCount+=v==1;invalidCount+=v==0;skippedCount+=v==2;}
        hostCandidates.resize(n);
        check(cudaMemcpy(hostCandidates.data(),candidates+first,n*sizeof(Candidate),cudaMemcpyDeviceToHost));
        writeValues(cf,hostCandidates);
        for(int i=1;i<=n;++i) offsets[i]+=total;
        of.write(reinterpret_cast<const char*>(offsets.data()+1),n*sizeof(uint64_t));
        total=offsets.back();
        if(first==0 || first/chunkSize%64==0 || first+n==size)
            std::cout<<"[Matching capture] candidates="<<first+n<<'/'<<size
                     <<" queries="<<total<<std::endl;
    }
    cf.close();of.close();qf.close();vf.close();
    std::ofstream manifest;
    manifest.exceptions(std::ios::badbit|std::ios::failbit);
    manifest.open(captureDirectory/"COMPLETE.json");
    manifest<<std::setprecision(17)<<"{\n  \"format_version\": 1,\n  \"endianness\": \"little\",\n"
        <<"  \"scope\": \"first nonempty scoring batch, all candidates before top filtering\",\n"
        <<"  \"scan_count\": "<<hostScans.size()<<",\n  \"peak_count\": "<<peakCount
        <<",\n  \"hub_count\": "<<hubCount<<",\n  \"peptide_count\": "<<peptideCount
        <<",\n  \"candidate_count\": "<<size<<",\n  \"query_count\": "<<total
        <<",\n  \"valid_candidates\": "<<validCount<<",\n  \"invalid_candidates\": "<<invalidCount
        <<",\n  \"skipped_candidates\": "<<skippedCount<<",\n  \"cache_charge_stride\": "<<stride
        <<",\n  \"fragment_tolerance_da\": "<<cfg.fragmentTolerance<<"\n}\n";
    manifest.close();captureDone=true;
    std::cout<<"[Matching capture] complete: "<<captureDirectory<<std::endl;
    return true;
}
}
