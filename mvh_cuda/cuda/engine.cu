#include "engine.h"
#include "profiling.h"
#include "sequence_ids.h"
#include "score_impact.h"
#ifdef MVH_CUDA_ENABLE_RT
#include "bridge.h"
#endif
#include "ms2scanvector.h"
#include "preprocess.cuh"
#include "scoring.cuh"
#include "assignment.cuh"
#include "theoretical.cuh"
#include <cub/cub.cuh>
#include <memory>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <limits>

namespace mvh_cuda {
namespace {
void resetScoringSpectra();
bool verification=false;
bool scoreImpactEnabled = false;
std::unique_ptr<ScoreImpactWriter> scoreImpactWriter;
std::string matchBackend="cuda";
int batchSize=2000000;
bool useIonCache=true;
bool keepSpectraOnDevice=false;
struct PreparedPeptides {
    std::unique_ptr<Buffer<char>> texts;
    std::vector<uint64_t> offsets;
};
std::unique_ptr<PreparedPeptides> preparedPeptides;

// Pure RT needs sorted masses/classes, but neither PeakList nor its mass hub.
// These compact arrays live for one dataset, in the same order as its scans.
struct UnindexedSpectrum {
    const MS2Scan *owner = nullptr;
    std::vector<double> masses;
    std::vector<char> classes;
};
std::vector<UnindexedSpectrum> unindexedSpectra;
bool deviceReady=false;
double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error("CUDA MVH: " + message);
}
// Avoid allocating a temporary std::string for successful checks in hot loops.
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(std::string("CUDA MVH: ") + message);
}
// Bucket lookup is a property of the selected matcher, not MVH scoring itself.
// Resolve it outside GPU kernels; the standalone CUDA build always uses buckets.
bool needsDeviceBuckets() {
#ifdef MVH_CUDA_ENABLE_RT
    return matchBackend == "cuda" || matchBackend == "rt-audit" || scoreImpactEnabled;
#else
    return true;
#endif
}

bool needsHostBuckets() {
    // CPU score verification still calls the original PeakList::findNear.
    return needsDeviceBuckets() || verification;
}

void storeProcessedSpectrum(MS2Scan &scan, std::map<double, char> &peaks,
                            UnindexedSpectrum *unindexed) {
    MVH_PROFILE_SCOPE("mvh/function/storeProcessedSpectrum");
    delete scan.peakData;
    scan.peakData = nullptr;
    // Preserve CUDA's original release/allocation order as well as its data.
    delete scan.intenClassCounts;
    delete scan.pPeakList;
    scan.pPeakList = nullptr;
    scan.intenClassCounts = new std::vector<int>();
    if (!unindexed) {
        // Preserve the original constructor and lookup layout for CUDA/audit.
        scan.pPeakList = new PeakList(&peaks);
        return;
    }

    unindexed->owner = &scan;
    unindexed->masses.reserve(peaks.size());
    unindexed->classes.reserve(peaks.size());
    for (const auto &peak : peaks) {
        unindexed->masses.push_back(peak.first);
        unindexed->classes.push_back(peak.second);
    }
}

const UnindexedSpectrum &unindexedSpectrum(size_t index, const MS2Scan *scan) {
    require(index < unindexedSpectra.size() && unindexedSpectra[index].owner == scan,
            "RT spectrum missing or scan order changed after preprocessing");
    return unindexedSpectra[index];
}

void initialize(){
    MVH_PROFILE_SCOPE("mvh/function/initialize");
    if(deviceReady)return;int count=0;check(cudaGetDeviceCount(&count));require(count>0,"No CUDA GPU available; no silent CPU fallback");
    cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));check(cudaSetDevice(0));
    std::cout<<"[CUDA device] "<<prop.name<<"; OpenMP compute loops disabled\n";
    // Audit tie ordering against the actual host libstdc++ implementation.
    std::vector<Top> cases(1000*TopN);unsigned seed=729;
    for(int c=0;c<1000;++c)for(int k=0;k<TopN;++k){seed=1664525*seed+1013904223;cases[c*TopN+k]={double((seed>>16)%(c%51+1)),k};}
    auto expected=cases;
    for(int c=0;c<1000;++c)std::sort(expected.begin()+c*TopN,expected.begin()+(c+1)*TopN,[](const Top&a,const Top&b){return a.score>b.score;});
    Buffer<Top> gpu(cases);sortTest<<<(1000+127)/128,128>>>(gpu.p,1000);synced();gpu.read(cases);
    for(size_t i=0;i<cases.size();++i)require(cases[i].score==expected[i].score&&cases[i].sequenceId==expected[i].sequenceId,"GPU top sort differs from host std::sort");
    deviceReady=true;
}
Config configuration(){
    MVH_PROFILE_SCOPE("mvh/function/configuration");
    Config c{};c.classes=ProNovoConfig::NumIntensityClasses;c.minClassCount=ProNovoConfig::minIntensityClassCount;
    c.maxPeaks=ProNovoConfig::MaxPeakCount;c.minMatched=ProNovoConfig::MinMatchedFragments;
    c.minLength=ProNovoConfig::getMinPeptideLength();c.smart=MVH::bUseSmartPlusThreeModel;
    c.bIon=MVH::fragmentTypes[FragmentType_B];c.yIon=MVH::fragmentTypes[FragmentType_Y];
    c.tic=ProNovoConfig::ticCutoffPercentage;c.multiplier=ProNovoConfig::ClassSizeMultiplier;
    c.fragmentTolerance=ProNovoConfig::getMassAccuracyFragmentIon();c.parentTolerance=ProNovoConfig::getMassAccuracyParentIon();
    c.mzLow=ProNovoConfig::minObservedMz;c.mzHigh=ProNovoConfig::maxObservedMz;
    c.water=WATER_MONO;c.proton=Proton;c.yWater=ProNovoConfig::precalcMasses.dCtermOH2;
    c.missingMass=ProNovoConfig::pdAAMassFragment.end();
    for(int i=0;i<256;++i)c.mass[i]=ProNovoConfig::pdAAMassFragment.vdMasses.at(i);
    require(c.classes>0&&c.classes<=MaxClasses,"intensity classes outside supported range 1..8");
    require(c.maxPeaks>0&&c.maxPeaks<=32767,"peak count exceeds original short index capacity");
    require(c.fragmentTolerance>0&&c.minClassCount>0,"invalid tolerance/class count");return c;
}
template<class T> void append(std::vector<T>&dest,const std::vector<T>&src){dest.insert(dest.end(),src.begin(),src.end());}
void same(double a,double b,const std::string &name){require(a==b,name+" differs: "+std::to_string(a)+" vs "+std::to_string(b));}
}
void setVerification(bool enabled){verification=enabled;}
void setScoreImpact(bool enabled) {
    require(!enabled || matchBackend == "rt-triangle" || matchBackend == "rt-instanced" ||
            matchBackend == "rt-custom",
            "--score-impact requires rt-triangle, rt-instanced or rt-custom");
    scoreImpactEnabled = enabled;
}
void startScoreImpact(const std::string &outputDirectory) {
    if (scoreImpactEnabled) scoreImpactWriter = std::make_unique<ScoreImpactWriter>(outputDirectory);
}

MatchBackendScope::~MatchBackendScope() {
    MVH_PROFILE_SCOPE("mvh/function/MatchBackendScope::~MatchBackendScope");
    scoreImpactWriter.reset();
    resetScoringSpectra();
    unindexedSpectra.clear();
#ifdef MVH_CUDA_ENABLE_RT
    mvh_rt_gpu::reset();
#endif
}
const std::string &matchBackendName() { return matchBackend; }
void setMatchBackend(const std::string &name) {
    require(name=="cuda" || name=="rt-triangle" || name=="rt-audit" || name=="rt-instanced" || name=="rt-custom", "Unknown match backend");
#ifndef MVH_CUDA_ENABLE_RT
    require(name=="cuda", "RT backend was not enabled at build time");
#endif
    matchBackend=name;
    if (name=="rt-custom") std::cout << "[RT backend] rt-custom; built-in spheres at (mz, class, 0); higher class number first\n";
    else if (name!="cuda") std::cout << "[RT backend] " << name
        << "; experimental unshifted triangles; known zero-distance/boundary differences\n";
}
void setPeptideBatchSize(int size) { require(size > 0, "batch size must be positive"); batchSize = size; }
int peptideBatchSize() { return batchSize; }
void setSpectrumDeviceCache(bool enabled) { keepSpectraOnDevice = enabled; }
bool spectrumDeviceCache() { return keepSpectraOnDevice; }

void preProcessAllMs2Mvh(std::vector<MS2Scan *> &scans){
    MVH_PROFILE_SCOPE("mvh/function/preProcessAllMs2Mvh");
    resetScoringSpectra();
#ifdef MVH_CUDA_ENABLE_RT
    mvh_rt_gpu::reset();
#endif
    initialize();double start=now();Config cfg=configuration();int n=scans.size();require(n>0,"empty scan collection");
    MVH_PROFILE_BEGIN(rawRange, "mvh/preprocess/scans/pack_host");
    std::vector<RawScan> raw;std::vector<double> mz,intensity;std::vector<int> charges;
    for(auto *s:scans){require(s->vdMZ.size()==s->vdIntensity.size()&&s->vdMZ.size()==s->viCharge.size(),"raw peak array lengths");
        require(s->vdMZ.size()<=std::numeric_limits<int>::max(),"scan too large");
        raw.push_back({mz.size(),int(s->vdMZ.size()),s->iParentChargeState,s->dParentNeutralMass,s->dParentMZ});
        append(mz,s->vdMZ);append(intensity,s->vdIntensity);append(charges,s->viCharge);
    }
    MVH_PROFILE_END(rawRange);
    std::vector<PrepResult> result;std::vector<double> kept;std::vector<int> classes;
    {
        Buffer<RawScan> dRaw(raw);Buffer<double>dMz(mz),dIntensity(intensity);Buffer<int>dCharge(charges),indices(mz.size());
        Buffer<PrepResult>dResult(n);Buffer<double>dKept(size_t(n)*cfg.maxPeaks);Buffer<int>dClasses(size_t(n)*cfg.maxPeaks);
        MVH_PROFILE_BEGIN(prepRange, "mvh/preprocess/scans/preprocess_and_wait");
        preprocessMvh<<<(n+127)/128,128>>>(dRaw.p,n,dMz.p,dIntensity.p,dCharge.p,indices.p,dResult.p,dKept.p,dClasses.p,cfg);synced();
        MVH_PROFILE_END(prepRange);
        MVH_PROFILE_BEGIN(sumRange, "mvh/preprocess/scans/sum_intensity_and_wait");
        sumIntensity<<<(n+127)/128,128>>>(dRaw.p,n,dIntensity.p,dResult.p);synced();
        MVH_PROFILE_END(sumRange);
        dMz.read(mz);dIntensity.read(intensity);dCharge.read(charges);dResult.read(result);dKept.read(kept);dClasses.read(classes);
    }
    MVH_PROFILE_BEGIN(restoreRange, "mvh/preprocess/scans/restore_and_verify");
    const bool buildHostBuckets = needsHostBuckets();
    unindexedSpectra.clear();
    if (!buildHostBuckets) unindexedSpectra.resize(scans.size());
    size_t hostBucketEntries = 0;
    int maxBins=0;std::multimap<double,double> workspace;
    for(int i=0;i<n;++i){auto *s=scans[i];const auto &r=result[i];const auto &in=raw[i];
        std::map<double,char> peakMap;for(int k=0;k<r.count;++k)peakMap[kept[size_t(i)*cfg.maxPeaks+k]]=classes[size_t(i)*cfg.maxPeaks+k];
        if(verification){MS2Scan ref;ref.vdMZ=s->vdMZ;ref.vdIntensity=s->vdIntensity;ref.viCharge=s->viCharge;
            ref.dParentMZ=s->dParentMZ;ref.dParentNeutralMass=s->dParentNeutralMass;ref.iParentChargeState=s->iParentChargeState;
            ref.preprocessMvh(&workspace);ref.sumIntensity();
            require(ref.bSkip==bool(r.skip),"preprocess skip mismatch at scan "+std::to_string(i));
            require(ref.vdMZ==std::vector<double>(mz.begin()+in.offset,mz.begin()+in.offset+in.count),"raw sort mismatch");
            require(ref.vdIntensity==std::vector<double>(intensity.begin()+in.offset,intensity.begin()+in.offset+in.count),"raw intensity sort mismatch");
            require(ref.viCharge==std::vector<int>(charges.begin()+in.offset,charges.begin()+in.offset+in.count),"raw charge sort mismatch");
            PeakList expected(&peakMap);require(ref.pPeakList->pPeaks==expected.pPeaks&&ref.pPeakList->pClasses==expected.pClasses&&ref.pPeakList->pMassHub==expected.pMassHub,"peak/class/bucket mismatch");
            if(!r.skip){require(ref.totalPeakBins==r.totalBins,"total bins mismatch");require(*ref.intenClassCounts==std::vector<int>(r.counts,r.counts+cfg.classes+1),"intensity class counts mismatch");}
            // Host compiler may reassociate raw intensity sums. They do not enter MVH scoring.
            require(std::abs(ref.dSumIntensity-r.sum)<=1e-12*std::max(1.0,std::abs(ref.dSumIntensity)),"intensity sum mismatch");same(ref.dMaxIntensity,r.max,"max intensity");
        }
        s->vdMZ.assign(mz.begin()+in.offset,mz.begin()+in.offset+in.count);
        s->vdIntensity.assign(intensity.begin()+in.offset,intensity.begin()+in.offset+in.count);
        s->viCharge.assign(charges.begin()+in.offset,charges.begin()+in.offset+in.count);
        storeProcessedSpectrum(*s, peakMap, buildHostBuckets ? nullptr : &unindexedSpectra[i]);
        if (s->pPeakList) hostBucketEntries += s->pPeakList->pMassHub.size();
        s->bSkip = r.skip;
        if(in.count>=cfg.minClassCount){s->mzLowerBound=cfg.mzLow;s->mzUpperBound=cfg.mzHigh;}
        if(!r.skip){s->totalPeakBins=r.totalBins;s->intenClassCounts->assign(r.counts,r.counts+cfg.classes+1);maxBins=std::max(maxBins,r.totalBins);}
        s->dSumIntensity=r.sum;s->dMaxIntensity=r.max;
    }
    MVH_PROFILE_END(restoreRange);
    { MVH_PROFILE_SCOPE("mvh/preprocess/scans/initialLnTable"); MVH::initialLnTable(maxBins); }
    std::cout<<"[CUDA preprocessing] scans="<<n<<" seconds="<<now()-start<<" verified="<<verification
             <<" host_bucket_entries="<<hostBucketEntries<<'\n';
}

void preprocessingMVH(std::vector<Peptide *> &peptides){
    MVH_PROFILE_SCOPE("mvh/batch/preprocess_peptides");
    initialize();if(peptides.empty())return;double start=now();
    MVH_PROFILE_BEGIN(textRange, "mvh/preprocess/peptides/pack_host");
    std::vector<Rule> rules;for(const auto &pair:ProNovoConfig::getNeutralLossList()){
        require(!pair.first.empty()&&pair.first.size()<MaxText&&pair.second.size()<MaxText,"neutral loss rule too long");
        Rule r{};r.fromLen=pair.first.size();r.toLen=pair.second.size();std::memcpy(r.from,pair.first.data(),r.fromLen);std::memcpy(r.to,pair.second.data(),r.toLen);rules.push_back(r);
    }
    std::vector<char> texts;std::vector<uint64_t> offsets;std::vector<int> capacities,lengths,errors;
    for(auto *p:peptides){size_t cap=p->sPeptide.size()+1;
        for(const auto &rule:rules)if(rule.toLen>1){require(cap<MaxText,"neutral loss expansion capacity");cap*=rule.toLen;}
        require(cap<=MaxText,"peptide text exceeds CUDA capacity 512");offsets.push_back(texts.size());capacities.push_back(cap);
        texts.resize(texts.size()+cap,0);std::memcpy(texts.data()+offsets.back(),p->sPeptide.data(),p->sPeptide.size());
    }
    MVH_PROFILE_END(textRange);
    int n=peptides.size();
    auto prepared=std::make_unique<PreparedPeptides>();
    prepared->texts=std::make_unique<Buffer<char>>(texts);
    prepared->offsets=offsets;
    Buffer<uint64_t>dOffsets(offsets);Buffer<int>dCap(capacities),dLengths(n),dErrors(n);Buffer<Rule>dRules(rules);
    MVH_PROFILE_BEGIN(peptideKernelRange, "mvh/preprocess/peptides/kernel_and_wait");
    preprocessingMVH<<<(n+127)/128,128>>>(prepared->texts->p,dOffsets.p,dCap.p,n,dRules.p,rules.size(),dLengths.p,dErrors.p);synced();
    MVH_PROFILE_END(peptideKernelRange);
    // Only verification needs neutral-loss strings on the host. Normal search
    // keeps the transformed text resident through theoretical-ion generation.
    MVH_PROFILE_BEGIN(peptideRestoreRange, "mvh/preprocess/peptides/download_and_verify");
    if(verification) prepared->texts->read(texts);
    dLengths.read(lengths);dErrors.read(errors);
    for(int i=0;i<n;++i){require(errors[i]==0,"neutral loss failed or self-repeating rule");require(lengths[i]<=MaxLength,"peptide exceeds 128 residues");
        if(verification){
            std::string neutral(texts.data()+offsets[i]);
            Peptide ref;ref.sPeptide=peptides[i]->sPeptide;ref.preprocessingMVH();
            require(ref.sNeutralLossPeptide==neutral&&ref.iPeptideLength==lengths[i],"peptide preprocessing mismatch");
            peptides[i]->sNeutralLossPeptide=std::move(neutral);
        }
        peptides[i]->iPeptideLength=lengths[i];
    }
    MVH_PROFILE_END(peptideRestoreRange);
    preparedPeptides=std::move(prepared);
    std::cout<<"[CUDA peptide preprocessing] peptides="<<n<<" seconds="<<now()-start<<" verified="<<verification<<'\n';
}

namespace {
struct AssignedBatch {
    std::unique_ptr<Buffer<Candidate>> candidates;
    std::vector<Precursor> precursors;
    size_t peptideCount = 0;
    int maxCharge = 0;
};
std::unique_ptr<AssignedBatch> assignedBatch;

// CUB owns no persistent storage here. Scratch memory is released at the end
// of each operation, so peak memory is bounded by the current peptide batch.
uint64_t exclusiveOffsets(Buffer<uint64_t> &counts, Buffer<uint64_t> &offsets) {
    MVH_PROFILE_SCOPE("mvh/function/exclusiveOffsets");
    size_t bytes = 0;
    check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, counts.p, offsets.p, counts.n));
    Buffer<unsigned char> scratch(bytes);
    check(cub::DeviceScan::ExclusiveSum(scratch.p, bytes, counts.p, offsets.p, counts.n));
    uint64_t total = 0;
    check(cudaMemcpy(&total, offsets.p + offsets.n - 1, sizeof(total), cudaMemcpyDeviceToHost));
    return total;
}
}

void assignPeptides2Scans(const std::vector<Peptide *> &peptides,
                         const std::vector<std::tuple<double, int, MS2Scan *>> &precursors,
                         const std::vector<MS2Scan *> &scans) {
    MVH_PROFILE_SCOPE("mvh/batch/assign_scans");
    initialize();
    const double start = now();
    MVH_PROFILE_BEGIN(assignHostRange, "mvh/assignment/pack_host");
    auto batch = std::make_unique<AssignedBatch>();
    batch->peptideCount = peptides.size();
    std::unordered_map<const MS2Scan *, int> scanIds;
    for (size_t i = 0; i < scans.size(); ++i) scanIds.emplace(scans[i], i);
    batch->precursors.reserve(precursors.size());
    for (const auto &entry : precursors) {
        const int charge = std::get<1>(entry);
        require(charge >= 0, "negative precursor charge");
        batch->precursors.push_back({std::get<0>(entry), scanIds.at(std::get<2>(entry)), charge});
        batch->maxCharge = std::max(batch->maxCharge, charge);
    }
    std::vector<double> masses;
    masses.reserve(peptides.size());
    for (auto *peptide : peptides) masses.push_back(peptide->getPeptideMass());
    std::vector<std::pair<double, double>> originalWindows;
    ProNovoConfig::getPeptideMassWindows(0, originalWindows);
    std::vector<MassWindow> windows;
    for (const auto &window : originalWindows) windows.push_back({window.first, window.second});
    MVH_PROFILE_END(assignHostRange);
    const int size = peptides.size(), windowCount = windows.size();
    Buffer<double> deviceMasses(masses);
    Buffer<Precursor> devicePrecursors(batch->precursors);
    Buffer<MassWindow> deviceWindows(windows);
    Buffer<MassRange> ranges(size_t(size) * windowCount);
    Buffer<int> rangeCounts(size);
    Buffer<uint64_t> counts(size + 1), offsets(size + 1);
    check(cudaMemset(counts.p, 0, counts.n * sizeof(uint64_t)));
    MVH_PROFILE_BEGIN(rangeKernelRange, "mvh/assignment/ranges_submit");
    GetAllRangeFromMass<<<(size + 127) / 128, 128>>>(deviceMasses.p, size,
        devicePrecursors.p, batch->precursors.size(), deviceWindows.p, windowCount,
        ranges.p, rangeCounts.p, counts.p);
    check(cudaGetLastError());
    MVH_PROFILE_END(rangeKernelRange);
    const uint64_t associations = exclusiveOffsets(counts, offsets);
    require(associations <= size_t(std::numeric_limits<int>::max()),
            "candidate batch exceeds CUB int indexing; lower the peptide batch size");
    batch->candidates = std::make_unique<Buffer<Candidate>>(associations);
    std::vector<int> hostRangeCounts;
    rangeCounts.read(hostRangeCounts);
    size_t matchedPeptides = 0;
    for (int i = 0; i < size; ++i) if (hostRangeCounts[i]) {
        ++matchedPeptides;
        ProNovoConfig::dMaxPeptideMass = std::max(ProNovoConfig::dMaxPeptideMass, masses[i]);
    }
    if (associations) {
        MVH_PROFILE_SCOPE("mvh/assignment/fill_and_sort");
        Buffer<Candidate> unsorted(associations);
        Buffer<int> keys(associations), sortedKeys(associations);
        assignPeptides2Scans<<<(size + 127) / 128, 128>>>(ranges.p, rangeCounts.p,
            offsets.p, size, windowCount, devicePrecursors.p, unsorted.p, keys.p);
        check(cudaGetLastError());
        size_t bytes = 0;
        check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys.p, sortedKeys.p,
            unsorted.p, batch->candidates->p, int(associations)));
        Buffer<unsigned char> scratch(bytes);
        // Stable sorting by scan preserves peptide/window/precursor order.
        check(cub::DeviceRadixSort::SortPairs(scratch.p, bytes, keys.p, sortedKeys.p,
            unsorted.p, batch->candidates->p, int(associations)));
    }
    if (verification) {
        MVH_PROFILE_SCOPE("mvh/assignment/verify");
        std::vector<MassRange> actual;
        ranges.read(actual);
        for (int i = 0; i < size; ++i) {
            std::vector<MassRange> expected;
            MassRange previous{-1, -1};
            for (const auto &window : windows) {
                auto first = std::lower_bound(precursors.begin(), precursors.end(), masses[i] + window.lower,
                    [](const auto &entry, double mass) { return std::get<0>(entry) < mass; });
                auto last = std::upper_bound(precursors.begin(), precursors.end(), masses[i] + window.upper,
                    [](double mass, const auto &entry) { return mass < std::get<0>(entry); });
                if (first == last) continue;
                MassRange range{int(first - precursors.begin()), int(last - precursors.begin()) - 1};
                if (previous.first < 0) previous = range;
                else if (previous.last > range.first) previous.last = range.last;
                else { expected.push_back(previous); previous = range; }
            }
            if (previous.first >= 0) expected.push_back(previous);
            require(expected.size() == size_t(hostRangeCounts[i]), "GPU mass range count mismatch");
            for (size_t w = 0; w < expected.size(); ++w) {
                const auto got = actual[size_t(i) * windowCount + w];
                require(got.first == expected[w].first && got.last == expected[w].last,
                        "GPU mass range endpoints mismatch");
            }
        }
    }
    std::cout << "[CUDA assignment] generated=" << size << " assigned=" << matchedPeptides
              << " associations=" << associations << " seconds=" << now() - start
              << " backend=" << matchBackend << " verified=" << verification << '\n';
    assignedBatch = std::move(batch);
}

namespace {
// Immutable spectrum data is prepared once per preprocessed dataset. Only
// candidate ranges and top lists change between scoring batches.
struct ScoringSpectra {
    std::vector<MS2Scan*> owners;
    std::vector<Scan> scans;
    std::vector<double> peaks, lnTable;
    std::vector<int> classes;
    std::vector<short> buckets;
    std::unique_ptr<Buffer<double>> devicePeaks, deviceTable;
    std::unique_ptr<Buffer<int>> deviceClasses;
    std::unique_ptr<Buffer<short>> deviceBuckets;
    int classCount = 0;
    bool withBuckets = false;

    void releaseDevice() {
        MVH_PROFILE_SCOPE("mvh/function/releaseDevice");
        deviceTable.reset();
        deviceBuckets.reset();
        deviceClasses.reset();
        devicePeaks.reset();
    }
    void uploadOnce() {
        MVH_PROFILE_SCOPE("mvh/function/uploadOnce");
        if (deviceTable) return;
        devicePeaks = std::make_unique<Buffer<double>>(peaks);
        deviceClasses = std::make_unique<Buffer<int>>(classes);
        deviceBuckets = std::make_unique<Buffer<short>>(buckets);
        deviceTable = std::make_unique<Buffer<double>>(lnTable);
    }
};
// Default mode retains packed host arrays but releases device copies after
// each scoring batch. Optional device mode also reuses those device copies.
class SpectrumDeviceLease {
public:
    explicit SpectrumDeviceLease(ScoringSpectra& spectra) : spectra_(spectra) {
        spectra_.uploadOnce();
    }
    ~SpectrumDeviceLease() {
        if (!keepSpectraOnDevice) spectra_.releaseDevice();
    }
    SpectrumDeviceLease(const SpectrumDeviceLease&) = delete;
    SpectrumDeviceLease& operator=(const SpectrumDeviceLease&) = delete;
private:
    ScoringSpectra& spectra_;
};
std::unique_ptr<ScoringSpectra> scoringSpectra;

void resetScoringSpectra() {
    MVH_PROFILE_SCOPE("mvh/function/resetScoringSpectra");
    scoringSpectra.reset();
}

ScoringSpectra& prepareScoringSpectra(const std::vector<MS2Scan*>& scans, const Config& config) {
    MVH_PROFILE_SCOPE("mvh/pack/prepare_or_reuse_spectra");
    if (scoringSpectra) {
        require(scoringSpectra->owners == scans, "scan dataset changed without preprocessing/reset");
        require(scoringSpectra->classCount == config.classes &&
                scoringSpectra->withBuckets == needsDeviceBuckets(),
                "spectrum packing configuration changed without preprocessing/reset");
        return *scoringSpectra;
    }
    auto next = std::make_unique<ScoringSpectra>();
    auto& data = *next;
    data.owners = scans;
    data.classCount = config.classes;
    data.withBuckets = needsDeviceBuckets();
    const bool uploadBuckets = needsDeviceBuckets();
    size_t peakCount = 0, bucketCount = 0;
    for (size_t index = 0; index < scans.size(); ++index) {
        const auto *scan = scans[index];
        if (scan->pPeakList) {
            peakCount += scan->pPeakList->pPeaks.size();
            if (uploadBuckets) bucketCount += scan->pPeakList->pMassHub.size();
        } else {
            require(!uploadBuckets, "bucket matcher requires an indexed spectrum");
            peakCount += unindexedSpectrum(index, scan).masses.size();
        }
    }
    data.scans.reserve(scans.size());
    data.peaks.reserve(peakCount); data.classes.reserve(peakCount);
    data.buckets.reserve(bucketCount);
    int maxBins = 0;
    for (size_t scanIndex = 0; scanIndex < scans.size(); ++scanIndex) {
        const auto *scan = scans[scanIndex];
        Scan packed{};
        packed.peakOffset = data.peaks.size(); packed.hubOffset = data.buckets.size();
        packed.skip = scan->bSkip;
        if (scan->pPeakList) {
            // CUDA and CPU-verified runs retain the original peak vectors.
            const auto &indexed = *scan->pPeakList;
            packed.peaks = indexed.pPeaks.size();
            if (packed.peaks) {
                packed.lowest = indexed.iLowestMass;
                packed.highest = indexed.iHighestMass;
            }
        } else {
            packed.peaks = unindexedSpectrum(scanIndex, scan).masses.size();
        }
        if (!packed.skip) {
            packed.lower = scan->mzLowerBound; packed.upper = scan->mzUpperBound;
            packed.totalBins = scan->totalPeakBins;
            maxBins = std::max(maxBins, packed.totalBins);
            require(scan->intenClassCounts->size() == size_t(config.classes + 1), "invalid class count array");
            std::copy(scan->intenClassCounts->begin(), scan->intenClassCounts->end(), packed.counts);
        }
        // Keep CUDA's packing order: metadata, masses, classes, then buckets.
        if (scan->pPeakList) {
            const auto &indexed = *scan->pPeakList;
            append(data.peaks, indexed.pPeaks);
            for (char intensityClass : indexed.pClasses) data.classes.push_back(intensityClass);
            if (uploadBuckets) append(data.buckets, indexed.pMassHub);
        } else {
            // Pure RT consumes compact peak arrays without any mass hub.
            const auto &unindexed = unindexedSpectrum(scanIndex, scan);
            append(data.peaks, unindexed.masses);
            for (char intensityClass : unindexed.classes) data.classes.push_back(intensityClass);
        }
        data.scans.push_back(packed);
    }
    data.lnTable.resize(maxBins + 1);
    for (int i = 0; i <= maxBins; ++i) data.lnTable[i] = (*MVH::lnTable)[i];
    scoringSpectra = std::move(next);
    return *scoringSpectra;
}

// Host work scales with peptides and scans, not with millions of associations.
struct PackedScoringBatch {
    std::vector<Scan> scans;
    std::vector<PeptideInput> peptides;
    std::vector<Peptide *> peptideObjects;
    std::unique_ptr<PreparedPeptides> prepared;
    std::vector<Top> initialTop;
    ScoringSpectra* spectra = nullptr; // Owned by the dataset, reset before preprocessing.
    std::unique_ptr<BatchSequenceIds> sequenceIds;
    std::unique_ptr<AssignedBatch> assignment;

    int sequenceId(const std::string &sequence) {
        return sequenceIds->get(sequence);
    }
};

PackedScoringBatch packScoringBatch(const std::vector<MS2Scan *> &scans,
                                   const std::vector<Peptide *> &peptides,
                                   const Config &config) {
    MVH_PROFILE_SCOPE("mvh/pack/all");
    MVH_PROFILE_BEGIN(setupRange, "mvh/pack/setup");
    PackedScoringBatch batch;
    batch.peptideObjects = peptides;
    batch.prepared = std::move(preparedPeptides);
    require(batch.prepared && batch.prepared->offsets.size() == peptides.size(),
            "peptide preprocessing must precede scoring");
    batch.sequenceIds = std::make_unique<BatchSequenceIds>(peptides.size());
    batch.peptides.reserve(peptides.size());
    batch.assignment = std::move(assignedBatch);
    // The synthetic contract test supplies associations directly. Production
    // searches always arrive with GPU-built associations from the parent call.
    if (!batch.assignment) {
        batch.assignment = std::make_unique<AssignedBatch>();
        std::unordered_map<Peptide *, int> ids;
        for (size_t i = 0; i < peptides.size(); ++i) ids.emplace(peptides[i], i);
        std::vector<Candidate> candidates;
        for (size_t s = 0; s < scans.size(); ++s)
            for (const auto &entry : scans[s]->vMassChargePeptidePtrTuples) {
                const int precursor = batch.assignment->precursors.size();
                const int charge = std::get<1>(entry);
                batch.assignment->precursors.push_back({std::get<0>(entry), int(s), charge});
                batch.assignment->maxCharge = std::max(batch.assignment->maxCharge, charge);
                candidates.push_back({ids.at(std::get<2>(entry)), precursor, int(s), charge});
            }
        batch.assignment->candidates = std::make_unique<Buffer<Candidate>>(candidates);
    }
    MVH_PROFILE_END(setupRange);
    MVH_PROFILE_BEGIN(sequenceRange, "mvh/pack/sequence_ids");
    for (size_t i = 0; i < peptides.size(); ++i)
        batch.peptides.push_back({batch.prepared->offsets[i], batch.sequenceId(peptides[i]->sPeptide)});
    MVH_PROFILE_END(sequenceRange);
    MVH_PROFILE_BEGIN(spectraRange, "mvh/pack/spectra_and_top");
    batch.spectra = &prepareScoringSpectra(scans, config);
    batch.scans = batch.spectra->scans;
    batch.initialTop.resize(scans.size() * TopN);
    for (size_t scanIndex = 0; scanIndex < scans.size(); ++scanIndex) {
        auto& packed = batch.scans[scanIndex];
        const auto& top = scans[scanIndex]->vpWeightSumTopPeptides;
        packed.topCount = top.size();
        require(packed.topCount <= TopN, "top list larger than original limit");
        for (int rank = 0; rank < packed.topCount; ++rank) {
            const auto* peptide = top[rank];
            batch.initialTop[scanIndex * TopN + rank] = {
                peptide->dScore, batch.sequenceId(peptide->sIdentifiedPeptide)};
        }
    }
    MVH_PROFILE_END(spectraRange);
    return batch;
}

struct ScoringOutput {
    ScoreImpactBatch scoreImpact;
    std::vector<ScoringEvent> events;
    std::vector<Top> finalTop;
    std::vector<ScanCounts> counts;
    double allocationAndUploadSeconds=0, theorySeconds=0, kernelSeconds=0;
    double retentionSeconds=0, compactionSeconds=0, downloadSeconds=0;
    uint64_t cachedIonCount=0;
    int chargeStride=0;
};

#ifdef MVH_CUDA_ENABLE_RT
__global__ void compareRtResults(const Result *a,const Result *b,int n,unsigned long long *counts) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    bool matched=a[i].matched!=b[i].matched,score=a[i].score!=b[i].score;
    bool predicted=a[i].predicted!=b[i].predicted;
    if(matched||score||predicted||a[i].status!=b[i].status)atomicAdd(counts,1ULL);
    if(matched)atomicAdd(counts+1,1ULL);
    if(score)atomicAdd(counts+2,1ULL);
    if(predicted)atomicAdd(counts+3,1ULL);
}
#endif

ScoringOutput executeScoringBatch(const PackedScoringBatch &batch, const Config &config) {
    MVH_PROFILE_SCOPE("mvh/gpu/service");
    // Declared before buffers so the cleanup range ends after their destructors.
    MVH_PROFILE_DEFER(deviceCleanupRange, "mvh/gpu/release_temporaries");
    MVH_PROFILE_BEGIN(uploadRange, "mvh/gpu/allocate_upload");
    ScoringOutput output;
    const double uploadStart = now();
    const auto &candidates = *batch.assignment->candidates;
    const int size = candidates.n, scanCount = batch.scans.size();
    Buffer<Scan> scans(batch.scans);
    Buffer<PeptideInput> peptides(batch.peptides);
    const auto &texts = *batch.prepared->texts;
    SpectrumDeviceLease spectrumLease(*batch.spectra);
    const auto& peaks = *batch.spectra->devicePeaks;
    const auto& classes = *batch.spectra->deviceClasses;
    const auto& buckets = *batch.spectra->deviceBuckets;
    const auto& table = *batch.spectra->deviceTable;
    Buffer<Top> initialTop(batch.initialTop), finalTop(batch.initialTop.size());
    Buffer<Result> results(size);
    Buffer<ScanCounts> counts(scanCount);
    if (size) setCandidateRanges<<<(size + 127) / 128, 128>>>(scans.p, candidates.p, size);
    synced();
    output.allocationAndUploadSeconds = now() - uploadStart;
    MVH_PROFILE_END(uploadRange);
#ifdef MVH_CUDA_ENABLE_RT
    if (matchBackend!="cuda") {
        MVH_PROFILE_SCOPE("mvh/rt/prepare_or_reuse");
        using mvh_rt_gpu::GeometryKind;
        const auto geometry =
            matchBackend == "rt-custom" ? GeometryKind::Spheres :
            matchBackend == "rt-instanced" ? GeometryKind::InstancedTriangles :
                                            GeometryKind::Triangles;

        mvh_rt_gpu::prepare(
            batch.scans, scans.p, peaks.p, classes.p,
            peaks.n, geometry, config);
            }
#endif

    MVH_PROFILE_BEGIN(theoryRange, "mvh/gpu/theory_cache");
    const double theoryStart = now();
    std::unique_ptr<Buffer<int>> active;
    std::unique_ptr<Buffer<uint64_t>> ionOffsets;
    std::unique_ptr<Buffer<double>> cachedIons;
    // Rare high-charge hypotheses use direct GPU generation. They must not
    // multiply a dense cache for every other peptide in the batch.
    const uint64_t stride = std::min(uint64_t(batch.assignment->maxCharge) + 1, uint64_t(9));
    const uint64_t keys = batch.peptides.size() * stride;
    size_t freeBytes=0, totalBytes=0;
    check(cudaMemGetInfo(&freeBytes, &totalBytes));
    // Dense IDs make cache lookups constant-time. For unusually high charges
    // or limited VRAM, score directly on the GPU instead of forcing a huge cache.
    if (useIonCache && size && keys < 32000000 && keys * 20 < freeBytes / 2) {
        active = std::make_unique<Buffer<int>>(keys);
        ionOffsets = std::make_unique<Buffer<uint64_t>>(keys + 1);
        Buffer<uint64_t> ionCounts(keys + 1);
        check(cudaMemset(active->p, 0, keys * sizeof(int)));
        check(cudaMemset(ionCounts.p, 0, (keys + 1) * sizeof(uint64_t)));
        markTheoreticalKeys<<<(size + 127) / 128, 128>>>(candidates.p, size, stride, active->p);
        countTheoreticalIons<<<(keys + 127) / 128, 128>>>(peptides.p, texts.p, keys,
            stride, active->p, ionCounts.p, config);
        check(cudaGetLastError());
        const auto ionCount = exclusiveOffsets(ionCounts, *ionOffsets);
        check(cudaMemGetInfo(&freeBytes, &totalBytes));
        if (ionCount <= freeBytes / (2 * sizeof(double))) {
            cachedIons = std::make_unique<Buffer<double>>(ionCount);
            generateTheoreticalIons<<<(keys + 127) / 128, 128>>>(peptides.p, texts.p,
                keys, stride, active->p, ionOffsets->p, cachedIons->p, config);
            synced();
            output.cachedIonCount = ionCount;
            output.chargeStride = stride;
        }
    }
    if (!output.chargeStride) { active.reset(); ionOffsets.reset(); }
    output.theorySeconds = now() - theoryStart;
    MVH_PROFILE_END(theoryRange);

    // Diagnostics allocate their reference only after theory-cache decisions.
    // They never replace the RT results used by the actual top-candidate filter.
    std::unique_ptr<Buffer<Result>> scoreReference;
    if (scoreImpactEnabled) scoreReference = std::make_unique<Buffer<Result>>(size);
    Result *bucketResults = scoreReference ? scoreReference->p : results.p;

    MVH_PROFILE_BEGIN(scoringRange, "mvh/gpu/match_and_score");
    CudaEvent kernelStart, kernelEnd;
    check(cudaEventRecord(kernelStart.event));
    if (size && (matchBackend=="cuda" || matchBackend=="rt-audit" || scoreImpactEnabled)) ScoreSequenceVsSpectrum<<<(size + 127) / 128, 128>>>(scans.p,
        candidates.p, size, peptides.p, texts.p, peaks.p, classes.p, buckets.p,
        table.p, bucketResults, config, ionOffsets ? ionOffsets->p : nullptr,
        active ? active->p : nullptr, cachedIons ? cachedIons->p : nullptr,
        output.chargeStride);
    check(cudaGetLastError());
#ifdef MVH_CUDA_ENABLE_RT
    if (size && matchBackend!="cuda") {
        std::unique_ptr<Buffer<Result>> auditResults;
        if (matchBackend=="rt-audit") auditResults=std::make_unique<Buffer<Result>>(size);
        mvh_rt_gpu::Params p{};
        p.scans=scans.p; p.candidates=candidates.p; p.size=size;
        p.peptides=peptides.p; p.texts=texts.p; p.peaks=peaks.p;
        p.classes=classes.p; p.hub=buckets.p; p.lnTable=table.p;
        p.results=auditResults ? auditResults->p : results.p; p.cfg=config;
        p.ionOffsets=ionOffsets ? ionOffsets->p : nullptr;
        p.ionValid=active ? active->p : nullptr;
        p.cachedIons=cachedIons ? cachedIons->p : nullptr;
        p.chargeStride=output.chargeStride;
        mvh_rt_gpu::launch(p);
        if (auditResults) {
            MVH_PROFILE_SCOPE("mvh/diagnostic/rt_audit");
            Buffer<unsigned long long> differences(4);
            check(cudaMemset(differences.p,0,4*sizeof(unsigned long long)));
            compareRtResults<<<(size+127)/128,128>>>(results.p,auditResults->p,size,differences.p);
            synced();
            std::vector<unsigned long long> stats; differences.read(stats);
            std::cout << "[RT GPU audit] candidates=" << size << " differing_results=" << stats[0]
                      << " matched_count_changes=" << stats[1] << " score_changes=" << stats[2]
                      << " predicted_count_changes=" << stats[3] << '\n';
        }
    }
#endif
    check(cudaEventRecord(kernelEnd.event));
    MVH_PROFILE_BEGIN(scoreWaitRange, "mvh/gpu/scoring_completion_wait");
    check(cudaEventSynchronize(kernelEnd.event));
    MVH_PROFILE_END(scoreWaitRange);
    float milliseconds=0;
    check(cudaEventElapsedTime(&milliseconds, kernelStart.event, kernelEnd.event));
    output.kernelSeconds = milliseconds / 1000.0;
    MVH_PROFILE_END(scoringRange);

    if (scoreReference) {
        output.scoreImpact = collectScoreImpact(candidates.p, size, scoreReference->p, results.p);
        scoreReference.reset();
    }

    MVH_PROFILE_BEGIN(retentionRange, "mvh/gpu/retain_top");
    const double retentionStart = now();
    scorePeptidesMVH<<<(scanCount + 127) / 128, 128>>>(scans.p, scanCount,
        candidates.p, peptides.p, initialTop.p, finalTop.p, results.p, counts.p);
    synced();
    output.retentionSeconds = now() - retentionStart;
    MVH_PROFILE_END(retentionRange);

    MVH_PROFILE_BEGIN(compactionRange, "mvh/gpu/compact_results");
    const double compactStart = now();
    Buffer<int> selected(size), selectedCount(1);
    int eventCount = size;
    if (size && !verification) {
        cub::CountingInputIterator<int> indices(0);
        size_t bytes=0;
        check(cub::DeviceSelect::If(nullptr, bytes, indices, selected.p,
            selectedCount.p, size, KeepScoringEvent{results.p}));
        Buffer<unsigned char> scratch(bytes);
        check(cub::DeviceSelect::If(scratch.p, bytes, indices, selected.p,
            selectedCount.p, size, KeepScoringEvent{results.p}));
        check(cudaMemcpy(&eventCount, selectedCount.p, sizeof(int), cudaMemcpyDeviceToHost));
    }
    Buffer<ScoringEvent> events(eventCount);
    if (eventCount) {
        if (verification) {
            // CUB copies the counting iterator without a separate index kernel.
            check(cudaMemset(selectedCount.p, 0, sizeof(int)));
            cub::CountingInputIterator<int> indices(0);
            size_t bytes=0;
            check(cub::DeviceSelect::If(nullptr, bytes, indices, selected.p,
                selectedCount.p, size, KeepEveryCandidate{}));
            Buffer<unsigned char> scratch(bytes);
            check(cub::DeviceSelect::If(scratch.p, bytes, indices, selected.p,
                selectedCount.p, size, KeepEveryCandidate{}));
        }
        gatherScoringEvents<<<(eventCount + 127) / 128, 128>>>(selected.p, eventCount,
            candidates.p, results.p, events.p);
        synced();
    }
    output.compactionSeconds = now() - compactStart;
    MVH_PROFILE_END(compactionRange);
    MVH_PROFILE_BEGIN(downloadRange, "mvh/gpu/download_results");
    const double downloadStart = now();
    events.read(output.events); finalTop.read(output.finalTop); counts.read(output.counts);
    output.downloadSeconds = now() - downloadStart;
    MVH_PROFILE_END(downloadRange);
    MVH_PROFILE_RESUME(deviceCleanupRange);
    return output;
}

struct ScoringCounts {
    unsigned long long calls=0, successes=0, predicted=0, matched=0;
    unsigned long long restoredMerges=0, restoredScores=0;
};

ScoringCounts restoreScoringResults(std::vector<MS2Scan *> &scans,
                                   PackedScoringBatch &batch,
                                   const ScoringOutput &output) {
    ScoringCounts counts;
    for (const auto &stats : output.counts) {
        require(!stats.error, "GPU sequence generation failed");
        counts.calls += stats.calls; counts.successes += stats.successes;
        counts.predicted += stats.predicted; counts.matched += stats.matched;
    }
    std::vector<double> ions, forward, reverse;
    std::vector<char> sequence;
    for (const auto &event : output.events) {
        const auto &candidate = event.candidate;
        const auto &result = event.result;
        auto *scan = scans[candidate.scanId];
        if (scan->bSkip) continue;
        auto *peptide = batch.peptideObjects[candidate.peptideId];
        const auto &precursor = batch.assignment->precursors[candidate.precursorId];
        const auto entry = std::make_tuple(precursor.mass, candidate.charge, peptide);
        if (verification || result.status == ResultMerged) {
            const bool merged = scan->mergePeptide(scan->vpWeightSumTopPeptides,
                peptide->getPeptideSeq(), peptide->getProteinName());
            require(merged == (result.status == ResultMerged), "CUDA/CPU merge mismatch");
        }
        if (result.status == ResultMerged) { ++counts.restoredMerges; continue; }
        if (verification) {
            double score=0;
            const bool ok = MVH::ScoreSequenceVsSpectrum(peptide->sNeutralLossPeptide,
                candidate.charge, scan, &ions, &forward, &reverse, score, &sequence);
            const bool gpuOk = result.status == ResultScored || result.status == ResultAccepted;
            require(ok == gpuOk && (!ok || score == result.score), "CUDA/CPU score mismatch");
        }
        if (result.status == ResultAccepted || (verification && result.status == ResultScored)) {
            ++counts.restoredScores;
            scan->saveScore(result.score, entry, scan->vpWeightSumTopPeptides, "MVH", 2);
        }
    }
    for (size_t s = 0; s < scans.size(); ++s) {
        const auto &top = scans[s]->vpWeightSumTopPeptides;
        require(top.size() == size_t(output.counts[s].topCount), "GPU/CPU top count mismatch");
        for (size_t rank = 0; rank < top.size(); ++rank) {
            const auto &gpu = output.finalTop[s * TopN + rank];
            require(batch.sequenceId(top[rank]->sIdentifiedPeptide) == gpu.sequenceId &&
                    top[rank]->dScore == gpu.score, "GPU/CPU retained top order mismatch");
        }
        scans[s]->vMassChargePeptidePtrTuples.clear();
    }
    return counts;
}
} // namespace

void scorePeptidesMVH(std::vector<MS2Scan *> &scans, const std::vector<Peptide *> &peptides) {
    MVH_PROFILE_SCOPE("mvh/function/scorePeptidesMVH");
    MVH_PROFILE_SCOPE("mvh/batch/score_and_restore");
    MVH_PROFILE_DEFER(batchCleanupRange, "mvh/host/release_packed_batch");
    initialize();
    const double start = now();
    const Config config = configuration();
    auto batch = packScoringBatch(scans, peptides, config);
    const double packed = now();
    const auto output = executeScoringBatch(batch, config);
    const double executed = now();
    if (scoreImpactEnabled) {
        require(bool(scoreImpactWriter), "score-impact output was not initialized");
        scoreImpactWriter->writeBatch(output.scoreImpact);
        for (const auto &difference : output.scoreImpact.differences) {
            const auto &candidate = difference.candidate;
            const auto *scan = scans.at(candidate.scanId);
            const auto *peptide = batch.peptideObjects.at(candidate.peptideId);
            const auto &precursor = batch.assignment->precursors.at(candidate.precursorId);
            scoreImpactWriter->writeDifference(difference, scan->iScanId, precursor.mass,
                                               peptide->sPeptide);
        }
    }
    MVH_PROFILE_BEGIN(restoreRange, "mvh/host/restore_results");
    const auto counts = restoreScoringResults(scans, batch, output);
    MVH_PROFILE_END(restoreRange);
    const double restored = now();
    std::cout << std::setprecision(12)
              << "[CUDA scoring] candidates=" << batch.assignment->candidates->n
              << " calls=" << counts.calls << " success=" << counts.successes
              << " inrange=" << counts.predicted << " matched=" << counts.matched
              << " pack_seconds=" << packed - start
              << " gpu_service_seconds=" << executed - packed
              << " allocation_upload_seconds=" << output.allocationAndUploadSeconds
              << " theory_seconds=" << output.theorySeconds
              << " kernel_seconds=" << output.kernelSeconds
              << " retention_seconds=" << output.retentionSeconds
              << " compaction_seconds=" << output.compactionSeconds
              << " download_seconds=" << output.downloadSeconds
              << " replay_verify_seconds=" << restored - executed
              << " cached_ions=" << output.cachedIonCount
              << " cache_charge_stride=" << output.chargeStride
              << " downloaded_events=" << output.events.size()
              << " restored_merges=" << counts.restoredMerges
              << " restored_scores=" << counts.restoredScores
              << " device_bucket_entries=" << batch.spectra->buckets.size()
              << " spectrum_device_bytes=" << (batch.spectra->peaks.size() * sizeof(double)
                  + batch.spectra->classes.size() * sizeof(int)
                  + batch.spectra->buckets.size() * sizeof(short)
                  + batch.spectra->lnTable.size() * sizeof(double))
              << " spectrum_cache=" << (keepSpectraOnDevice ? "device" : "host")
              << " backend=" << matchBackend << " verified=" << verification << '\n';
    MVH_PROFILE_RESUME(batchCleanupRange);
}

}

namespace mvh_cuda {
void runContractTests(){
    setVerification(true);initialize();
    {
        std::map<double,char> data{{99.5,1},{100.0,2},{100.5,3},{101.0,1},{102.0,2}};
        PeakList list(&data);Scan s{};s.peaks=list.size();s.lowest=list.iLowestMass;s.highest=list.iHighestMass;
        std::vector<double> q{100.25,100.25,99.0,99.5,100.01,100.01,102.25,100.0,500};
        std::vector<double> t{0.25,0.25000000001,0.5,0.01,0.01,0.009999999,0.25,0.0,0.01};
        std::vector<int> classes;for(char c:list.pClasses)classes.push_back(c);
        Buffer<double>dp(list.pPeaks),dq(q),dt(t);Buffer<int>dc(classes),result(q.size());Buffer<short>dh(list.pMassHub);
        matchContract<<<1,128>>>(s,dp.p,dc.p,dh.p,dq.p,dt.p,q.size(),result.p);synced();std::vector<int>got;result.read(got);
        for(size_t i=0;i<q.size();++i){char c=list.findNear(q[i],t[i]);int expected=c==list.end()?0:c;require(got[i]==expected,"findNear boundary/tie contract");}
        require(got[0]==0&&got[1]==3,"strict tolerance and highest-class priority");
    }
    {
        struct PeakSelectionCase {
            const char* name;
            std::map<double, char> peaks;
            double query, tolerance;
            int expectedClass;
        };
        // Exactly representable distances keep these boundary cases unambiguous.
        const std::vector<PeakSelectionCase> cases{
            {"higher class beats exact mass", {{100.0,1},{100.0625,3}}, 100.0,0.125,3},
            {"outside higher class cannot mask a hit", {{100.0,1},{100.25,3}}, 100.0,0.125,1},
            {"zero class cannot mask a positive hit", {{100.0,0},{100.0625,1}}, 100.0,0.125,1},
            {"zero class alone is unscored", {{100.0,0}}, 100.0,0.125,0},
            {"strict tolerance boundary", {{100.125,3}}, 100.0,0.125,0},
            {"inside tolerance", {{100.125,3}}, 100.0,0.25,3},
            {"class priority across buckets", {{99.9375,3},{100.0,1}}, 100.0,0.125,3},
            {"same class candidates", {{99.9375,2},{100.03125,2}}, 100.0,0.125,2},
            {"no in-range peak", {{100.25,3}}, 100.0,0.125,0},
            {"empty spectrum", {}, 100.0,0.125,0},
            {"zero tolerance", {{100.0,3}}, 100.0,0.0,0},
        };
        for (const auto& test : cases) {
            auto peakData = test.peaks;
            PeakList reference(&peakData);
            Scan scan{};
            scan.peaks = reference.size();
            if (scan.peaks) {
                scan.lowest = reference.iLowestMass;
                scan.highest = reference.iHighestMass;
            }
            // Exercise offsets as well as selection; prefix entries must be ignored.
            scan.peakOffset = 1;
            scan.hubOffset = 2;
            std::vector<double> masses{0.0};
            masses.insert(masses.end(), reference.pPeaks.begin(), reference.pPeaks.end());
            std::vector<int> classes{3};
            for (char cls : reference.pClasses) classes.push_back(cls);
            std::vector<short> buckets{-1, -1};
            buckets.insert(buckets.end(), reference.pMassHub.begin(), reference.pMassHub.end());
            Buffer<double> deviceMasses(masses), query(std::vector<double>{test.query}),
                           tolerance(std::vector<double>{test.tolerance});
            Buffer<int> deviceClasses(classes), result(1);
            Buffer<short> deviceBuckets(buckets);
            matchContract<<<1,1>>>(scan, deviceMasses.p, deviceClasses.p, deviceBuckets.p,
                                  query.p, tolerance.p, 1, result.p);
            synced();
            std::vector<int> actual;
            result.read(actual);
            const char cpuClass = reference.findNear(test.query, test.tolerance);
            const int expectedCpu = cpuClass == reference.end() ? 0 : cpuClass;
            require(expectedCpu == test.expectedClass, std::string("CPU peak selection: ") + test.name);
            require(actual[0] == test.expectedClass, std::string("CUDA peak selection: ") + test.name);
        }
    }
    {
        // Inclusive endpoints, duplicated boundary entries, no match, and
        // stable scan grouping are all observable assignment semantics.
        std::vector<double> masses{100, 200, 0};
        std::vector<Precursor> precursors{{99.875,1,2},{100,0,3},{100.125,1,2},
                                         {100.25,0,2},{200,0,2}};
        std::vector<MassWindow> windows{{-0.125,0.125},{0.125,0.25}};
        Buffer<double> dm(masses); Buffer<Precursor> dp(precursors);
        Buffer<MassWindow> dw(windows); Buffer<MassRange> ranges(6);
        Buffer<int> rangesPerPeptide(3); Buffer<uint64_t> counts(4), offsets(4);
        check(cudaMemset(counts.p,0,4*sizeof(uint64_t)));
        GetAllRangeFromMass<<<1,128>>>(dm.p,3,dp.p,5,dw.p,2,ranges.p,
                                       rangesPerPeptide.p,counts.p);
        require(exclusiveOffsets(counts,offsets)==6,"assignment duplicate boundary lost");
        std::vector<int> rc; rangesPerPeptide.read(rc);
        require(rc==std::vector<int>({2,1,0}),"assignment inclusive endpoints/no-match");
        Buffer<Candidate> candidates(6), sorted(6); Buffer<int> keys(6), sortedKeys(6);
        assignPeptides2Scans<<<1,128>>>(ranges.p,rangesPerPeptide.p,offsets.p,
                                       3,2,dp.p,candidates.p,keys.p);
        size_t bytes=0;
        check(cub::DeviceRadixSort::SortPairs(nullptr,bytes,keys.p,sortedKeys.p,
                                             candidates.p,sorted.p,6));
        Buffer<unsigned char> scratch(bytes);
        check(cub::DeviceRadixSort::SortPairs(scratch.p,bytes,keys.p,sortedKeys.p,
                                             candidates.p,sorted.p,6));
        std::vector<Candidate> got; sorted.read(got);
        const int expectedPeptide[]={0,0,1,0,0,0};
        const int expectedPrecursor[]={1,3,4,0,2,2};
        for(int i=0;i<6;++i)
            require(got[i].peptideId==expectedPeptide[i] &&
                    got[i].precursorId==expectedPrecursor[i],"assignment stable order");
    }
    ProNovoConfig::minObservedMz=0;ProNovoConfig::maxObservedMz=10000;
    const std::vector<std::string> seqs{"[LDNM~ATK]","[LDN!M~ATK]","[FDITEEGLRYLRQWVNESGIR]","[~LDNMATK]","[LDNMATK]~","[AAAAAAAK]","["+std::string(127,'A')+"K]"};
    for(bool smart:{false,true}){
        useIonCache=smart; // Exercise both the cached and direct device scoring paths.
        MVH::bUseSmartPlusThreeModel=smart;
        std::vector<Peptide *> peptides;std::vector<MS2Scan *> scans;
        for(int z=1;z<=9;++z)for(const auto &str:seqs){
            auto *p=new Peptide;p->setPeptide(str,str,"contract",0,5000,'-','-','-','-');peptides.push_back(p);
            auto *s=new MS2Scan;s->iScanId=scans.size();s->iParentChargeState=z;s->dParentMZ=2000;s->dParentNeutralMass=20000;
            std::string text=str;std::vector<double>ions,forward,reverse;std::vector<char>sequence;
            require(MVH::CalculateSequenceIons(text,z,smart,&ions,&forward,&reverse,&sequence),"CPU contract ions");
            for(size_t k=0;k<ions.size();++k){s->vdMZ.push_back(ions[k]+(k%2?0.001:-0.001));s->vdIntensity.push_back(100.0+(k%5));s->viCharge.push_back(1);}
            // Duplicate m/z and equal intensity exercise map replacement and stable order.
            s->vdMZ.push_back(s->vdMZ.front());s->vdIntensity.push_back(100);s->viCharge.push_back(2);
            s->vMassChargePeptidePtrTuples.push_back({5000,z,p});
            s->vMassChargePeptidePtrTuples.push_back({5000,z,p});
            scans.push_back(s);
        }
        preProcessAllMs2Mvh(scans);preprocessingMVH(peptides);scorePeptidesMVH(scans,peptides);
        for(auto *s:scans)delete s;for(auto*p:peptides)delete p;MVH::destroyLnTable();
    }
    useIonCache=true;
    std::cout<<"PASS: CUDA strict/tie matching, charges 1..9, mass assignment boundaries/order, both ion models, PTM termini, long sequences, duplicates\n";
}
}
