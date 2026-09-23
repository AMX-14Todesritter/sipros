#include "engine.h"
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
bool verification=false;
bool scoreImpactEnabled = false;
std::unique_ptr<ScoreImpactWriter> scoreImpactWriter;
std::string matchBackend="cuda";
int batchSize=2000000;
bool useIonCache=true;
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
    require(!enabled || matchBackend == "rt-triangle" || matchBackend == "rt-instanced",
            "--score-impact requires rt-triangle or rt-instanced");
    scoreImpactEnabled = enabled;
}
void startScoreImpact(const std::string &outputDirectory) {
    if (scoreImpactEnabled) scoreImpactWriter = std::make_unique<ScoreImpactWriter>(outputDirectory);
}

MatchBackendScope::~MatchBackendScope() {
    scoreImpactWriter.reset();
    unindexedSpectra.clear();
#ifdef MVH_CUDA_ENABLE_RT
    mvh_rt_gpu::reset();
#endif
}
const std::string &matchBackendName() { return matchBackend; }
void setMatchBackend(const std::string &name) {
    require(name=="cuda" || name=="rt-triangle" || name=="rt-audit" || name=="rt-instanced", "Unknown match backend");
#ifndef MVH_CUDA_ENABLE_RT
    require(name=="cuda", "RT backend was not enabled at build time");
#endif
    matchBackend=name;
    if (name!="cuda") std::cout << "[RT backend] " << name
        << "; experimental unshifted triangles; known zero-distance/boundary differences\n";
}
void setPeptideBatchSize(int size) { require(size > 0, "batch size must be positive"); batchSize = size; }
int peptideBatchSize() { return batchSize; }

void preProcessAllMs2Mvh(std::vector<MS2Scan *> &scans){
#ifdef MVH_CUDA_ENABLE_RT
    mvh_rt_gpu::reset();
#endif
    initialize();double start=now();Config cfg=configuration();int n=scans.size();require(n>0,"empty scan collection");
    std::vector<RawScan> raw;std::vector<double> mz,intensity;std::vector<int> charges;
    for(auto *s:scans){require(s->vdMZ.size()==s->vdIntensity.size()&&s->vdMZ.size()==s->viCharge.size(),"raw peak array lengths");
        require(s->vdMZ.size()<=std::numeric_limits<int>::max(),"scan too large");
        raw.push_back({mz.size(),int(s->vdMZ.size()),s->iParentChargeState,s->dParentNeutralMass,s->dParentMZ});
        append(mz,s->vdMZ);append(intensity,s->vdIntensity);append(charges,s->viCharge);
    }
    std::vector<PrepResult> result;std::vector<double> kept;std::vector<int> classes;
    {
        Buffer<RawScan> dRaw(raw);Buffer<double>dMz(mz),dIntensity(intensity);Buffer<int>dCharge(charges),indices(mz.size());
        Buffer<PrepResult>dResult(n);Buffer<double>dKept(size_t(n)*cfg.maxPeaks);Buffer<int>dClasses(size_t(n)*cfg.maxPeaks);
        preprocessMvh<<<(n+127)/128,128>>>(dRaw.p,n,dMz.p,dIntensity.p,dCharge.p,indices.p,dResult.p,dKept.p,dClasses.p,cfg);synced();
        sumIntensity<<<(n+127)/128,128>>>(dRaw.p,n,dIntensity.p,dResult.p);synced();
        dMz.read(mz);dIntensity.read(intensity);dCharge.read(charges);dResult.read(result);dKept.read(kept);dClasses.read(classes);
    }
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
    MVH::initialLnTable(maxBins);
    std::cout<<"[CUDA preprocessing] scans="<<n<<" seconds="<<now()-start<<" verified="<<verification
             <<" host_bucket_entries="<<hostBucketEntries<<'\n';
}

void preprocessingMVH(std::vector<Peptide *> &peptides){
    initialize();if(peptides.empty())return;double start=now();
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
    int n=peptides.size();
    auto prepared=std::make_unique<PreparedPeptides>();
    prepared->texts=std::make_unique<Buffer<char>>(texts);
    prepared->offsets=offsets;
    Buffer<uint64_t>dOffsets(offsets);Buffer<int>dCap(capacities),dLengths(n),dErrors(n);Buffer<Rule>dRules(rules);
    preprocessingMVH<<<(n+127)/128,128>>>(prepared->texts->p,dOffsets.p,dCap.p,n,dRules.p,rules.size(),dLengths.p,dErrors.p);synced();
    // Only verification needs neutral-loss strings on the host. Normal search
    // keeps the transformed text resident through theoretical-ion generation.
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
    initialize();
    const double start = now();
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
    const int size = peptides.size(), windowCount = windows.size();
    Buffer<double> deviceMasses(masses);
    Buffer<Precursor> devicePrecursors(batch->precursors);
    Buffer<MassWindow> deviceWindows(windows);
    Buffer<MassRange> ranges(size_t(size) * windowCount);
    Buffer<int> rangeCounts(size);
    Buffer<uint64_t> counts(size + 1), offsets(size + 1);
    check(cudaMemset(counts.p, 0, counts.n * sizeof(uint64_t)));
    GetAllRangeFromMass<<<(size + 127) / 128, 128>>>(deviceMasses.p, size,
        devicePrecursors.p, batch->precursors.size(), deviceWindows.p, windowCount,
        ranges.p, rangeCounts.p, counts.p);
    check(cudaGetLastError());
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
// Host work scales with peptides and scans, not with millions of associations.
struct PackedScoringBatch {
    std::vector<Scan> scans;
    std::vector<PeptideInput> peptides;
    std::vector<Peptide *> peptideObjects;
    std::unique_ptr<PreparedPeptides> prepared;
    std::vector<double> peaks;
    std::vector<int> classes;
    std::vector<short> buckets;
    std::vector<Top> initialTop;
    std::vector<double> lnTable;
    std::unordered_map<std::string, int> sequenceIds;
    std::unique_ptr<AssignedBatch> assignment;

    int sequenceId(const std::string &sequence) {
        const auto found = sequenceIds.find(sequence);
        if (found != sequenceIds.end()) return found->second;
        const int id = sequenceIds.size();
        sequenceIds.emplace(sequence, id);
        return id;
    }
};

PackedScoringBatch packScoringBatch(const std::vector<MS2Scan *> &scans,
                                   const std::vector<Peptide *> &peptides,
                                   const Config &config) {
    PackedScoringBatch batch;
    batch.peptideObjects = peptides;
    batch.prepared = std::move(preparedPeptides);
    require(batch.prepared && batch.prepared->offsets.size() == peptides.size(),
            "peptide preprocessing must precede scoring");
    batch.sequenceIds.reserve(peptides.size());
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
    for (size_t i = 0; i < peptides.size(); ++i)
        batch.peptides.push_back({batch.prepared->offsets[i], batch.sequenceId(peptides[i]->sPeptide)});
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
    batch.scans.reserve(scans.size());
    batch.peaks.reserve(peakCount); batch.classes.reserve(peakCount);
    batch.buckets.reserve(bucketCount);
    batch.initialTop.resize(scans.size() * TopN);
    int maxBins = 0;
    for (size_t scanIndex = 0; scanIndex < scans.size(); ++scanIndex) {
        const auto *scan = scans[scanIndex];
        Scan packed{};
        packed.peakOffset = batch.peaks.size(); packed.hubOffset = batch.buckets.size();
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
            append(batch.peaks, indexed.pPeaks);
            for (char intensityClass : indexed.pClasses) batch.classes.push_back(intensityClass);
            if (uploadBuckets) append(batch.buckets, indexed.pMassHub);
        } else {
            // Pure RT consumes compact peak arrays without any mass hub.
            const auto &unindexed = unindexedSpectrum(scanIndex, scan);
            append(batch.peaks, unindexed.masses);
            for (char intensityClass : unindexed.classes) batch.classes.push_back(intensityClass);
        }
        packed.topCount = scan->vpWeightSumTopPeptides.size();
        require(packed.topCount <= TopN, "top list larger than original limit");
        for (int rank = 0; rank < packed.topCount; ++rank) {
            const auto *peptide = scan->vpWeightSumTopPeptides[rank];
            batch.initialTop[scanIndex * TopN + rank] = {
                peptide->dScore, batch.sequenceId(peptide->sIdentifiedPeptide)};
        }
        batch.scans.push_back(packed);
    }
    batch.lnTable.resize(maxBins + 1);
    for (int i = 0; i <= maxBins; ++i) batch.lnTable[i] = (*MVH::lnTable)[i];
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
    ScoringOutput output;
    const double uploadStart = now();
    const auto &candidates = *batch.assignment->candidates;
    const int size = candidates.n, scanCount = batch.scans.size();
    Buffer<Scan> scans(batch.scans);
    Buffer<PeptideInput> peptides(batch.peptides);
    const auto &texts = *batch.prepared->texts;
    Buffer<double> peaks(batch.peaks), table(batch.lnTable);
    Buffer<int> classes(batch.classes);
    // Empty in pure RT (also with CPU verification): Buffer performs neither
    // cudaMalloc nor cudaMemcpy for an empty vector and exposes a null pointer.
    Buffer<short> buckets(batch.buckets);
    Buffer<Top> initialTop(batch.initialTop), finalTop(batch.initialTop.size());
    Buffer<Result> results(size);
    Buffer<ScanCounts> counts(scanCount);
    if (size) setCandidateRanges<<<(size + 127) / 128, 128>>>(scans.p, candidates.p, size);
    synced();
    output.allocationAndUploadSeconds = now() - uploadStart;
#ifdef MVH_CUDA_ENABLE_RT
    if (matchBackend!="cuda")
        mvh_rt_gpu::prepare(batch.scans,scans.p,peaks.p,peaks.n,matchBackend=="rt-instanced");
#endif

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

    // Diagnostics allocate their reference only after theory-cache decisions.
    // They never replace the RT results used by the actual top-candidate filter.
    std::unique_ptr<Buffer<Result>> scoreReference;
    if (scoreImpactEnabled) scoreReference = std::make_unique<Buffer<Result>>(size);
    Result *bucketResults = scoreReference ? scoreReference->p : results.p;

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
    check(cudaEventSynchronize(kernelEnd.event));
    float milliseconds=0;
    check(cudaEventElapsedTime(&milliseconds, kernelStart.event, kernelEnd.event));
    output.kernelSeconds = milliseconds / 1000.0;

    if (scoreReference) {
        output.scoreImpact = collectScoreImpact(candidates.p, size, scoreReference->p, results.p);
        scoreReference.reset();
    }

    const double retentionStart = now();
    scorePeptidesMVH<<<(scanCount + 127) / 128, 128>>>(scans.p, scanCount,
        candidates.p, peptides.p, initialTop.p, finalTop.p, results.p, counts.p);
    synced();
    output.retentionSeconds = now() - retentionStart;

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
    const double downloadStart = now();
    events.read(output.events); finalTop.read(output.finalTop); counts.read(output.counts);
    output.downloadSeconds = now() - downloadStart;
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
    const auto counts = restoreScoringResults(scans, batch, output);
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
              << " device_bucket_entries=" << batch.buckets.size()
              << " backend=" << matchBackend << " verified=" << verification << '\n';
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
        require(got[0]==0&&got[1]==2,"strict tolerance and first-encountered tie");
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
