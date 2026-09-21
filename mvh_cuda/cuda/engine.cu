#include "engine.h"
#include "ms2scanvector.h"
#include "preprocess.cuh"
#include "scoring.cuh"
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
bool deviceReady=false;
double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error("CUDA MVH: " + message);
}
// Avoid allocating a temporary std::string for successful checks in hot loops.
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(std::string("CUDA MVH: ") + message);
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

void preProcessAllMs2Mvh(std::vector<MS2Scan *> &scans){
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
        delete s->peakData;s->peakData=nullptr;delete s->intenClassCounts;delete s->pPeakList;
        s->intenClassCounts=new std::vector<int>();s->pPeakList=new PeakList(&peakMap);s->bSkip=r.skip;
        if(in.count>=cfg.minClassCount){s->mzLowerBound=cfg.mzLow;s->mzUpperBound=cfg.mzHigh;}
        if(!r.skip){s->totalPeakBins=r.totalBins;s->intenClassCounts->assign(r.counts,r.counts+cfg.classes+1);maxBins=std::max(maxBins,r.totalBins);}
        s->dSumIntensity=r.sum;s->dMaxIntensity=r.max;
    }
    MVH::initialLnTable(maxBins);
    std::cout<<"[CUDA preprocessing] scans="<<n<<" seconds="<<now()-start<<" verified="<<verification<<'\n';
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
    int n=peptides.size();Buffer<char>dText(texts);Buffer<uint64_t>dOffsets(offsets);Buffer<int>dCap(capacities),dLengths(n),dErrors(n);Buffer<Rule>dRules(rules);
    preprocessingMVH<<<(n+127)/128,128>>>(dText.p,dOffsets.p,dCap.p,n,dRules.p,rules.size(),dLengths.p,dErrors.p);synced();
    dText.read(texts);dLengths.read(lengths);dErrors.read(errors);
    for(int i=0;i<n;++i){require(errors[i]==0,"neutral loss failed or self-repeating rule");require(lengths[i]<=MaxLength,"peptide exceeds 128 residues");
        std::string neutral(texts.data()+offsets[i]);
        if(verification){Peptide ref;ref.sPeptide=peptides[i]->sPeptide;ref.preprocessingMVH();require(ref.sNeutralLossPeptide==neutral&&ref.iPeptideLength==lengths[i],"peptide preprocessing mismatch");}
        peptides[i]->iPeptideLength=lengths[i];peptides[i]->sNeutralLossPeptide=std::move(neutral);
    }
    std::cout<<"[CUDA peptide preprocessing] peptides="<<n<<" seconds="<<now()-start<<" verified="<<verification<<'\n';
}

namespace {

// Own all host arrays for one batch. Offsets, rather than host pointers, cross
// the CUDA boundary. Candidate order within each scan must never change.
struct PackedScoringBatch {
    std::vector<Scan> scans;
    std::vector<Candidate> candidates;
    std::vector<char> texts;
    std::vector<double> peaks;
    std::vector<int> classes;
    std::vector<short> buckets;
    std::vector<Top> initialTop;
    std::vector<double> lnTable;
    std::unordered_map<std::string, int> sequenceIds;

    int sequenceId(const std::string &sequence) {
        const auto found = sequenceIds.find(sequence);
        if (found != sequenceIds.end()) return found->second;
        const int id = sequenceIds.size();
        sequenceIds.emplace(sequence, id);
        return id;
    }
};

PackedScoringBatch packScoringBatch(const std::vector<MS2Scan *> &scans,
                                   const Config &config) {
    PackedScoringBatch batch;
    size_t candidateCount = 0, peakCount = 0, bucketCount = 0;
    for (const auto *scan : scans) {
        candidateCount += scan->vMassChargePeptidePtrTuples.size();
        peakCount += scan->pPeakList->pPeaks.size();
        bucketCount += scan->pPeakList->pMassHub.size();
    }

    // A real batch contains tens of millions of associations. Reserving once
    // avoids repeated reallocations and copies of these large arrays.
    batch.scans.reserve(scans.size());
    batch.candidates.reserve(candidateCount);
    batch.peaks.reserve(peakCount);
    batch.classes.reserve(peakCount);
    batch.buckets.reserve(bucketCount);
    batch.initialTop.resize(scans.size() * TopN);
    const size_t peptideEstimate = std::min(candidateCount, size_t(PEPTIDE_ARRAY_SIZE));
    batch.sequenceIds.reserve(peptideEstimate);
    std::unordered_map<Peptide *, std::pair<uint64_t, int>> peptideInfo;
    peptideInfo.reserve(peptideEstimate);
    int maxBins = 0;

    for (size_t scanIndex = 0; scanIndex < scans.size(); ++scanIndex) {
        const auto *scan = scans[scanIndex];
        Scan packed{};
        packed.peakOffset = batch.peaks.size();
        packed.hubOffset = batch.buckets.size();
        packed.candidateOffset = batch.candidates.size();
        packed.skip = scan->bSkip;
        packed.candidates = scan->vMassChargePeptidePtrTuples.size();
        packed.peaks = scan->pPeakList->pPeaks.size();
        if (packed.peaks) {
            packed.lowest = scan->pPeakList->iLowestMass;
            packed.highest = scan->pPeakList->iHighestMass;
        }
        if (!packed.skip) {
            packed.lower = scan->mzLowerBound;
            packed.upper = scan->mzUpperBound;
            packed.totalBins = scan->totalPeakBins;
            maxBins = std::max(maxBins, packed.totalBins);
            require(scan->intenClassCounts->size() == size_t(config.classes + 1),
                    "invalid class count array");
            std::copy(scan->intenClassCounts->begin(), scan->intenClassCounts->end(),
                      packed.counts);
        }
        append(batch.peaks, scan->pPeakList->pPeaks);
        for (char intensityClass : scan->pPeakList->pClasses)
            batch.classes.push_back(intensityClass);
        append(batch.buckets, scan->pPeakList->pMassHub);

        packed.topCount = scan->vpWeightSumTopPeptides.size();
        require(packed.topCount <= TopN, "top list larger than original limit");
        for (int rank = 0; rank < packed.topCount; ++rank) {
            const auto *peptide = scan->vpWeightSumTopPeptides[rank];
            batch.initialTop[scanIndex * TopN + rank] = {
                peptide->dScore, batch.sequenceId(peptide->sIdentifiedPeptide)};
        }
        for (const auto &entry : scan->vMassChargePeptidePtrTuples) {
            auto *peptide = std::get<2>(entry);
            auto info = peptideInfo.find(peptide);
            if (info == peptideInfo.end()) {
                const auto offset = batch.texts.size();
                const auto &sequence = peptide->sNeutralLossPeptide;
                require(sequence.size() < MaxText, "sequence too long");
                batch.texts.insert(batch.texts.end(), sequence.begin(), sequence.end());
                batch.texts.push_back(0);
                info = peptideInfo.emplace(peptide, std::make_pair(
                    offset, batch.sequenceId(peptide->sPeptide))).first;
            }
            batch.candidates.push_back({info->second.first, info->second.second,
                                        std::get<1>(entry)});
        }
        batch.scans.push_back(packed);
    }
    // Transfer the original CPU-built table verbatim; do not recompute logs.
    batch.lnTable.resize(maxBins + 1);
    for (int i = 0; i <= maxBins; ++i) batch.lnTable[i] = (*MVH::lnTable)[i];
    return batch;
}

struct ScoringOutput {
    std::vector<Result> results;
    std::vector<Top> finalTop;
    double allocationAndUploadSeconds = 0;
    double kernelSeconds = 0;
    double downloadSeconds = 0;
};

ScoringOutput executeScoringBatch(const PackedScoringBatch &batch, const Config &config) {
    ScoringOutput output;
    const double uploadStart = now();
    Buffer<Scan> scans(batch.scans);
    Buffer<Candidate> candidates(batch.candidates);
    Buffer<char> texts(batch.texts);
    Buffer<double> peaks(batch.peaks), table(batch.lnTable);
    Buffer<int> classes(batch.classes);
    Buffer<short> buckets(batch.buckets);
    Buffer<Top> initialTop(batch.initialTop), finalTop(batch.initialTop.size());
    Buffer<Result> results(batch.candidates.size());
    output.allocationAndUploadSeconds = now() - uploadStart;

    // CUDA events measure only the kernel, excluding host packing and transfers.
    CudaEvent kernelStart, kernelEnd;
    check(cudaEventRecord(kernelStart.event));
    const int scanCount = batch.scans.size();
    scorePeptidesMVH<<<(scanCount + 127) / 128, 128>>>(
        scans.p, scanCount, candidates.p, texts.p, peaks.p, classes.p,
        buckets.p, table.p, initialTop.p, finalTop.p, results.p, config);
    check(cudaGetLastError());
    check(cudaEventRecord(kernelEnd.event));
    check(cudaEventSynchronize(kernelEnd.event));
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds, kernelStart.event, kernelEnd.event));
    output.kernelSeconds = milliseconds / 1000.0;

    const double downloadStart = now();
    results.read(output.results);
    finalTop.read(output.finalTop);
    output.downloadSeconds = now() - downloadStart;
    return output;
}

struct ScoringCounts {
    unsigned long long calls = 0, successes = 0, predicted = 0, matched = 0;
    unsigned long long restoredMerges = 0, restoredScores = 0;
};

ScoringCounts restoreScoringResults(std::vector<MS2Scan *> &scans,
                                   PackedScoringBatch &batch,
                                   const ScoringOutput &output) {
    ScoringCounts counts;
    std::vector<double> ions, forward, reverse;
    std::vector<char> sequence;
    for (size_t scanIndex = 0; scanIndex < scans.size(); ++scanIndex) {
        auto *scan = scans[scanIndex];
        const auto &packed = batch.scans[scanIndex];
        if (!scan->bSkip) {
            for (int candidateIndex = 0; candidateIndex < packed.candidates; ++candidateIndex) {
                const auto &entry = scan->vMassChargePeptidePtrTuples[candidateIndex];
                const auto &result = output.results[packed.candidateOffset + candidateIndex];
                require(result.status >= 0, "GPU sequence generation failed");

                // A failed, non-merged candidate cannot change the original top
                // list. Trust that GPU decision in normal mode; verification mode
                // still checks EVERY merge decision with the original CPU method.
                if (verification || result.status == ResultMerged) {
                    auto *peptide = std::get<2>(entry);
                    const bool merged = scan->mergePeptide(scan->vpWeightSumTopPeptides,
                        peptide->getPeptideSeq(), peptide->getProteinName());
                    if (merged != (result.status == ResultMerged)) {
                        throw std::runtime_error("CUDA/CPU merge mismatch at scan " +
                            std::to_string(scan->iScanId) + " candidate " +
                            std::to_string(candidateIndex));
                    }
                }
                if (result.status == ResultMerged) {
                    ++counts.restoredMerges;
                    continue;
                }
                ++counts.calls;
                counts.predicted += result.predicted;
                counts.matched += result.matched;

                if (verification) {
                    double score = 0;
                    auto *peptide = std::get<2>(entry);
                    const bool ok = MVH::ScoreSequenceVsSpectrum(
                        peptide->sNeutralLossPeptide, std::get<1>(entry), scan,
                        &ions, &forward, &reverse, score, &sequence);
                    if (ok != (result.status == ResultScored) || (ok && score != result.score)) {
                        std::cerr << std::setprecision(17) << "Mismatch scan=" << scan->iScanId
                                  << " peptide=" << peptide->sNeutralLossPeptide
                                  << " charge=" << std::get<1>(entry)
                                  << " cpu_score=" << score << " gpu_score=" << result.score << '\n';
                        throw std::runtime_error("CUDA/CPU score mismatch");
                    }
                }
                if (result.status == ResultScored) {
                    ++counts.successes;
                    ++counts.restoredScores;
                    // Replay state-changing operations in EXACT candidate order.
                    // In particular, do not reconstruct only the final top list:
                    // that would lose order-dependent protein-name merges.
                    scan->saveScore(result.score, entry, scan->vpWeightSumTopPeptides, "MVH", 2);
                }
            }
        }
        for (size_t rank = 0; rank < scan->vpWeightSumTopPeptides.size(); ++rank) {
            const auto *peptide = scan->vpWeightSumTopPeptides[rank];
            const auto &gpu = output.finalTop[scanIndex * TopN + rank];
            require(batch.sequenceId(peptide->sIdentifiedPeptide) == gpu.sequenceId &&
                    peptide->dScore == gpu.score, "GPU/CPU retained top order mismatch");
        }
        scan->vMassChargePeptidePtrTuples.clear();
    }
    return counts;
}
} // namespace

void scorePeptidesMVH(std::vector<MS2Scan *> &scans) {
    initialize();
    const double start = now();
    const Config config = configuration();
    auto batch = packScoringBatch(scans, config);
    const double packed = now();
    const auto output = executeScoringBatch(batch, config);
    const double executed = now();
    const auto counts = restoreScoringResults(scans, batch, output);
    const double restored = now();
    std::cout << std::setprecision(12)
              << "[CUDA scoring] candidates=" << batch.candidates.size()
              << " calls=" << counts.calls << " success=" << counts.successes
              << " inrange=" << counts.predicted << " matched=" << counts.matched
              << " pack_seconds=" << packed - start
              << " gpu_service_seconds=" << executed - packed
              << " allocation_upload_seconds=" << output.allocationAndUploadSeconds
              << " kernel_seconds=" << output.kernelSeconds
              << " download_seconds=" << output.downloadSeconds
              << " replay_verify_seconds=" << restored - executed
              << " restored_merges=" << counts.restoredMerges
              << " restored_scores=" << counts.restoredScores
              << " verified=" << verification << '\n';
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
    ProNovoConfig::minObservedMz=0;ProNovoConfig::maxObservedMz=10000;
    const std::vector<std::string> seqs{"[LDNM~ATK]","[LDN!M~ATK]","[FDITEEGLRYLRQWVNESGIR]","[~LDNMATK]","[LDNMATK]~","[AAAAAAAK]","["+std::string(127,'A')+"K]"};
    for(bool smart:{false,true}){
        MVH::bUseSmartPlusThreeModel=smart;
        std::vector<Peptide *> peptides;std::vector<MS2Scan *> scans;
        for(int z=1;z<=8;++z)for(const auto &str:seqs){
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
        preProcessAllMs2Mvh(scans);preprocessingMVH(peptides);scorePeptidesMVH(scans);
        for(auto *s:scans)delete s;for(auto*p:peptides)delete p;MVH::destroyLnTable();
    }
    std::cout<<"PASS: CUDA strict/tie matching, charges 1..8, both ion models, PTM termini, long sequences, duplicates\n";
}
}
