#include "peptide_generation.h"
#include "generation.cuh"
#include "engine.h"
#include "proteindatabase.h"
#include <fstream>
#include <memory>
#include <iomanip>
#include <limits>

namespace mvh_cuda {
namespace {
std::string generationMode="cuda";
void ensure(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
std::vector<uint64_t> prefix(const std::vector<uint64_t>& counts) {
    std::vector<uint64_t> offsets{0};offsets.reserve(counts.size()+1);
    for(auto count:counts) {
        ensure(count<=uint64_t(std::numeric_limits<int64_t>::max())-offsets.back(),"Peptide enumeration count overflow");
        offsets.push_back(offsets.back()+count);
    }
    return offsets;
}
}
void setPeptideGeneration(const std::string& mode) {
    ensure(mode=="cpu" || mode=="cuda","--peptide-generation must be cpu or cuda");
    generationMode=mode;
}
const std::string& peptideGenerationName() { return generationMode; }
struct SearchPeptideGenerator::Impl {
    bool screen, gpu, verify;
    std::unique_ptr<ProteinDatabase> reference;
    std::ifstream fasta;
    std::string pending, legal;
    GenerationConfig config{};
    std::vector<GenerationMod> mods;
    std::unique_ptr<Buffer<GenerationConfig>> deviceConfig;
    std::unique_ptr<Buffer<GenerationMod>> deviceMods;
    std::vector<std::string> names, sequences;
    std::vector<GenerationBase> bases;
    std::vector<uint64_t> variants;
    std::unique_ptr<Buffer<char>> deviceText;
    std::unique_ptr<Buffer<GenerationProtein>> deviceProteins;
    std::unique_ptr<Buffer<GenerationBase>> deviceBases;
    std::unique_ptr<Buffer<double>> deviceMasses;
    std::unique_ptr<Buffer<uint64_t>> deviceVariants;
    std::vector<char> pageText;
    std::vector<GenerationResult> pageResults;
    uint64_t cursor=0,total=0,generated=0,blocks=0;
    size_t pageIndex=0;
    int stride=0;
    explicit Impl(bool output):screen(output),gpu(generationMode=="cuda"),verify(verificationEnabled()) {
        if(!gpu || verify)reference=std::make_unique<ProteinDatabase>(output);
    }
    void configure() {
        ensure(ProNovoConfig::getSearchType()=="Regular","CUDA generation supports Regular search only");
        config.minimum=ProNovoConfig::getMinPeptideLength();config.maximum=ProNovoConfig::getMaxPeptideLength();
        ensure(config.minimum>0 && config.maximum>=config.minimum && config.maximum<=GenerationMaxLength,
               "CUDA generation requires peptide length in 1..128; use --peptide-generation cpu for other lengths");
        config.missed=ProNovoConfig::getMaxMissedCleavages();
        ensure(config.missed>=0,"Negative missed cleavage count");
        config.maxPtm=std::max(0,std::min(ProNovoConfig::getMaxPTMcount(),GenerationSites));
        config.removeM=ProNovoConfig::getTestStartRemoval();
        for(unsigned char c:ProNovoConfig::getCleavageAfterResidues())config.after[c]=1;
        for(unsigned char c:ProNovoConfig::getCleavageBeforeResidues())config.before[c]=1;
        for(const auto& name:ProNovoConfig::vsSingleResidueNames)
            if(!name.empty() && std::isalpha(static_cast<unsigned char>(name[0])))legal+=name;
        const auto& compositions=ProNovoConfig::configIsotopologue.mResidueAtomicComposition;
        for(const auto& item:compositions)if(item.first.size()==1) {
            ensure(item.second.size()==6,"CUDA generation requires six-element residue compositions");
            for(int k=0;k<6;++k)config.atoms[(unsigned char)item.first[0]][k]=item.second[k];
        }
        for(const auto* name:{"Nterm","Cterm"}) {
            const auto& atoms=compositions.at(name);ensure(atoms.size()==6,"Invalid terminal composition");
            for(int k=0;k<6;++k)config.termini[k]+=atoms[k];
        }
        PTM_List ptms;ensure(ptms.populate_from_xml_config(),"Invalid PTM configuration");
        for(int c=0;c<256;++c) {
            config.modBegin[c]=mods.size();
            for(int j=0;j<ptms.size();++j)if((unsigned char)ptms.residue(j)==c) {
                ensure(ptms.symbol(j).size()==1,"CUDA generation requires single-byte PTM symbols");
                mods.push_back({ptms.mass_shift(j),ptms.symbol(j)[0]});++config.modCount[c];
            }
        }
        stride=config.maximum+std::min(config.maxPtm,config.maximum+2)+3;
        deviceConfig=std::make_unique<Buffer<GenerationConfig>>(std::vector<GenerationConfig>{config});
        deviceMods=std::make_unique<Buffer<GenerationMod>>(mods);
        resetGenerationMass();
    }
    bool readProtein(std::string& name,std::string& sequence) {
        std::string line;
        if(pending.empty()) {
            while(std::getline(fasta,line))if(!line.empty() && line[0]=='>'){pending=line;break;}
        }
        if(pending.empty())return false;
        name=pending.substr(1,pending.find_first_of(" \t\f\v\n\r")-1);pending.clear();sequence.clear();
        while(std::getline(fasta,line)) {
            if(!line.empty() && line[0]=='>'){pending=line;break;}
            for(char c:line)if(legal.find(c)!=std::string::npos)sequence.push_back(c);
        }
        ensure(sequence.size()<size_t(std::numeric_limits<int>::max()),"Protein exceeds CUDA indexing");
        return true;
    }
    bool loadBlock() {
        names.clear();sequences.clear();bases.clear();
        std::vector<char> text;std::vector<GenerationProtein> proteins;
        std::string name,sequence;
        // Protein and residue bounds keep temporary digestion storage finite.
        while(names.size()<256 && text.size()<1048576 && readProtein(name,sequence)) {
            if(sequence.empty())continue;
            proteins.push_back({text.size(),int(sequence.size())});
            text.insert(text.end(),sequence.begin(),sequence.end());
            names.push_back(name);
            if(config.removeM && sequence[0]=='M')sequence.erase(0,1);
            sequences.push_back(sequence);
        }
        if(proteins.empty())return false;
        ++blocks;
        deviceText=std::make_unique<Buffer<char>>(text);
        deviceProteins=std::make_unique<Buffer<GenerationProtein>>(proteins);
        Buffer<int> cuts(text.size()+2*proteins.size()), cutCounts(proteins.size());
        Buffer<uint64_t> counts(proteins.size());
        digestProteins<<<(proteins.size()+127)/128,128>>>(deviceText->p,deviceProteins->p,proteins.size(),
            deviceConfig->p,cuts.p,cutCounts.p,counts.p,nullptr,nullptr);
        synced();std::vector<uint64_t> hostCounts;counts.read(hostCounts);
        auto offsets=prefix(hostCounts);
        ensure(offsets.back()<=uint64_t(std::numeric_limits<int>::max()),"Digestion block exceeds CUDA indexing");
        deviceBases=std::make_unique<Buffer<GenerationBase>>(offsets.back());
        Buffer<uint64_t> deviceOffsets(offsets);
        if(offsets.back())digestProteins<<<(proteins.size()+127)/128,128>>>(deviceText->p,deviceProteins->p,proteins.size(),
            deviceConfig->p,cuts.p,cutCounts.p,counts.p,deviceOffsets.p,deviceBases->p);
        synced();deviceBases->read(bases);
        const int n=bases.size();
        Buffer<GenerationAtoms> atoms(n);Buffer<uint64_t> variantCounts(n);Buffer<int> error(1);
        check(cudaMemset(error.p,0,sizeof(int)));
        if(n)countVariants<<<(n+127)/128,128>>>(deviceText->p,deviceProteins->p,deviceBases->p,n,
            deviceConfig->p,atoms.p,variantCounts.p,error.p);
        synced();std::vector<int> errors;error.read(errors);ensure(!errors[0],"PTM combination count overflow");
        std::vector<GenerationAtoms> hostAtoms;atoms.read(hostAtoms);
        std::vector<double> masses; masses.reserve(n);
        for(const auto& atom:hostAtoms)masses.push_back(generationMass(atom));
        variantCounts.read(hostCounts);variants=prefix(hostCounts);
        total=variants.back();cursor=0;
        deviceMasses=std::make_unique<Buffer<double>>(masses);
        deviceVariants=std::make_unique<Buffer<uint64_t>>(variants);
        return true;
    }
    bool nextGpu(Peptide* peptide) {
        if(pageIndex==pageResults.size()) {
            while(cursor==total)if(!loadBlock())return false;
            const size_t count=std::min<uint64_t>(65536,total-cursor);
            Buffer<char> text(count*stride);Buffer<GenerationResult> results(count);
            emitVariants<<<(bases.size()+127)/128,128>>>(deviceText->p,deviceProteins->p,deviceBases->p,bases.size(),
                deviceConfig->p,deviceMods->p,deviceMasses->p,deviceVariants->p,cursor,cursor+count,stride,text.p,results.p);
            synced();text.read(pageText);results.read(pageResults);cursor+=count;pageIndex=0;
        }
        const auto row=pageResults[pageIndex];const auto base=bases.at(row.base);
        const auto& sequence=sequences.at(base.protein);
        const char left=base.begin ? sequence[base.begin-1] : '-';
        const char right=base.begin+base.length<int(sequence.size()) ? sequence[base.begin+base.length] : '-';
        peptide->setPeptide(pageText.data()+pageIndex*stride,"["+sequence.substr(base.begin,base.length)+"]",
            names[base.protein],base.begin,row.mass,left,right,left,right);
        ++pageIndex;++generated;return true;
    }
};
SearchPeptideGenerator::SearchPeptideGenerator(bool screen):impl(std::make_unique<Impl>(screen)) {}
SearchPeptideGenerator::~SearchPeptideGenerator() {
    if(impl->gpu)std::cout<<"[CUDA peptide generation] proteins_blocks="<<impl->blocks
                         <<" peptides="<<impl->generated<<" verified="<<impl->verify<<'\n';
    resetGenerationMass();
}
void SearchPeptideGenerator::loadDatabase() {
    if(impl->reference)impl->reference->loadDatabase();
    if(impl->gpu) {
        impl->fasta.open(ProNovoConfig::getFASTAfilename());
        ensure(impl->fasta.is_open(),"Cannot open FASTA for CUDA generation");impl->configure();
    }
    std::cout<<"[Peptide generation] backend="<<(impl->gpu?"cuda":"cpu")<<'\n';
}
bool SearchPeptideGenerator::getFirstProtein() {
    const bool first=impl->reference ? impl->reference->getFirstProtein() : true;
    return impl->gpu ? true : first;
}
bool SearchPeptideGenerator::getNextPeptide(Peptide* peptide) {
    if(!impl->gpu)return impl->reference->getNextPeptide(peptide);
    const bool found=impl->nextGpu(peptide);
    if(impl->verify) {
        Peptide expected;const bool cpu=impl->reference->getNextPeptide(&expected);
        ensure(found==cpu,"CUDA peptide generation count mismatch");
        if(found) {
            if(peptide->dPeptideMass!=expected.dPeptideMass) {
                std::cerr<<std::setprecision(17)<<"Generation mass mismatch "<<peptide->sPeptide<<" GPU="
                         <<peptide->dPeptideMass<<" CPU="<<expected.dPeptideMass<<'\n';
                throw std::runtime_error("CUDA generation mass mismatch");
            }
            ensure(peptide->sPeptide==expected.sPeptide && peptide->sOriginalPeptide==expected.sOriginalPeptide &&
                   peptide->sProteinName==expected.sProteinName && peptide->ibeginPos==expected.ibeginPos &&
                   peptide->cIdentifyPrefix==expected.cIdentifyPrefix && peptide->cIdentifySuffix==expected.cIdentifySuffix &&
                   peptide->cOriginalPrefix==expected.cOriginalPrefix && peptide->cOriginalSuffix==expected.cOriginalSuffix,
                   "CUDA peptide generation sequence/order/protein metadata mismatch");
        }
    }
    return found;
}
}
