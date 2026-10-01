#pragma once
#include "types.cuh"
#include "generation_types.h"
namespace mvh_cuda {
__device__ inline int generationStart(const char* sequence, int length, const GenerationConfig& cfg) {
    return length && cfg.removeM && sequence[0]=='M' ? 1 : 0;
}
__global__ void digestProteins(const char* text, const GenerationProtein* proteins, int size,
                              const GenerationConfig* config, int* cuts, int* cutCounts,
                              uint64_t* baseCounts, const uint64_t* offsets, GenerationBase* bases) {
    const int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=size)return;
    const auto& cfg=*config;
    const auto protein=proteins[p];
    const char* sequence=text+protein.offset;
    const int start=generationStart(sequence,protein.length,cfg);
    const int length=protein.length-start;
    sequence+=start;
    int* positions=cuts+protein.offset+2*uint64_t(p);
    int n;
    if (!bases) {
        n=0;positions[n++]=-1;
        for(int i=0;i+1<length;++i)
            if(cfg.after[(unsigned char)sequence[i]] && cfg.before[(unsigned char)sequence[i+1]]) positions[n++]=i;
        positions[n++]=length-1;
        cutCounts[p]=n;
    } else n=cutCounts[p];
    uint64_t count=0;
    for(int missed=0;missed<=cfg.missed && missed+1<n;++missed)
        for(int i=0;i+missed+1<n;++i) {
            const int len=positions[i+missed+1]-positions[i];
            if(len<cfg.minimum || len>cfg.maximum)continue;
            if(bases) bases[offsets[p]+count]={p,positions[i]+1,len};
            ++count;
        }
    if(!bases)baseCounts[p]=count;
}
__device__ inline char generationResidue(const char* sequence,int length,int pos) {
    return pos==0 ? '[' : pos==length+1 ? ']' : sequence[pos-1];
}
__device__ inline int generationPositions(const char* seq,int length,const GenerationConfig& cfg,int* positions) {
    int n=0;
    for(int j=0;j<length+2;++j)
        if(cfg.modCount[(unsigned char)generationResidue(seq,length,j)])positions[n++]=j;
    return n;
}
__global__ void countVariants(const char* text,const GenerationProtein* proteins,
                             const GenerationBase* bases,int size,const GenerationConfig* config,
                             GenerationAtoms* atoms,uint64_t* counts,int* error) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size)return;
    const auto& cfg=*config;const auto b=bases[i];const auto p=proteins[b.protein];
    const char* seq=text+p.offset;seq+=generationStart(seq,p.length,cfg)+b.begin;
    GenerationAtoms result{};
    for(int k=0;k<6;++k)result.count[k]=cfg.termini[k];
    for(int j=0;j<b.length;++j)for(int k=0;k<6;++k)result.count[k]+=cfg.atoms[(unsigned char)seq[j]][k];
    atoms[i]=result;
    uint64_t dp[GenerationSites+1]{};dp[0]=1;
    int sites=0;
    constexpr uint64_t limit=0x7fffffffffffffffULL;
    for(int j=0;j<b.length+2;++j) {
        const int choices=cfg.modCount[(unsigned char)generationResidue(seq,b.length,j)];
        if(!choices)continue;
        ++sites;
        for(int k=min(sites,cfg.maxPtm);k>0;--k) {
            if(dp[k-1]>(limit-dp[k])/uint64_t(choices)){atomicExch(error,1);counts[i]=0;return;}
            dp[k]+=dp[k-1]*choices;
        }
    }
    uint64_t total=0;
    for(int k=0;k<=min(sites,cfg.maxPtm);++k) {
        if(dp[k]>limit-total){atomicExch(error,1);counts[i]=0;return;}
        total+=dp[k];
    }
    counts[i]=total;
}
__device__ inline void writeGenerated(const char* seq,const GenerationBase& base,int baseId,
                                      const GenerationConfig& cfg,const GenerationMod* mods,
                                      double mass,const int* positions,const int* chosen,int count,uint64_t variant,
                                      char* target,GenerationResult& result) {
    int symbols[GenerationSites];
    for(int k=0;k<count;++k) {
        const int residue=(unsigned char)generationResidue(seq,base.length,positions[chosen[k]]);
        symbols[k]=cfg.modBegin[residue]+int(variant%cfg.modCount[residue]);
        variant/=cfg.modCount[residue];
    }
    // Original code inserts and adds shifts from the last selected site back.
    for(int k=count-1;k>=0;--k)mass=__dadd_rn(mass,mods[symbols[k]].mass);
    int used=0,k=0;
    for(int j=0;j<base.length+2;++j) {
        target[used++]=generationResidue(seq,base.length,j);
        if(k<count && positions[chosen[k]]==j)target[used++]=mods[symbols[k++]].symbol;
    }
    target[used]='\0';result={mass,baseId};
}
__global__ void emitVariants(const char* text,const GenerationProtein* proteins,
                            const GenerationBase* bases,int size,const GenerationConfig* config,
                            const GenerationMod* mods,const double* masses,const uint64_t* offsets,
                            uint64_t first,uint64_t last,int stride,char* output,GenerationResult* results) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size || offsets[i]>=last || offsets[i+1]<=first)return;
    const auto& cfg=*config;const auto b=bases[i];const auto p=proteins[b.protein];
    const char* seq=text+p.offset;seq+=generationStart(seq,p.length,cfg)+b.begin;
    int positions[GenerationSites],chosen[GenerationSites];
    const int sites=generationPositions(seq,b.length,cfg,positions);
    uint64_t cursor=offsets[i];
    if(cursor>=first)writeGenerated(seq,b,i,cfg,mods,masses[i],positions,chosen,0,0,
                                   output+(cursor-first)*stride,results[cursor-first]);
    ++cursor;
    for(int count=1;count<=min(sites,cfg.maxPtm);++count) {
        for(int k=0;k<count;++k)chosen[k]=k;
        while(true) {
            uint64_t permutations=1;
            for(int k=0;k<count;++k)
                permutations*=cfg.modCount[(unsigned char)generationResidue(seq,b.length,positions[chosen[k]])];
            const uint64_t begin=cursor>first?cursor:first;
            const uint64_t end=cursor+permutations<last?cursor+permutations:last;
            for(uint64_t index=begin;index<end;++index)
                writeGenerated(seq,b,i,cfg,mods,masses[i],positions,chosen,count,index-cursor,
                               output+(index-first)*stride,results[index-first]);
            cursor+=permutations;
            if(cursor>=last)return;
            int k=count-1;
            while(k>=0 && chosen[k]==sites-count+k)--k;
            if(k<0)break;
            ++chosen[k];for(int j=k+1;j<count;++j)chosen[j]=chosen[j-1]+1;
        }
    }
}
}
