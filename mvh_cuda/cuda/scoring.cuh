#pragma once
#include "types.cuh"
namespace mvh_cuda {
__device__ inline void exchange(Top &a,Top &b){Top t=a;a=b;b=t;}
__device__ inline bool GreaterScore(const Top &a,const Top &b){return a.score>b.score;}
__device__ void pushHeap(Top *a,int hole,int top,Top value){
    int parent=(hole-1)/2;while(hole>top&&GreaterScore(a[parent],value)){a[hole]=a[parent];hole=parent;parent=(hole-1)/2;}a[hole]=value;
}
__device__ void adjustHeap(Top *a,int hole,int n,Top value){
    int top=hole,child=hole;
    while(child<(n-1)/2){child=2*(child+1);if(GreaterScore(a[child],a[child-1]))--child;a[hole]=a[child];hole=child;}
    if((n&1)==0&&child==(n-2)/2){child=2*(child+1);a[hole]=a[child-1];hole=child-1;}
    pushHeap(a,hole,top,value);
}
__device__ void heapSort(Top *a,int n){
    if(n<2)return;for(int parent=(n-2)/2;;--parent){adjustHeap(a,parent,n,a[parent]);if(!parent)break;}
    for(int end=n-1;end>0;--end){Top value=a[end];a[end]=a[0];adjustHeap(a,0,end,value);}
}
// Match GCC/libstdc++'s std::sort control flow, including equal-score permutations.
// This matters because original mergePeptide searches only the CURRENT top list.
__device__ void saveScoreSort(Top *a,int n){
    if(n<2)return;
    int initialDepth=0;for(int k=n;k>1;k>>=1)++initialDepth;initialDepth*=2;
    int loStack[32],hiStack[32],depthStack[32],sp=0;
    loStack[0]=0;hiStack[0]=n;depthStack[0]=initialDepth;
    while(sp>=0){int lo=loStack[sp],hi=hiStack[sp],depth=depthStack[sp--];
        while(hi-lo>16){if(!depth){heapSort(a+lo,hi-lo);break;}--depth;
            int x=lo+1,y=lo+(hi-lo)/2,z=hi-1,median;
            if(GreaterScore(a[x],a[y])){if(GreaterScore(a[y],a[z]))median=y;else if(GreaterScore(a[x],a[z]))median=z;else median=x;}
            else if(GreaterScore(a[x],a[z]))median=x;else if(GreaterScore(a[y],a[z]))median=z;else median=y;
            exchange(a[lo],a[median]);int left=lo+1,right=hi;
            while(true){while(GreaterScore(a[left],a[lo]))++left;--right;while(GreaterScore(a[lo],a[right]))--right;
                if(left>=right)break;exchange(a[left],a[right]);++left;}
            ++sp;loStack[sp]=lo;hiStack[sp]=left;depthStack[sp]=depth;lo=left;
        }
    }
    // Guarded insertion has the same shifts as the final guarded/unguarded passes.
    for(int i=1;i<n;++i){Top value=a[i];int j=i;while(j>0&&GreaterScore(value,a[j-1])){a[j]=a[j-1];--j;}a[j]=value;}
}
__device__ int findNear(double mz,double tolerance,const Scan &s,const double *peaks,const int *classes,const short *hub){
    if(!s.peaks)return 0;int lower=int(mz-tolerance),upper=int(mz+tolerance);
    if(upper<s.lowest||lower>s.highest)return 0;
    int begin=lower>=s.lowest?lower-s.lowest:0,end=upper<=s.highest?upper-s.lowest:s.highest-s.lowest;
    double best=1000000;int cls=0;
    for(int bucket=begin;bucket<=end;++bucket){int first=hub[s.hubOffset+2*bucket],last=hub[s.hubOffset+2*bucket+1];
        if(first!=-1)for(int i=first;i<last;++i){double error=fabs(mz-peaks[s.peakOffset+i]);if(error<best){best=error;cls=classes[s.peakOffset+i];}}
    }
    return best<tolerance?cls:0;
}
struct IonCounter {
    const Scan &scan;const Config &cfg;const double *peaks;const int *classes;const short *hub;
    int key[MaxClasses+1],predicted=0,matched=0;
    __device__ IonCounter(const Scan&s,const Config&c,const double*p,const int*cl,const short*h):scan(s),cfg(c),peaks(p),classes(cl),hub(h){for(int i=0;i<=MaxClasses;++i)key[i]=0;}
    __device__ void add(double mz){if(mz<scan.lower||mz>scan.upper)return;++predicted;
        int cls=findNear(mz,cfg.fragmentTolerance,scan,peaks,classes,hub);
        if(cls>0){++key[cls-1];++matched;}else ++key[cfg.classes];
    }
};
__device__ bool CalculateSequenceIons(const char *text,int charge,const Config &cfg,IonCounter &ions){
    int bytes=0,n=0;char seq[MaxLength];double forward[MaxLength],reverse[MaxLength];
    while(text[bytes]){if(alpha(text[bytes])){if(n>=MaxLength)return false;seq[n++]=text[bytes];}++bytes;}
    if(n<cfg.minLength||text[0]!='['||charge<1)return false;
    int j=1,k=bytes-1;double b=0,y=cfg.yWater;
    if(!alpha(text[j])){double m=cfg.mass[(unsigned char)text[j++]];if(m==cfg.missingMass)return false;b+=m;}
    if(text[k]==']')--k;else {double m=cfg.mass[(unsigned char)text[k--]];if(m==cfg.missingMass)return false;y+=m;if(text[k--]!=']')return false;}
    for(int i=0;i<n;++i){if(!alpha(text[j]))return false;double m=cfg.mass[(unsigned char)text[j++]];if(m==cfg.missingMass)return false;b+=m;
        if(j<bytes&&!alpha(text[j])&&text[j]!=']'){m=cfg.mass[(unsigned char)text[j++]];if(m==cfg.missingMass)return false;b+=m;}forward[i]=b;
        if(!alpha(text[k])){m=cfg.mass[(unsigned char)text[k--]];if(m==cfg.missingMass)return false;y+=m;}
        if(!alpha(text[k]))return false;m=cfg.mass[(unsigned char)text[k--]];if(m==cfg.missingMass)return false;y+=m;reverse[i]=y;
    }
    if(charge>2){
        if(cfg.smart){int strong=0,weak=0;
            for(int i=0;i<n;++i)if(seq[i]=='R'||seq[i]=='K'||seq[i]=='H')++strong;else if(seq[i]=='Q'||seq[i]=='N')++weak;
            int total=strong*4+weak*2+n-2;
            for(int c=0;c<=n;++c){int bs=0,bw=0;for(int i=0;i<c;++i)if(seq[i]=='R'||seq[i]=='K'||seq[i]=='H')++bs;else if(seq[i]=='Q'||seq[i]=='N')++bw;
                // Original GCC -ffast-math hoists reciprocal divisors in these loops.
                double ratio=__dmul_rn(double(bs*4+bw*2+c),__ddiv_rn(1.0,double(total)));int bz=1;
                for(int z=1;z<charge-1;++z)if(__dmul_rn(double(z),__ddiv_rn(1.0,double(charge-1)))<=ratio)bz=z+1;else break;
                int yz=charge-bz;
                if(c>0&&cfg.bIon)ions.add(__ddiv_rn(__fma_rn(cfg.proton,double(bz),forward[c-1]),double(bz)));
                if(c<n&&cfg.yIon)ions.add(__ddiv_rn(__fma_rn(cfg.proton,double(yz),reverse[n-1-c]),double(yz)));
            }
        }else for(int z=1;z<charge;++z)for(int c=0;c<n;++c){
            if(c>0&&cfg.bIon)ions.add(__dmul_rn(__dadd_rn(forward[c-1],__dmul_rn(cfg.proton,double(z))),__ddiv_rn(1.0,double(z))));
            if(cfg.yIon)ions.add(__dmul_rn(__dadd_rn(reverse[c],__dmul_rn(cfg.proton,double(z))),__ddiv_rn(1.0,double(z))));
        }
    }else {
        for(int c=0;c<n-1;++c){if(cfg.bIon)ions.add(forward[c]+cfg.proton);if(cfg.yIon)ions.add(reverse[c]+cfg.proton);}
        if(cfg.yIon)ions.add(reverse[n-1]+cfg.proton);
    }
    return true;
}
__device__ double lnCombin(int n,int k,const double *table){if(n<0||k<0||n<k)return -1;return (table[n]-table[n-k])-table[k];}
__global__ void scorePeptidesMVH(const Scan *scans,int size,const Candidate *candidates,const char *texts,
                                const double *peaks,const int *classes,const short *hub,
                                const double *lnTable,const Top *initial,Top *finalTop,Result *results,Config cfg){
    int scanId=blockIdx.x*blockDim.x+threadIdx.x;if(scanId>=size)return;
    const Scan &s=scans[scanId];Top top[TopN];int count=s.topCount;
    for(int i=0;i<count;++i)top[i]=initial[uint64_t(scanId)*TopN+i];
    for(int i=0;i<s.candidates;++i){auto ix=s.candidateOffset+i;auto c=candidates[ix];Result r{};
        if(s.skip){r.status=ResultSkipped;results[ix]=r;continue;}
        bool merged=false;for(int k=0;k<count;++k)if(top[k].sequenceId==c.sequenceId){merged=true;break;}
        if(merged){r.status=ResultMerged;results[ix]=r;continue;}
        IonCounter ions(s,cfg,peaks,classes,hub);
        if(!CalculateSequenceIons(texts+c.text,c.charge,cfg,ions)){r.status=-1;results[ix]=r;continue;}
        r.predicted=ions.predicted;r.matched=ions.matched;r.status=ResultInsufficient;
        if(ions.matched&&ions.matched>=cfg.minMatched){double value=0;
            for(int k=0;k<=cfg.classes;++k)value+=lnCombin(s.counts[k],ions.key[k],lnTable);
            value-=lnCombin(s.totalBins,ions.predicted,lnTable);r.score=-value;r.status=ResultScored;
            if(count<TopN){top[count++]={r.score,c.sequenceId};saveScoreSort(top,count);}
            else if(r.score>top[TopN-1].score){top[TopN-1]={r.score,c.sequenceId};saveScoreSort(top,count);}
        }
        results[ix]=r;
    }
    for(int k=0;k<count;++k)finalTop[uint64_t(scanId)*TopN+k]=top[k];
}
__global__ void sortTest(Top *data,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)saveScoreSort(data+uint64_t(i)*TopN,TopN);}
}
namespace mvh_cuda {
__global__ void matchContract(Scan scan,const double *peaks,const int *classes,const short *hub,
                              const double *queries,const double *tolerances,int n,int *out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=findNear(queries[i],tolerances[i],scan,peaks,classes,hub);
}
}
