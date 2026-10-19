#pragma once
#include "types.cuh"
#include <cmath>
namespace mvh_cuda {
// Preserve device arithmetic; host builds use the same operation ordering.
__host__ __device__ inline double ionMul(double a,double b){
#ifdef __CUDA_ARCH__
 return __dmul_rn(a,b);
#else
 return a*b;
#endif
}
__host__ __device__ inline double ionDiv(double a,double b){
#ifdef __CUDA_ARCH__
 return __ddiv_rn(a,b);
#else
 return a/b;
#endif
}
__host__ __device__ inline double ionAdd(double a,double b){
#ifdef __CUDA_ARCH__
 return __dadd_rn(a,b);
#else
 return a+b;
#endif
}
__host__ __device__ inline double ionFma(double a,double b,double c){
#ifdef __CUDA_ARCH__
 return __fma_rn(a,b,c);
#else
 return std::fma(a,b,c);
#endif
}
template<class IonSink>
__host__ __device__ bool CalculateSequenceIons(const char *text,int charge,const Config &cfg,IonSink &ions){
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
                double ratio=ionMul(double(bs*4+bw*2+c),ionDiv(1.0,double(total)));int bz=1;
                for(int z=1;z<charge-1;++z)if(ionMul(double(z),ionDiv(1.0,double(charge-1)))<=ratio)bz=z+1;else break;
                int yz=charge-bz;
                if(c>0&&cfg.bIon)ions.add(ionDiv(ionFma(cfg.proton,double(bz),forward[c-1]),double(bz)));
                if(c<n&&cfg.yIon)ions.add(ionDiv(ionFma(cfg.proton,double(yz),reverse[n-1-c]),double(yz)));
            }
        }else for(int z=1;z<charge;++z)for(int c=0;c<n;++c){
            if(c>0&&cfg.bIon)ions.add(ionMul(ionAdd(forward[c-1],ionMul(cfg.proton,double(z))),ionDiv(1.0,double(z))));
            if(cfg.yIon)ions.add(ionMul(ionAdd(reverse[c],ionMul(cfg.proton,double(z))),ionDiv(1.0,double(z))));
        }
    }else {
        for(int c=0;c<n-1;++c){if(cfg.bIon)ions.add(forward[c]+cfg.proton);if(cfg.yIon)ions.add(reverse[c]+cfg.proton);}
        if(cfg.yIon)ions.add(reverse[n-1]+cfg.proton);
    }
    return true;
}
__host__ __device__ inline double lnCombin(int n,int k,const double *table){if(n<0||k<0||n<k)return -1;return (table[n]-table[n-k])-table[k];}
}
