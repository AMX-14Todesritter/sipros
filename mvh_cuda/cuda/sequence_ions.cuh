#pragma once
#include "types.cuh"
namespace mvh_cuda {
template<class IonSink>
__device__ bool CalculateSequenceIons(const char *text,int charge,const Config &cfg,IonSink &ions){
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
}
