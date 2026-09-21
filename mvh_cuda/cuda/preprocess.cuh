#pragma once
#include "types.cuh"
namespace mvh_cuda {
// One CUDA thread per scan, preserving each original scan's sequential arithmetic.
__global__ void preprocessMvh(RawScan *scans,int size,double *mz,double *intensity,int *charge,
                             int *indices,PrepResult *out,double *keptMz,int *keptClass,Config cfg){
    int s=blockIdx.x*blockDim.x+threadIdx.x;if(s>=size)return;
    auto in=scans[s];auto &r=out[s];r={};r.skip=1;
    double *m=mz+in.offset,*v=intensity+in.offset;int *z=charge+in.offset,*ix=indices+in.offset;
    // Original sortPeakList: stable bubble sort, including peak charge association.
    for(int pass=1;pass<in.count;++pass)for(int i=0;i<in.count-pass;++i)if(m[i]>m[i+1]){
        double a=m[i];m[i]=m[i+1];m[i+1]=a;a=v[i];v[i]=v[i+1];v[i+1]=a;
        int b=z[i];z[i]=z[i+1];z[i+1]=b;
    }
    if(in.count<cfg.minClassCount)return;
    int n=0;double total=0;
    // Stable insertion into intensity order models multimap equivalent-key order.
    for(int i=0;i<in.count;++i)if(m[i]<=in.parentMass){
        total+=v[i];int pos=n;while(pos>0&&v[ix[pos-1]]>v[i]){ix[pos]=ix[pos-1];--pos;}ix[pos]=i;++n;
    }
    if(!n)return;
    double relative=0;int reverse=n-1;
    while(relative<cfg.tic&&reverse>=0){relative+=v[ix[reverse]]/total;--reverse;}
    if(reverse<0)reverse=0;
    double boundary=v[ix[reverse]];int begin=0;
    while(begin<n&&v[ix[begin]]<boundary)++begin;
    if(n-begin>cfg.maxPeaks)begin=n-cfg.maxPeaks;
    double one=in.parentMz-cfg.water/in.charge,two=in.parentMz-2*cfg.water/in.charge;
    double maxOne=0,maxTwo=0,massOne=0,massTwo=0;
    for(int j=begin;j<n;++j){int i=ix[j];
        if(m[i]>one-cfg.parentTolerance&&m[i]<one+cfg.parentTolerance){if(maxOne<v[i]){maxOne=v[i];massOne=m[i];}}
        else if(m[i]>two-cfg.parentTolerance&&m[i]<two+cfg.parentTolerance){if(maxTwo<v[i]){maxTwo=v[i];massTwo=m[i];}}
    }
    for(int water=0;water<2;++water){double target=water?massTwo:massOne,maximum=water?maxTwo:maxOne;
        if(target>0)for(int j=begin;j<n;++j)if(v[ix[j]]==maximum&&m[ix[j]]==target){for(int k=j;k<n-1;++k)ix[k]=ix[k+1];--n;break;}
    }
    double *km=keptMz+uint64_t(s)*cfg.maxPeaks;int *kc=keptClass+uint64_t(s)*cfg.maxPeaks;
    reverse=n-1;int count=0;
    for(int cls=0;cls<cfg.classes;++cls){
        int quota=int(floor(pow(cfg.multiplier,cls)*(n-begin)/cfg.minClassCount+0.5));
        for(int j=0;j<quota&&reverse>=begin;++j,--reverse){double mass=m[ix[reverse]];int pos=0;
            while(pos<count&&km[pos]<mass)++pos;
            if(pos<count&&km[pos]==mass)kc[pos]=cls+1;
            else {for(int k=count;k>pos;--k){km[k]=km[k-1];kc[k]=kc[k-1];}km[pos]=mass;kc[pos]=cls+1;++count;}
        }
    }
    r.count=count;r.skip=0;r.totalBins=int(floor((cfg.mzHigh-cfg.mzLow)/(cfg.fragmentTolerance*2.0)+0.5));
    for(int i=0;i<count;++i)++r.counts[kc[i]-1];r.counts[cfg.classes]=r.totalBins-count;
}
__global__ void sumIntensity(const RawScan *scans,int n,const double *intensity,PrepResult *out){
    int s=blockIdx.x*blockDim.x+threadIdx.x;if(s>=n)return;double sum=0,maximum=0;
    for(int i=0;i<scans[s].count;++i){double v=intensity[scans[s].offset+i];sum+=v;if(maximum<v)maximum=v;}
    out[s].sum=sum;out[s].max=maximum;
}
__global__ void preprocessingMVH(char *texts,const uint64_t *offsets,const int *capacities,int n,
                                const Rule *rules,int ruleCount,int *lengths,int *errors){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    char *text=texts+offsets[i];int len=0,residues=0;while(text[len]){if(alpha(text[len]))++residues;++len;}
    lengths[i]=residues;errors[i]=0;
    for(int r=0;r<ruleCount;++r){const Rule &rule=rules[r];int pos=0,iterations=0;
        while(pos+rule.fromLen<=len){bool found=true;
            for(int j=0;j<rule.fromLen;++j)if(text[pos+j]!=rule.from[j]){found=false;break;}
            if(!found){++pos;continue;}
            // Original neutralLossProcess replaces ONE byte, then searches from same position.
            if(++iterations>10000){errors[i]=1;return;}
            int delta=rule.toLen-1;
            if(len+delta>=capacities[i]){errors[i]=2;return;}
            if(delta>0)for(int k=len;k>pos;--k)text[k+delta]=text[k];
            if(delta<0)for(int k=pos+1;k<=len;++k)text[k+delta]=text[k];
            for(int j=0;j<rule.toLen;++j)text[pos+j]=rule.to[j];len+=delta;
        }
    }
}
}
