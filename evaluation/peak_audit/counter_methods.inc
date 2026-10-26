    int auditCandidate=0,auditScan=0;
    unsigned long long auditCounts[7]{};
    __device__ void setAuditContext(int candidate,int scanIndex) {auditCandidate=candidate;auditScan=scanIndex;}
    __device__ void auditMatch(double mz,int actualPeak) {
        if(!cfg.peakAudit)return;
        int lo=0,hi=scan.peaks;
        while(lo<hi){int mid=lo+(hi-lo)/2;if(peaks[scan.peakOffset+mid]<mz)lo=mid+1;else hi=mid;}
        int expected=-1;double distance=cfg.fragmentTolerance;
        // Sorted retained peak identity, independent of geometry primitive IDs.
        for(int k=(lo?lo-1:lo);k<=lo&&k<scan.peaks;++k){
            double d=fabs(peaks[scan.peakOffset+k]-mz);
            if(d<distance){distance=d;expected=k;}
        }
        if(expected>=0){
            const double value=peaks[scan.peakOffset+expected];int first=0,last=expected;
            while(first<last){int mid=first+(last-first)/2;if(peaks[scan.peakOffset+mid]<value)first=mid+1;else last=mid;}
            expected=first;
        }
        int expectedClass=expected>=0?classes[scan.peakOffset+expected]:0;
        int actualClass=actualPeak>=0?classes[scan.peakOffset+actualPeak]:0;
        ++auditCounts[0];++auditCounts[expected>=0?1:2];++auditCounts[actualPeak>=0?5:6];
        bool idDifferent=actualPeak!=expected,classDifferent=actualClass!=expectedClass;
        auditCounts[3]+=idDifferent;auditCounts[4]+=classDifferent;
        if(idDifferent||classDifferent){
            const auto slot=atomicAdd(&cfg.peakAudit->exampleCount,1ULL);
            if(slot<128)cfg.peakAudit->examples[slot]={auditCandidate,auditScan,predicted-1,expected,actualPeak,expectedClass,actualClass,mz,cfg.fragmentTolerance,
                expected>=0?peaks[scan.peakOffset+expected]:-1.0,actualPeak>=0?peaks[scan.peakOffset+actualPeak]:-1.0};
        }
    }
    __device__ void finishAudit(){
        if(!cfg.peakAudit)return;
        for(int i=0;i<7;++i)if(auditCounts[i])atomicAdd(&cfg.peakAudit->shards[auditCandidate&4095][i],auditCounts[i]);
    }
