"""Observational per-candidate query fingerprints plus existing peak oracle audit."""
import shutil,subprocess,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'output/evaluation/diagnostics/marine_query_audit';D.mkdir(exist_ok=False);S=D/'source';B=D/'build'
shutil.copytree(R/'output/evaluation/build/peak_audit_source',S,symlinks=True)
shutil.copyfile(R/'output/evaluation/diagnostics/build_source/mvh_cuda/src/spectrum_input.cpp',S/'mvh_cuda/src/spectrum_input.cpp')
def edit(rel,a,b):
 p=S/rel;s=p.read_text();assert s.count(a)==1,(rel,a,s.count(a));p.write_text(s.replace(a,b))
edit('mvh_cuda/cuda/types.cuh','    PeakAuditState* peakAudit;','    PeakAuditState* peakAudit;\n    unsigned long long* queryFingerprint;')
edit('mvh_cuda/cuda/scoring.cuh','    int auditCandidate=0,auditScan=0;', '''    unsigned long long queryCount=0,queryH1=1469598103934665603ULL,queryH2=1099511628211ULL;
    __device__ void fingerprintQuery(double mz) {
        if(!cfg.queryFingerprint)return;
        const unsigned long long bits=__double_as_longlong(mz);
        ++queryCount;queryH1=(queryH1^bits)*1099511628211ULL;
        queryH2=(queryH2+bits+0x9e3779b97f4a7c15ULL);queryH2=(queryH2<<17)|(queryH2>>47);
    }
    int auditCandidate=0,auditScan=0;''')
# Observe every in-range matching request before either backend resolves a peak.
edit('mvh_cuda/cuda/scoring.cuh','if(mz<scan.lower||mz>scan.upper)return;++predicted;','if(mz<scan.lower||mz>scan.upper)return;++predicted;\n        fingerprintQuery(mz);')
for f in ('MVH_RT/gpu_bridge/device.cu','MVH_RT/gpu_bridge/custom_device.cu'):
 edit(f,'        ++predicted;','        ++predicted;\n        fingerprintQuery(mz);')
edit('mvh_cuda/cuda/scoring.cuh','    __device__ void finishAudit(){','''    __device__ void finishAudit(){
        if(cfg.queryFingerprint){
            auto* out=cfg.queryFingerprint+3ULL*auditCandidate;
            out[0]=queryCount;out[1]=queryH1;out[2]=queryH2;
        }''')
edit('mvh_cuda/cuda/engine.cu','bool useIonCache=true;','bool useIonCache=std::getenv("SIPROS_DIAGNOSTIC_NO_ION_CACHE")==nullptr;')
edit('mvh_cuda/cuda/engine.cu','    std::unique_ptr<Buffer<PeakAuditState>> peakAuditBuffer;', '''    std::unique_ptr<Buffer<PeakAuditState>> peakAuditBuffer;
    std::unique_ptr<Buffer<unsigned long long>> queryBuffer;
    if(std::getenv("SIPROS_PEAK_AUDIT_DIR")){
        queryBuffer=std::make_unique<Buffer<unsigned long long>>(3ULL*size);
        check(cudaMemset(queryBuffer->p,0,3ULL*size*sizeof(unsigned long long)));
        scoringConfig.queryFingerprint=queryBuffer->p;
    }''')
edit('mvh_cuda/cuda/engine.cu','        const auto summaryPath=dir/"peak_audit_batches.tsv";', '''        std::vector<unsigned long long> queryHost;queryBuffer->read(queryHost);
        std::vector<Candidate> candidateHost;const_cast<Buffer<Candidate>&>(candidates).read(candidateHost);
        std::vector<char> textHost;const_cast<Buffer<char>&>(texts).read(textHost);
        std::ofstream fingerprints(dir/("queries_"+std::to_string(auditBatch)+".tsv"));
        fingerprints.exceptions(std::ios::failbit|std::ios::badbit);fingerprints<<std::setprecision(17);
        fingerprints<<"index\\tscan_id\\tcharge\\tprecursor_mass\\tsequence\\tquery_count\\thash1\\thash2\\n";
        for(int i=0;i<size;++i){const auto& c=candidateHost[i];
            fingerprints<<i<<'\\t'<<batch.spectra->owners.at(c.scanId)->iScanId<<'\\t'<<c.charge
             <<'\\t'<<batch.assignment->precursors.at(c.precursorId).mass
             <<'\\t'<<(textHost.data()+batch.peptides.at(c.peptideId).text)
             <<'\\t'<<queryHost[3ULL*i]<<'\\t'<<queryHost[3ULL*i+1]<<'\\t'<<queryHost[3ULL*i+2]<<'\\n';}
        std::ofstream spectrumDump(dir/("spectra_"+std::to_string(auditBatch)+".tsv"));spectrumDump<<std::setprecision(17);
        for(size_t i=0;i<batch.scans.size();++i){const auto& scan=batch.scans[i];
            spectrumDump<<"S\\t"<<batch.spectra->owners.at(i)->iScanId<<'\\t'<<scan.skip<<'\\t'<<scan.lower<<'\\t'<<scan.upper<<'\\n';
            for(int k=0;k<scan.peaks;++k)spectrumDump<<batch.spectra->peaks.at(scan.peakOffset+k)<<'\\t'<<batch.spectra->classes.at(scan.peakOffset+k)<<'\\n';}
        const auto summaryPath=dir/"peak_audit_batches.tsv";''')
commands=[['cmake','-S',str(S/'mvh_cuda'),'-B',str(B),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DMVH_CUDA_ENABLE_RT=ON','-DMVH_ENABLE_EVALUATION_TIMING=ON','-DMVH_ENABLE_FLOW_COUNTERS=OFF','-DMVH_ENABLE_PROFILING=OFF','-DOPTIX_ROOT='+str(R/'build/mvh_rt/optix_sdk')],['cmake','--build',str(B),'--target','sipros_mvh_cuda','-j','4']]
m={'status':'running','scope':'diagnostic query observer and peak oracle only; instrumented timings excluded','commands':commands};mp=D/'build.json'
try:
 for i,c in enumerate(commands):
  mp.write_text(json.dumps(m,indent=2))
  with (D/f'build_{i}.log').open('w') as log:subprocess.run(c,stdout=log,stderr=subprocess.STDOUT,check=True)
 m.update(status='completed',binary_sha256=hashlib.sha256((B/'bin/sipros_mvh_cuda').read_bytes()).hexdigest())
finally:mp.write_text(json.dumps(m,indent=2))
