"""Build isolated fused-path peak identity instrumentation; production source untouched."""
import json,shutil,subprocess
from pathlib import Path
from inventory import ROOT,OUT,sha
from run import save
base=OUT/'build/peak_audit_source';logs=OUT/'results/build';logs.mkdir(parents=True,exist_ok=True)
mp=OUT/'manifests/build_peak_audit.json'
if base.exists():raise FileExistsError('Audit overlay already exists; preserve it and use existing build manifest')
base.mkdir(parents=True)
for name in ('mvh_cuda','MVH_RT'):shutil.copytree(ROOT/name,base/name,ignore=shutil.ignore_patterns('__pycache__','runs'))
for name in ('MSToolkit','shared'):(base/name).symlink_to(ROOT/name,target_is_directory=True)
def replace(rel,a,b,count=1):
 p=base/rel;s=p.read_text();assert s.count(a)==count,(rel,s.count(a),a[:100]);p.write_text(s.replace(a,b))
records=(ROOT/'evaluation/peak_audit/records.cuh').read_text()
replace('mvh_cuda/cuda/types.cuh','struct Config {',records+'\nstruct Config {\n    PeakAuditState* peakAudit;')
replace('mvh_cuda/CMakeLists.txt','set(MVH_FLOW_PTX_FLAGS)','set(MVH_FLOW_PTX_FLAGS -DMVH_EVALUATION_PEAK_AUDIT=1)')
replace('mvh_cuda/CMakeLists.txt','add_library(mvh_profiling INTERFACE)','add_library(mvh_profiling INTERFACE)\ntarget_compile_definitions(mvh_profiling INTERFACE MVH_EVALUATION_PEAK_AUDIT=1)')
replace('mvh_cuda/cuda/scoring.cuh','const double *peaks, const int *classes, const short *hub\n','const double *peaks, const int *classes, const short *hub, int* selectedPeak=nullptr\n')
replace('mvh_cuda/cuda/scoring.cuh','    if (!scan.peaks) return 0;','    if(selectedPeak)*selectedPeak=-1;\n    if (!scan.peaks) return 0;')
replace('mvh_cuda/cuda/scoring.cuh','                bestClass = candidateClass;','                bestClass = candidateClass;\n                if(selectedPeak)*selectedPeak=i;')
replace('mvh_cuda/cuda/scoring.cuh','    int key[MaxClasses+1],predicted=0,matched=0;','    int key[MaxClasses+1],predicted=0,matched=0;\n'+(ROOT/'evaluation/peak_audit/counter_methods.inc').read_text())
replace('mvh_cuda/cuda/scoring.cuh','        int cls=findNear(mz,cfg.fragmentTolerance,scan,peaks,classes,hub);','        int actualPeak=-1;\n        int cls=findNear(mz,cfg.fragmentTolerance,scan,peaks,classes,hub,&actualPeak);\n        auditMatch(mz,actualPeak);')
replace('mvh_cuda/cuda/scoring.cuh','    Counter ions(scan, cfg, peaks, classes, hub);','    Counter ions(scan, cfg, peaks, classes, hub);\n    ions.setAuditContext(index,candidate.scanId);')
replace('mvh_cuda/cuda/scoring.cuh','    if (!valid) { result.status = -1; return result; }','    ions.finishAudit();\n    if (!valid) { result.status = -1; return result; }')
replace('MVH_RT/gpu_bridge/device.cu','        const int cls=best==~0u ? 0 : classes[scan.peakOffset+best];','        auditMatch(mz,best==~0u?-1:int(best));\n        const int cls=best==~0u ? 0 : classes[scan.peakOffset+best];')
replace('MVH_RT/gpu_bridge/custom_device.cu','        const unsigned peakIndex = tracePeak(handle, mz);','        const unsigned peakIndex = tracePeak(handle, mz);\n        auditMatch(mz,peakIndex==NoPeak?-1:int(peakIndex));')
replace('mvh_cuda/cuda/engine.cu','#include "engine.h"','#include "engine.h"\n#include <set>')
replace('mvh_cuda/cuda/engine.cu','    Config scoringConfig=config;','''    Config scoringConfig=config;
    std::unique_ptr<Buffer<PeakAuditState>> peakAuditBuffer;
    if(std::getenv("SIPROS_PEAK_AUDIT_DIR")){
        peakAuditBuffer=std::make_unique<Buffer<PeakAuditState>>(1);
        check(cudaMemset(peakAuditBuffer->p,0,sizeof(PeakAuditState)));
        scoringConfig.peakAudit=peakAuditBuffer->p;
    }''')
replace('mvh_cuda/cuda/engine.cu','p.results=auditResults ? auditResults->p : results.p; p.cfg=config;','p.results=auditResults ? auditResults->p : results.p; p.cfg=scoringConfig;')
replace('mvh_cuda/cuda/engine.cu','    output.kernelSeconds = milliseconds / 1000.0;','    output.kernelSeconds = milliseconds / 1000.0;\n'+(ROOT/'evaluation/peak_audit/host_export.inc').read_text())
if (ROOT/'evaluation/peak_audit/boundary.cu').is_file():
 with (base/'mvh_cuda/CMakeLists.txt').open('a') as f:
  f.write('\nadd_executable(evaluation_fused_boundary "'+str(ROOT/'evaluation/peak_audit/boundary.cu')+'")\ntarget_link_libraries(evaluation_fused_boundary PRIVATE mvh_gpu_rt CUDA::cudart mvh_profiling)\ntarget_include_directories(evaluation_fused_boundary PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cuda")\nset_target_properties(evaluation_fused_boundary PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")\n')
commands=[['cmake','-S',str(base/'mvh_cuda'),'-B',str(OUT/'build/peak_audit'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DMVH_CUDA_ENABLE_RT=ON','-DMVH_ENABLE_EVALUATION_TIMING=ON','-DMVH_ENABLE_FLOW_COUNTERS=OFF','-DMVH_ENABLE_PROFILING=OFF','-DOPTIX_ROOT='+str(ROOT/'build/mvh_rt/optix_sdk')],['cmake','--build',str(OUT/'build/peak_audit'),'--target','sipros_mvh_cuda',*(['evaluation_fused_boundary'] if (ROOT/'evaluation/peak_audit/boundary.cu').is_file() else []),'-j','4']]
m={'status':'running','commands':commands,'source_parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'original_source_hashes':{str(p.relative_to(ROOT)):sha(p) for folder in ('mvh_cuda','MVH_RT') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.cpp','.cu','.cuh','.h')},'scope':'complete production fused scoring with observational per-query peak identity auditing; diagnostic timings excluded'};save(mp,m)
try:
 for i,cmd in enumerate(commands):
  with (logs/f'peak_audit_{i}.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 m['status']='completed';m['binary_sha256']=sha(OUT/'build/peak_audit/bin/sipros_mvh_cuda')
except BaseException as e:m.update(status='failed',error=repr(e));raise
finally:save(mp,m)
print('Fused peak-audit build completed.')
