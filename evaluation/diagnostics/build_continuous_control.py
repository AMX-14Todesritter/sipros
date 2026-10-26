"""Isolated host replay with freshly rebuilt cache and direct wall timing."""
import shutil,subprocess,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'output/evaluation/diagnostics/continuous_control';D.mkdir(exist_ok=False)
S=D/'source';B=D/'build';shutil.copytree(R/'output/evaluation/diagnostics/build_source',S,symlinks=True)
p=S/'mvh_cuda/cuda/engine.cu';s=p.read_text();a='    MVH_PROFILE_BEGIN(theoryRange, "mvh/gpu/theory_cache");'
s=s.replace(a,'''    const bool continuousReplay=std::getenv("SIPROS_CONTINUOUS_REPLAY")!=nullptr;
    const bool initialCache=useIonCache;
    const int repeats=continuousReplay?20:1;
    for(int iteration=0;iteration<repeats;++iteration) {
    const int pair=iteration/2;
    const bool cacheThis=continuousReplay ? ((iteration%2)==(pair%2)) : initialCache;
    useIonCache=cacheThis;
    output.cachedIonCount=0;output.chargeStride=0;
    synced();
    const double continuousStart=now();
'''+a)
a='#ifdef MVH_ENABLE_FLOW_COUNTERS\n    mvh_flow::time("gpu/scoring_fused",output.kernelSeconds);'
s=s.replace(a,'''    const double continuousEnd=now();
    if(continuousReplay)std::cout << std::setprecision(12)
      << "[CONTINUOUS replay] trial=" << pair << " warmup=" << (pair<3)
      << " cache=" << (cacheThis?"auto":"disabled") << " candidates=" << size
      << " theory_seconds=" << output.theorySeconds
      << " fused_seconds=" << output.fusedDeviceSeconds
      << " continuous_seconds=" << continuousEnd-continuousStart
      << " cached_ions=" << output.cachedIonCount << " backend=" << matchBackend << '\\n';
    }
    useIonCache=initialCache;
'''+a)
s=s.replace('    std::unique_ptr<Buffer<Result>> scoreReference;','').replace('    const bool continuousReplay=', '    std::unique_ptr<Buffer<Result>> scoreReference;\n    const bool continuousReplay=')
p.write_text(s)
protected=['mvh_cuda/cuda/scoring.cuh','mvh_cuda/cuda/theoretical.cuh','MVH_RT/gpu_bridge/device.cu','MVH_RT/gpu_bridge/custom_device.cu','MVH_RT/gpu_bridge/custom_geometry.cu']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(R/f)==sha(S/f) for f in protected)
commands=[['cmake','-S',str(S/'mvh_cuda'),'-B',str(B),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DMVH_CUDA_ENABLE_RT=ON','-DMVH_ENABLE_EVALUATION_TIMING=ON','-DMVH_ENABLE_PROFILING=OFF','-DMVH_ENABLE_FLOW_COUNTERS=OFF','-DOPTIX_ROOT='+str(R/'build/mvh_rt/optix_sdk')],['cmake','--build',str(B),'--target','sipros_mvh_cuda','-j','4']]
m={'status':'running','commands':commands,'protected_hashes':{f:sha(S/f) for f in protected}};mp=D/'build.json'
try:
 for i,c in enumerate(commands):
  mp.write_text(json.dumps(m,indent=2))
  with (D/f'build_{i}.log').open('w') as log:subprocess.run(c,stdout=log,stderr=subprocess.STDOUT,check=True)
 m.update(status='completed',binary_sha256=sha(B/'bin/sipros_mvh_cuda'))
finally:mp.write_text(json.dumps(m,indent=2))
