"""Launch-only diagnostic replay: warm/repeat immutable scoring inputs in one CUDA context."""
import shutil,subprocess,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/evaluation/diagnostics';source=OUT/'hot_source';build=OUT/'hot_build'
if source.exists():raise FileExistsError('Preserve previous hot control source')
shutil.copytree(OUT/'build_source',source,symlinks=True)
p=source/'mvh_cuda/cuda/engine.cu';s=p.read_text();a='    check(cudaEventRecord(kernelStart.event));';assert s.count(a)==1
s=s.replace(a,'''    // Diagnostic driver only: scoreCandidate writes results but never updates Top.
    // Repeating this launch overwrites identical Result slots; retention executes once.
    const bool diagnosticReplay = std::getenv("SIPROS_DIAGNOSTIC_HOT_REPLAY") != nullptr;
    const int replayWarmups = diagnosticReplay ? 3 : 0;
    const int replayMeasured = diagnosticReplay ? 7 : 1;
    for (int trial=-replayWarmups; trial<replayMeasured; ++trial) {
    check(cudaEventRecord(kernelStart.event));''')
anchor='''#ifdef MVH_ENABLE_FLOW_COUNTERS
    mvh_flow::time("gpu/scoring_fused",output.kernelSeconds);'''
assert s.count(anchor)==1
s=s.replace(anchor,'''    if(diagnosticReplay)std::cout << std::setprecision(12)
        << "[FUSED diagnostic replay] trial=" << trial
        << " warmup=" << (trial<0) << " candidates=" << size
        << " fused_device_seconds=" << output.fusedDeviceSeconds
        << " event_interval_seconds=" << output.kernelSeconds
        << " cached_ions=" << output.cachedIonCount << " backend=" << matchBackend << '\\n';
    }
'''+anchor)
p.write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected=['mvh_cuda/cuda/scoring.cuh','mvh_cuda/cuda/theoretical.cuh','MVH_RT/gpu_bridge/device.cu','MVH_RT/gpu_bridge/custom_device.cu','MVH_RT/gpu_bridge/custom_geometry.cu']
assert all(sha(ROOT/f)==sha(source/f) for f in protected)
cmds=[['cmake','-S',str(source/'mvh_cuda'),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DMVH_CUDA_ENABLE_RT=ON','-DMVH_ENABLE_EVALUATION_TIMING=ON','-DMVH_ENABLE_PROFILING=OFF','-DMVH_ENABLE_FLOW_COUNTERS=OFF','-DOPTIX_ROOT='+str(ROOT/'build/mvh_rt/optix_sdk')],['cmake','--build',str(build),'--target','sipros_mvh_cuda','-j','4']]
m={'status':'running','commands':cmds,'protected_source_sha256':{f:sha(source/f) for f in protected},'scope':'launch-only replay, no changes to ion formula, matching, geometry, score or retention algorithms'};mp=OUT/'reports/hot_build.json';mp.write_text(json.dumps(m,indent=2))
try:
 for i,cmd in enumerate(cmds):
  with (OUT/'results'/f'hot_build_{i}.log').open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 m.update(status='completed',binary_sha256=sha(build/'bin/sipros_mvh_cuda'))
except BaseException as e:m.update(status='failed',error=repr(e));raise
finally:mp.write_text(json.dumps(m,indent=2))
