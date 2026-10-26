"""Build evaluation binaries with explicit scopes; all artifacts under output."""
import argparse,json,subprocess
from pathlib import Path
from inventory import ROOT,OUT,sha
from run import save
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['timing','flow','profile','cpu'],default='timing');a=p.parse_args()
kind=a.kind;dest=OUT/'build'/('gpu' if kind=='timing' else kind);logs=OUT/'results/build';logs.mkdir(parents=True,exist_ok=True)
cmd=['cmake','-S',str(ROOT/('mvh' if kind=='cpu' else 'mvh_cuda')),'-B',str(dest),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release']
if kind!='cpu':cmd+=['-DMVH_CUDA_ENABLE_RT=ON','-DMVH_ENABLE_EVALUATION_TIMING=ON','-DMVH_ENABLE_FLOW_COUNTERS='+('ON' if kind=='flow' else 'OFF'),'-DMVH_ENABLE_PROFILING='+('ON' if kind in ('flow','profile') else 'OFF'),'-DOPTIX_ROOT='+str(ROOT/'build/mvh_rt/optix_sdk')]
targets=['sipros_mvh'] if kind=='cpu' else ['sipros_mvh_cuda','mvh_cuda_contract','mvh_gpu_rt_contract','mvh_gpu_rt_sphere_contract','score_impact_contract']
commands=[cmd,['cmake','--build',str(dest),'--target',*targets,'-j','4']]
manifest={'status':'running','kind':kind,'commands':commands};mp=OUT/'manifests'/f'build_{kind}.json';save(mp,manifest)
try:
    for i,c in enumerate(commands):
        with (logs/f'{kind}_{i}.log').open('w') as log:subprocess.run(c,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=dest/'bin'/('sipros_mvh' if kind=='cpu' else 'sipros_mvh_cuda')
    manifest.update(status='completed',binary=str(binary),binary_sha256=sha(binary),cache=(dest/'CMakeCache.txt').read_text())
except BaseException as e:manifest.update(status='failed',error=repr(e));raise
finally:save(mp,manifest)
print(str(binary),flush=True)
