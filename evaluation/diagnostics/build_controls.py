"""Isolated precursor/cache controls; matching and scoring source stays identical."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/evaluation/diagnostics';source=OUT/'build_source';build=OUT/'build'
sys.path.insert(0,str(ROOT/'mvh/tests'));from source_methods import methods
if source.exists():raise FileExistsError('Preserve existing control build; do not overwrite')
source.mkdir()
for name in ('mvh_cuda','MVH_RT'):shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__','runs'))
for name in ('MSToolkit','shared'):(source/name).symlink_to(ROOT/name,target_is_directory=True)
p=source/'mvh_cuda/src/spectrum_input.cpp';current=p.read_text();old=subprocess.check_output(['git','show','82522a4:mvh_cuda/src/spectrum_input.cpp'],cwd=ROOT,text=True)
a,b=methods(current,'MvhScanVector'),methods(old,'MvhScanVector')
for name in ('ReadFT2File','loadFT2File','saveFT2Scan'):
 new,legacy=a[name],b[name];signature=new[:new.index('{')]
 body=signature+'{\n    if(std::getenv("SIPROS_DIAGNOSTIC_MULTI_PRECURSOR")) {\n'+legacy[legacy.index('{')+1:-1]+'\n    } else {\n'+new[new.index('{')+1:-1]+'\n    }\n}'
 assert current.count(new)==1;current=current.replace(new,body)
p.write_text('#include <cstdlib>\n'+current)
p=source/'mvh_cuda/cuda/engine.cu';text=p.read_text();assert text.count('bool useIonCache=true;')==1
p.write_text(text.replace('bool useIonCache=true;','bool useIonCache=std::getenv("SIPROS_DIAGNOSTIC_NO_ION_CACHE")==nullptr;'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected=['mvh_cuda/cuda/scoring.cuh','mvh_cuda/cuda/theoretical.cuh','MVH_RT/gpu_bridge/device.cu','MVH_RT/gpu_bridge/custom_device.cu','MVH_RT/gpu_bridge/custom_geometry.cu']
assert all(sha(ROOT/f)==sha(source/f) for f in protected)
commands=[['cmake','-S',str(source/'mvh_cuda'),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DMVH_CUDA_ENABLE_RT=ON','-DMVH_ENABLE_EVALUATION_TIMING=ON','-DMVH_ENABLE_PROFILING=OFF','-DMVH_ENABLE_FLOW_COUNTERS=OFF','-DOPTIX_ROOT='+str(ROOT/'build/mvh_rt/optix_sdk')],['cmake','--build',str(build),'--target','sipros_mvh_cuda','-j','4']]
m={'status':'running','commands':commands,'historical_reader_commit':'82522a4','parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'controls':['only the three FT2 precursor-reader methods select historical/current mode','existing theoretical-ion-cache enable bool controlled by environment; equations unchanged'],'protected_matching_sha256':{f:sha(source/f) for f in protected}}
mp=OUT/'reports/control_build.json';mp.write_text(json.dumps(m,indent=2))
try:
 for i,cmd in enumerate(commands):
  with (OUT/'results'/f'control_build_{i}.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 m.update(status='completed',binary_sha256=sha(build/'bin/sipros_mvh_cuda'),modified_files={f:sha(source/f) for f in ['mvh_cuda/src/spectrum_input.cpp','mvh_cuda/cuda/engine.cu']})
except BaseException as e:m.update(status='failed',error=repr(e));raise
finally:mp.write_text(json.dumps(m,indent=2))
