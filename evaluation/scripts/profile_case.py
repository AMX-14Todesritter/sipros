"""Serial Nsight capture with explicit commands/counters and no baseline timing claims."""
import argparse,fcntl,json,os,re,subprocess
from pathlib import Path
from inventory import ROOT,OUT,sha
from run import save
p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--dataset',required=True);p.add_argument('--backend',choices=['cuda','rt-instanced','rt-custom'],required=True);p.add_argument('--tool',choices=['nsys','ncu'],required=True);a=p.parse_args()
w=json.loads((ROOT/'evaluation/configs/workloads.json').read_text())['workloads'][a.dataset]
dest=OUT/'results/profiling'/a.id;dest.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env.update(TMPDIR=str(OUT/'tmp'),LD_LIBRARY_PATH=str(ROOT/'build/mvh_rt/optix_runtime')+':/usr/local/cuda/lib64',OPTIX_CACHE_PATH=str(OUT/'cache/optix'),CUDA_CACHE_PATH=str(OUT/'cache/cuda'))
binary=OUT/'build/profile/bin/sipros_mvh_cuda'
search=[str(binary),'-f',str(ROOT/w['ft2']),'-fasta',str(ROOT/w['fasta']),'-c',str(ROOT/'experiments/Regular.cfg'),'-o',str(dest/'search'),'--match-backend',a.backend,'--peptide-batch-size','2000000','--spectrum-cache','host','--peptide-generation','cuda','--result-restoration','final']
if a.tool=='nsys':
 commands=[['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--output='+str(dest/'capture'),*search],['nsys','stats','--report','nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum','--format','csv','--output',str(dest/'timings'),str(dest/'capture.nsys-rep')]]
else:
 kernel='regex:.*'+('ScoreSequenceVsSpectrum' if a.backend=='cuda' else 'optixLaunch')+'.*'
 commands=[['ncu','--query-metrics'],['ncu','--set','detailed','--kernel-name',kernel,'--launch-skip','0','--launch-count','1','--kill','yes','--clock-control','none','--export',str(dest/'capture'),*search],['ncu','--import',str(dest/'capture.ncu-rep'),'--page','raw','--csv']]
m={'status':'running','dataset':a.dataset,'backend':a.backend,'tool':a.tool,'commands':commands,'binary_sha256':sha(binary),'inputs_sha256':{x:sha(ROOT/w[x]) for x in ('fasta','ft2')},'scope':'profiled fused scoring; NCU first scoring launch only, intentionally terminates search','steps':[],'clocks_locked':False}
mp=dest/'profile_manifest.json';save(mp,m)
try:
 with (OUT/'.benchmark.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for i,cmd in enumerate(commands):
   with (dest/f'step_{i}.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   m['steps'].append({'command':cmd,'exit_code':r.returncode});save(mp,m)
   if r.returncode:raise RuntimeError(a.tool+' command failed; see step '+str(i))
  if not (dest/('capture.nsys-rep' if a.tool=='nsys' else 'capture.ncu-rep')).is_file():raise RuntimeError('Profiler generated no capture')
  if a.tool=='ncu':
   text=(dest/'step_1.log').read_text();n=len(re.findall(r'==PROF== Profiling "',text));m['collected_scoring_launches']=n
   if n!=1:raise RuntimeError('Unexpected profiled launch count '+str(n))
  else:m['psm_sha256']=sha(dest/'search/mvh_psms.tsv')
  m['status']='completed'
except BaseException as e:m.update(status='failed',error=repr(e));raise
finally:save(mp,m)
