"""Compare same-run event intervals with Nsight GPU kernel timestamps on small controls."""
import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/evaluation/diagnostics';folder=OUT/'timelines';folder.mkdir(exist_ok=True);mp=OUT/'reports/control_timelines.json'
if mp.exists():raise FileExistsError('Preserve previous timelines')
m={'status':'running','scope':'256-spectrum control; same-run CUDA events vs GPU kernel timestamps; profiler timings are not baseline speedups','runs':[]};mp.write_text(json.dumps(m,indent=2))
try:
 for precursor,cache in [('primary','auto'),('multi','auto'),('primary','disabled'),('multi','disabled')]:
  for backend in ('cuda','rt-instanced','rt-custom'):
   dest=folder/(precursor+'_'+cache+'_'+backend);dest.mkdir()
   env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(ROOT/'build/mvh_rt/optix_runtime')+':/usr/local/cuda/lib64',TMPDIR=str(ROOT/'output/evaluation/tmp'),HOME=str(OUT/'profiler_home'),OPTIX_CACHE_PATH=str(ROOT/'output/evaluation/cache/optix'),CUDA_CACHE_PATH=str(ROOT/'output/evaluation/cache/cuda'))
   for key,enabled in [('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR',precursor=='multi'),('SIPROS_DIAGNOSTIC_NO_ION_CACHE',cache=='disabled')]:
    if enabled:env[key]='1'
    else:env.pop(key,None)
   search=[str(OUT/'build/bin/sipros_mvh_cuda'),'-f',str(ROOT/'output/evaluation/inputs/ecoli_x1_first256/spectra.FT2'),'-fasta',str(ROOT/'raw/Ecoli.fasta'),'-c',str(ROOT/'experiments/Regular.cfg'),'-o',str(dest/'search'),'--match-backend',backend,'--peptide-batch-size','2000000','--spectrum-cache','host']
   commands=[['nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','--output='+str(dest/'capture'),*search],['nsys','stats','--report','cuda_gpu_kern_sum,cuda_api_sum','--format','csv','--output',str(dest/'timings'),str(dest/'capture.nsys-rep')]]
   item={'precursor_mode':precursor,'cache_policy':cache,'backend':backend,'status':'running','commands':commands,'directory':str(dest)};m['runs'].append(item);mp.write_text(json.dumps(m,indent=2));print('TIMELINE',precursor,cache,backend,flush=True)
   for i,cmd in enumerate(commands):
    with (dest/f'step_{i}.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:item.update(status='failed',exit_code=r.returncode);break
   else:item['status']='completed'
   mp.write_text(json.dumps(m,indent=2))
 m['status']='completed' if all(r['status']=='completed' for r in m['runs']) else 'failed'
except BaseException as e:m.update(status='failed',error=repr(e));raise
finally:mp.write_text(json.dumps(m,indent=2))
