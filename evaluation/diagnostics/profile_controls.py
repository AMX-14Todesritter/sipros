"""Profile only one scoring launch from each bounded 256-spectrum control."""
import json,os,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/evaluation/diagnostics';binary=OUT/'build/bin/sipros_mvh_cuda';folder=OUT/'profiles';folder.mkdir(exist_ok=True)
mp=OUT/'reports/control_profiles.json'
if mp.exists():raise FileExistsError('Preserve previous profiling evidence')
m={'status':'running','scope':'first fused scoring launch only; NCU replay time is not baseline performance','clock_control':'none','profiles':[]};mp.write_text(json.dumps(m,indent=2))
try:
 for precursor,cache in [('primary','auto'),('multi','auto'),('primary','disabled'),('multi','disabled')]:
  for backend in ('cuda','rt-instanced','rt-custom'):
   dest=folder/(precursor+'_'+cache+'_'+backend);dest.mkdir()
   env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(ROOT/'build/mvh_rt/optix_runtime')+':/usr/local/cuda/lib64',HOME=str(OUT/'profiler_home'),TMPDIR=str(ROOT/'output/evaluation/tmp'),OPTIX_CACHE_PATH=str(ROOT/'output/evaluation/cache/optix'),CUDA_CACHE_PATH=str(ROOT/'output/evaluation/cache/cuda'))
   Path(env['HOME']).mkdir(exist_ok=True)
   if precursor=='multi':env['SIPROS_DIAGNOSTIC_MULTI_PRECURSOR']='1'
   else:env.pop('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR',None)
   if cache=='disabled':env['SIPROS_DIAGNOSTIC_NO_ION_CACHE']='1'
   else:env.pop('SIPROS_DIAGNOSTIC_NO_ION_CACHE',None)
   search=[str(binary),'-f',str(ROOT/'output/evaluation/inputs/ecoli_x1_first256/spectra.FT2'),'-fasta',str(ROOT/'raw/Ecoli.fasta'),'-c',str(ROOT/'experiments/Regular.cfg'),'-o',str(dest/'partial_search'),'--match-backend',backend,'--peptide-batch-size','2000000','--spectrum-cache','host','--peptide-generation','cuda','--result-restoration','final']
   kernel='regex:.*'+('ScoreSequenceVsSpectrum' if backend=='cuda' else 'optixLaunch')+'.*'
   cmd=['ncu','--set','detailed','--kernel-name',kernel,'--launch-skip','0','--launch-count','1','--kill','yes','--clock-control','none','--export',str(dest/'capture'),*search]
   item={'precursor_mode':precursor,'cache_policy':cache,'backend':backend,'status':'running','command':cmd};m['profiles'].append(item);mp.write_text(json.dumps(m,indent=2));print('PROFILE',precursor,cache,backend,flush=True)
   with (dest/'capture.log').open('w') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
   item['exit_code']=r.returncode
   if r.returncode==0 and (dest/'capture.ncu-rep').is_file():
    export=['ncu','--import',str(dest/'capture.ncu-rep'),'--page','raw','--csv'];item['export_command']=export
    with (dest/'raw_metrics.csv').open('w') as f:e=subprocess.run(export,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    item.update(status='completed' if e.returncode==0 else 'failed',metrics_file=str(dest/'raw_metrics.csv'))
   else:item['status']='failed'
   mp.write_text(json.dumps(m,indent=2))
 m['status']='completed' if all(x['status']=='completed' for x in m['profiles']) else 'failed'
except BaseException as e:m.update(status='failed',error=repr(e));raise
finally:m['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();mp.write_text(json.dumps(m,indent=2))
