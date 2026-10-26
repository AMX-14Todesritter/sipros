"""Bounded in-process hot scoring diagnostics, results remain non-application benchmarks."""
import subprocess,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/evaluation/diagnostics';mp=OUT/'reports/hot_controls.json'
if mp.exists():raise FileExistsError('Preserve previous hot runs')
m={'status':'running','scope':'256 X1 spectra, 3 warmups + 7 repetitions of each immutable scoring batch within one process; Top retention once','cells':[]};mp.write_text(json.dumps(m,indent=2))
try:
 for precursor,cache in [('primary','auto'),('multi','auto'),('primary','disabled'),('multi','disabled')]:
  env=os.environ.copy();env['SIPROS_DIAGNOSTIC_HOT_REPLAY']='1'
  for k,yes in [('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR',precursor=='multi'),('SIPROS_DIAGNOSTIC_NO_ION_CACHE',cache=='disabled')]:
   if yes:env[k]='1'
   else:env.pop(k,None)
  rid='diagnostic_hot256_'+precursor+'_'+cache
  cmd=[sys.executable,'-B',str(ROOT/'evaluation/scripts/run.py'),'--id',rid,'--category','accuracy','--scans',str(ROOT/'output/evaluation/inputs/ecoli_x1_first256/spectra.FT2'),'--fasta',str(ROOT/'raw/Ecoli.fasta'),'--backends','cuda','rt-instanced','rt-custom','--batch','2000000','--repeats','1','--warmups','0','--gpu-binary',str(OUT/'hot_build/bin/sipros_mvh_cuda'),'--continue-on-backend-failure']
  item={'precursor_mode':precursor,'cache_policy':cache,'command':cmd,'status':'running','run_manifest':str(ROOT/'output/evaluation/results/accuracy'/rid/'manifest.json')};m['cells'].append(item);mp.write_text(json.dumps(m,indent=2))
  with (OUT/'results'/(rid+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  item.update(exit_code=r.returncode,status='completed' if r.returncode==0 else 'failed');mp.write_text(json.dumps(m,indent=2))
 m['status']='completed' if all(c['status']=='completed' for c in m['cells']) else 'failed'
except BaseException as e:m.update(status='failed',error=repr(e));raise
finally:mp.write_text(json.dumps(m,indent=2))
