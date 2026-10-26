"""Run bounded continuous replay and interleaved fresh-process service controls."""
import os,subprocess,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'output/evaluation/diagnostics/continuous_control';records=[]
base=[sys.executable,'-B',str(R/'evaluation/scripts/run.py'),'--category','accuracy','--scans',str(R/'output/evaluation/inputs/ecoli_x1_first256/spectra.FT2'),'--fasta',str(R/'output/evaluation/diagnostics/marine_control/marine_2048families.fasta'),'--batch','2000000','--repeats','1','--warmups','0']
for i in range(4):
 for cache in (('auto','disabled') if i%2==0 else ('disabled','auto')):
  env=os.environ.copy();env['SIPROS_DIAGNOSTIC_MULTI_PRECURSOR']='1';env.pop('SIPROS_DIAGNOSTIC_HOT_REPLAY',None)
  if i==0:
   if cache=='disabled':continue
   env['SIPROS_CONTINUOUS_REPLAY']='1';binary=D/'build/bin/sipros_mvh_cuda';rid='marine_continuous_replay'
  else:
   env.pop('SIPROS_CONTINUOUS_REPLAY',None);binary=R/'output/evaluation/diagnostics/build/bin/sipros_mvh_cuda';rid=f'marine_service_{i}_{cache}'
  if cache=='disabled':env['SIPROS_DIAGNOSTIC_NO_ION_CACHE']='1'
  else:env.pop('SIPROS_DIAGNOSTIC_NO_ION_CACHE',None)
  backends=['cuda','rt-instanced','rt-custom'];backends=backends[i%3:]+backends[:i%3]
  cmd=base+['--id',rid,'--backends',*backends,'--gpu-binary',str(binary)]
  print(rid,flush=True)
  with (D/(rid+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
  records.append({'id':rid,'cache':cache,'command':cmd,'exit_code':r.returncode});(D/'runs.json').write_text(json.dumps(records,indent=2))
  if r.returncode:raise RuntimeError(rid)
