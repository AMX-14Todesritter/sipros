"""Run non-timing per-candidate query and peak audits on the fixed Marine workload."""
import os,subprocess,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'output/evaluation/diagnostics/marine_query_audit';rows=[]
for cache in ('auto','disabled'):
 env=os.environ.copy();env['SIPROS_DIAGNOSTIC_MULTI_PRECURSOR']='1'
 if cache=='disabled':env['SIPROS_DIAGNOSTIC_NO_ION_CACHE']='1'
 else:env.pop('SIPROS_DIAGNOSTIC_NO_ION_CACHE',None)
 rid='marine_query_audit_'+cache
 cmd=[sys.executable,'-B',str(R/'evaluation/scripts/run.py'),'--id',rid,'--category','accuracy','--scans',str(R/'output/evaluation/inputs/ecoli_x1_first256/spectra.FT2'),'--fasta',str(R/'output/evaluation/diagnostics/marine_control/marine_2048families.fasta'),'--backends','cuda','rt-instanced','rt-custom','--batch','2000000','--repeats','1','--warmups','0','--peak-audit','--gpu-binary',str(D/'build/bin/sipros_mvh_cuda')]
 print(rid,flush=True)
 with (D/(cache+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
 rows.append({'id':rid,'cache':cache,'command':cmd,'exit_code':r.returncode});(D/'runs.json').write_text(json.dumps(rows,indent=2))
 mp=R/'output/evaluation/results/accuracy'/rid/'manifest.json'
 if mp.exists():
  m=json.loads(mp.read_text());m['isolated_audit_build']=json.loads((D/'build.json').read_text());mp.write_text(json.dumps(m,indent=2))
 if r.returncode:raise RuntimeError(rid)
