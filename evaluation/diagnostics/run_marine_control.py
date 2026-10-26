"""Small Marine multi-precursor/cache control; never starts the large queue."""
import os,sys,json,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'evaluation/scripts'))
from scaling_inputs import fasta_records,ident,family_id
D=ROOT/'output/evaluation/diagnostics/marine_control';D.mkdir(parents=True,exist_ok=False)
source=ROOT/'raw/Marine_fw_3rev.fasta';selected=set()
for rec in fasta_records(source):
 name=ident(rec)
 if family_id(name)==name:selected.add(name)
 if len(selected)==2048:break
fasta=D/'marine_2048families.fasta';entries=0
with fasta.open('wb') as f:
 for rec in fasta_records(source):
  if family_id(ident(rec)) in selected:f.write(rec);entries+=1
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
scans=ROOT/'output/evaluation/inputs/ecoli_x1_first256/spectra.FT2'
m={'status':'running','families':len(selected),'entries':entries,'fasta_sha256':sha(fasta),'scans_sha256':sha(scans),'mode':'multi','batch':2000000,'warmups_per_batch':3,'measured_replays_per_batch':7,'cells':[]}
mp=D/'manifest.json'
for cache in ('auto','disabled'):
 env=os.environ.copy();env['SIPROS_DIAGNOSTIC_HOT_REPLAY']='1';env['SIPROS_DIAGNOSTIC_MULTI_PRECURSOR']='1'
 if cache=='disabled':env['SIPROS_DIAGNOSTIC_NO_ION_CACHE']='1'
 else:env.pop('SIPROS_DIAGNOSTIC_NO_ION_CACHE',None)
 rid='marine_control_multi_'+cache
 cmd=[sys.executable,'-B',str(ROOT/'evaluation/scripts/run.py'),'--id',rid,'--category','accuracy','--scans',str(scans),'--fasta',str(fasta),'--backends','cuda','rt-instanced','rt-custom','--batch','2000000','--repeats','1','--warmups','0','--gpu-binary',str(ROOT/'output/evaluation/diagnostics/hot_build/bin/sipros_mvh_cuda'),'--continue-on-backend-failure']
 item={'cache':cache,'id':rid,'command':cmd,'status':'running'};m['cells'].append(item);mp.write_text(json.dumps(m,indent=2))
 with (D/(cache+'.log')).open('w') as log:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 item.update(status='completed' if r.returncode==0 else 'failed',exit_code=r.returncode);mp.write_text(json.dumps(m,indent=2))
m['status']='completed' if all(x['status']=='completed' for x in m['cells']) else 'failed';mp.write_text(json.dumps(m,indent=2))
print(json.dumps(m,indent=2))
