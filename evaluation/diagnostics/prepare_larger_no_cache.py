"""One authorized larger fixed diagnostic workload; does not resume scaling queues."""
import sys,json,hashlib,itertools
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'evaluation/scripts'))
from prepare import pilot
from scaling_inputs import fasta_records,ident,family_id
D=R/'output/evaluation/diagnostics/larger_no_cache';D.mkdir(exist_ok=False)
pilot(4096)
source=R/'raw/Marine_fw_3rev.fasta';families=set()
for rec in fasta_records(source):
 name=ident(rec)
 if family_id(name)==name:families.add(name)
 if len(families)==8192:break
fasta=D/'marine_8192families.fasta';entries=0
with fasta.open('wb') as f:
 for rec in fasta_records(source):
  if family_id(ident(rec)) in families:f.write(rec);entries+=1
scans=R/'output/evaluation/inputs/ecoli_x1_first4096/spectra.FT2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m={'status':'prepared','scans':str(scans),'scans_count':4096,'scans_sha256':sha(scans),'fasta':str(fasta),'fasta_entries':entries,'families':len(families),'fasta_sha256':sha(fasta),'precursor_mode':'multi','cache':'disabled','backends':['cuda','rt-custom'],'batch':2000000,'hot_warmups_per_batch':1,'hot_measurements_per_batch':3,'authorization':'User requested one larger multi-precursor validation, CUDA and RT-custom only; unrelated large-scale suites remain paused.'}
(D/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
