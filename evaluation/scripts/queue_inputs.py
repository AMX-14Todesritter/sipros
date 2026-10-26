"""Prepare family-preserving resource probes and deterministic correctness subsets."""
import argparse,csv,hashlib,itertools,json,re
from pathlib import Path
from inventory import ROOT,OUT,sha,inspect
from scaling_inputs import fasta_records,ident
from prepare import records
from run import save
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
w=json.loads((ROOT/'evaluation/configs/workloads.json').read_text())['workloads'][a.dataset]
folder=OUT/'inputs/pilots_x1_fixed_v2'/a.dataset;folder.mkdir(parents=True,exist_ok=False)
source=ROOT/w['fasta'];canon=lambda s:re.sub(r'^(?:Rev_[12]_|TestRev_|rev_)','',s)
families=set()
for rec in fasta_records(source):
 families.add(canon(ident(rec)))
 if len(families)>=128:break
entries=0
with (folder/'resource.fasta').open('wb') as f:
 for rec in fasta_records(source):
  if canon(ident(rec)) in families:f.write(rec);entries+=1
spectrum=ROOT/w['ft2'];header=[]
with spectrum.open('rb') as f:
 for line in f:
  if line.startswith(b'S\t') or line.startswith(b'S '):break
  header.append(line)
with (folder/'spectra256.FT2').open('wb') as f,(folder/'spectrum_map.csv').open('w') as mapping:
 f.writelines(header);writer=csv.writer(mapping);writer.writerow(['subset_index','source','scan_id','record_sha256'])
 for i,rec in enumerate(itertools.islice(records(spectrum),256)):
  f.write(rec);writer.writerow([i,str(spectrum),rec.splitlines()[0].split()[1].decode(),hashlib.sha256(rec).hexdigest()])
save(folder/'manifest.json',{'status':'completed','scope':'resource FASTA family probe; prefix FT2 correctness subset, not main workload','family_count':len(families),'fasta_entries':entries,'source_fasta_sha256':sha(source),'source_ft2_sha256':sha(spectrum),'resource_fasta_sha256':sha(folder/'resource.fasta'),'subset':inspect(folder/'spectra256.FT2','ft2')})
print(folder,flush=True)
