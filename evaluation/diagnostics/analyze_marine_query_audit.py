"""Compare canonical candidate keys, per-candidate double query fingerprints and peak audits."""
import json,csv,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';D=E/'diagnostics/marine_query_audit'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=E/'results/accuracy/marine_query_audit_auto/cuda_r1/peak_audit'
reference={p.name:list(csv.DictReader(p.open(),delimiter='\t')) for p in sorted(base.glob('queries_*.tsv'))}
rows=[];checks=[]
for cache in ('auto','disabled'):
 path=E/'results/accuracy'/('marine_query_audit_'+cache);m=json.loads((path/'manifest.json').read_text())
 normal=json.loads((E/'results/accuracy'/('marine_service_1_'+cache)/'manifest.json').read_text())
 for run in m['runs']:
  folder=path/run['name']/'peak_audit';sums={};canonical_bad=0;query_bad=0;physical=0;examples=[]
  with (folder/'peak_audit_batches.tsv').open() as f:
   for b in csv.DictReader(f,delimiter='\t'):
    for k,v in b.items():
     if k!='batch':sums[k]=sums.get(k,0)+int(v)
  for name,a in reference.items():
   p=folder/name;b=list(csv.DictReader(p.open(),delimiter='\t'));assert len(a)==len(b)
   for i,(x,y) in enumerate(zip(a,b)):
    key=('index','scan_id','charge','precursor_mass','sequence');q=('query_count','hash1','hash2')
    kd=any(x[k]!=y[k] for k in key);qd=any(x[k]!=y[k] for k in q)
    canonical_bad+=kd;query_bad+=qd;physical+=int(y['query_count'])
    if (kd or qd) and len(examples)<20:examples.append({'file':name,'index':i,'reference':x,'actual':y})
  spectra_equal=all(sha(folder/p.name)==sha(p) for p in base.glob('spectra_*.tsv'))
  native=next(x for x in normal['runs'] if x['backend']==run['backend'])
  rows.append({'cache':cache,'backend':run['backend'],'candidates':sum(len(x) for x in reference.values()),'canonical_candidate_differences':canonical_bad,'query_fingerprint_differences':query_bad,'physical_queries':physical,'preprocessed_spectra_identical':spectra_equal,'audit_psm_matches_native':run['psm_sha256']==native['psm_sha256'],**sums})
  checks.append({'cache':cache,'backend':run['backend'],'query_files_sha256':{p.name:sha(p) for p in folder.glob('queries_*.tsv')},'examples':examples})
with (D/'comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(D/'comparison.json').write_text(json.dumps({'rows':rows,'file_checks':checks,'scope':'all candidates, all in-range physical query sequence double-bit fingerprints (two 64-bit hashes per candidate), not a stored per-query dump; diagnostic times excluded'},indent=2))
print(json.dumps(rows,indent=2))
