"""Bounded 256-spectrum precursor/cache diagnostic; never resumes large benchmarks."""
import json,os,subprocess,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/evaluation/diagnostics';binary=OUT/'build/bin/sipros_mvh_cuda'
path=OUT/'reports/controls_manifest.json'
if path.exists():raise FileExistsError('Preserve existing controlled diagnostic')
meta={'status':'running','scope':'256 fixed X1 spectrum records + full small Ecoli FASTA; not a Marine full-workload reproduction','batch_generated_peptide_entries':2000000,'repeats':5,'warmups':1,'cells':[],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
path.write_text(json.dumps(meta,indent=2))
try:
 for precursor,cache in [('primary','auto'),('multi','auto'),('primary','disabled'),('multi','disabled')]:
  env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1')
  if precursor=='multi':env['SIPROS_DIAGNOSTIC_MULTI_PRECURSOR']='1'
  else:env.pop('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR',None)
  if cache=='disabled':env['SIPROS_DIAGNOSTIC_NO_ION_CACHE']='1'
  else:env.pop('SIPROS_DIAGNOSTIC_NO_ION_CACHE',None)
  rid='diagnostic_256_'+precursor+'_'+cache
  cmd=[sys.executable,'-B',str(ROOT/'evaluation/scripts/run.py'),'--id',rid,'--category','accuracy','--scans',str(ROOT/'output/evaluation/inputs/ecoli_x1_first256/spectra.FT2'),'--fasta',str(ROOT/'raw/Ecoli.fasta'),'--backends','cuda','rt-instanced','rt-custom','--batch','2000000','--repeats','5','--warmups','1','--gpu-binary',str(binary),'--max-host-rss-gib','18','--continue-on-backend-failure']
  cell={'precursor_mode':precursor,'cache_policy':cache,'command':cmd,'environment_controls':{k:v for k,v in env.items() if k.startswith('SIPROS_DIAGNOSTIC_')},'status':'running'};meta['cells'].append(cell);path.write_text(json.dumps(meta,indent=2))
  print('CONTROL '+precursor+' '+cache,flush=True)
  with (OUT/'results'/(rid+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  cell.update(status='completed' if r.returncode==0 else 'failed',exit_code=r.returncode,run_manifest=str(ROOT/'output/evaluation/results/accuracy'/rid/'manifest.json'));path.write_text(json.dumps(meta,indent=2))
 meta['status']='completed' if all(x['status']=='completed' for x in meta['cells']) else 'failed'
except BaseException as e:meta.update(status='failed',error=repr(e));raise
finally:meta['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();path.write_text(json.dumps(meta,indent=2))
