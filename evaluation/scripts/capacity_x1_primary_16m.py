"""Single-precursor Marine/X1 16M capacity check; CUDA and RT-custom only."""
import datetime, hashlib, json, subprocess, sys
from pathlib import Path
R=Path(__file__).resolve().parents[2]
E=R/'output/evaluation'
D=E/'capacity_marine_x1_primary_16m'
def save(p,m):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(m,indent=2));t.replace(p)
def main():
 D.mkdir(exist_ok=False)
 baseline=json.loads((E/'batch_sensitivity_marine_x1/manifest.json').read_text())
 binary=E/'batch_sensitivity_marine_x1/build/bin/sipros_mvh_cuda'
 assert hashlib.sha256(binary.read_bytes()).hexdigest()==baseline['binary_sha256']
 scans=R/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2';fasta=R/'raw/Marine_fw_3rev.fasta';config=R/'experiments/Regular.cfg'
 for p in (scans,fasta,config):assert hashlib.sha256(p.read_bytes()).hexdigest()==baseline['inputs'][str(p)]
 m={'status':'running','precursor':'primary-only','ion_cache':'compiled_off','batch':16000000,'repeats':1,'warmups':0,'profiling':False,'binary_sha256':baseline['binary_sha256'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[]}
 pause=E/'manifests/LARGE_BENCHMARKS_PAUSED.json';paused=json.loads(pause.read_text())
 for backend in ('cuda','rt-custom'):
  rid=f'marine_x1_primary_nocache_b16000000_{backend}_r1'
  parameters={'category':'batch_size','scans':str(scans),'fasta':str(fasta),'config':str(config),'gpu_binary':str(binary),'backends':[backend],'batch':16000000,'repeats':1,'warmups':0,'threads':1}
  paused.setdefault('authorized_runs',{})[rid]={'parameters':parameters,'study':str(D/'manifest.json'),'request':'User requests 16M single-precursor capacity test, excluding RT-instanced'}
  cmd=[sys.executable,'-B',str(R/'evaluation/scripts/run.py'),'--id',rid,'--category','batch_size','--scans',str(scans),'--fasta',str(fasta),'--config',str(config),'--backends',backend,'--batch','16000000','--repeats','1','--warmups','0','--threads','1','--gpu-binary',str(binary)]
  m['jobs'].append({'id':rid,'backend':backend,'status':'pending','command':cmd})
 save(pause,paused);save(D/'manifest.json',m)
 try:
  for j in m['jobs']:
   j['status']='running';save(D/'manifest.json',m);print('START',j['id'],flush=True)
   with (D/(j['backend']+'.log')).open('w') as log:p=subprocess.run(j['command'],cwd=R,stdout=log,stderr=subprocess.STDOUT)
   result=E/'results/batch_size'/j['id']/'manifest.json';j['result_manifest']=str(result)
   n=json.loads(result.read_text()) if result.exists() else {};runs=n.get('runs',[])
   if p.returncode or not runs or runs[0]['status']!='completed':
    j.update(status='failed',exit_code=p.returncode);save(D/'manifest.json',m)
    raise RuntimeError('Capacity measurement failed; stop further capacity tests and preserve logs')
   r=runs[0];t=r['totals']
   assert all(int(b['cached_ions'])==0 and int(b['cache_charge_stride'])==0 for b in r['batches'])
   assert {'generated':r['generated_peptides'],'candidates':int(t['candidates'])}==baseline['workload_counts']
   j.update(status='completed',metrics={'fused_seconds':t['fused_device_seconds'],'complete_scoring_seconds':sum(t[k] for k in ('pack_seconds','gpu_service_seconds','replay_verify_seconds')),'search_seconds':float(r['summary']['search_seconds']),'device_peak_bytes':r['device_peak_bytes'],'host_peak_bytes':r['host_observed_hwm_bytes'],'psm_sha256':r['psm_sha256']})
   save(D/'manifest.json',m);print('DONE',j['id'],j['metrics'],flush=True)
  m['status']='completed'
 except BaseException as e:m.update(status='stopped_with_error',error=repr(e));raise
 finally:save(D/'manifest.json',m)
if __name__=='__main__':main()
