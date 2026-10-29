"""Nested X1/X2/X3, fixed Marine, multi-hypothesis, cache-off scaling study."""
import csv,datetime,fcntl,hashlib,json,re,subprocess,sys,statistics,shutil
from pathlib import Path
from prepare import records,recover
from inventory import sha,inspect
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';D=E/'scan_scaling_marine_nested_multi_4m_sm120';MP=D/'manifest.json'
def save(p,m):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(m,indent=2));t.replace(p)
def report(m):
 lines=['# Marine nested scan scaling','',f"Status: {m['status']}; selected batch: {m.get('selected_batch','capacity check pending')}",'','Multi-hypothesis; theory cache off; CUDA and RT-custom; three complete searches per cell; capacity checks count when successful. All timings are unprofiled. Memory is sampled device-wide, including desktop.','']
 lines+=['|Sample|Backend|Completed|Fused median (s)|Complete scoring median (s)|Search median (s)|GPU peak maximum (GiB)|','|---|---|---:|---:|---:|---:|---:|']
 rows=[]
 for sample in ('S1','S2','S3'):
  for backend in ('cuda','rt-custom'):
   js=[j for j in m['jobs'] if j['sample']==sample and j['backend']==backend and j['status']=='completed' and j['batch']==m.get('selected_batch')]
   row={'sample':sample,'backend':backend,'completed':len(js)}
   for key in ('fused_seconds','complete_scoring_seconds','search_seconds','device_peak_bytes','host_peak_bytes'):
    vs=[j['metrics'][key] for j in js]
    if vs:row.update({key+'_median':statistics.median(vs),key+'_mean':statistics.mean(vs),key+'_min':min(vs),key+'_max':max(vs),key+'_stdev':statistics.stdev(vs) if len(vs)>1 else 0})
   rows.append(row)
   val=lambda k:f"{row[k]:.3f}" if k in row else '—'
   lines.append(f"|{sample}|{backend}|{len(js)}/3|{val('fused_seconds_median')}|{val('complete_scoring_seconds_median')}|{val('search_seconds_median')}|{max([j['metrics']['device_peak_bytes']/2**30 for j in js],default=0):.2f}|")
 (D/'progress.md').write_text('\n'.join(lines)+'\n')
 with (D/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def prepare():
 D.mkdir(exist_ok=False)
 original=json.loads((E/'inputs/nested_x1_x2_x3/manifest.json').read_text())
 provenance=json.loads((E/'batch_studies/marine_x1_multi_sm120_b2m_to10m_5r_20261028/manifest.json').read_text())
 build=E/'build/gpu_multi_sm120_batch_20261028';binary=build/'bin/sipros_mvh_cuda'
 assert sha(binary)==provenance['fingerprints']['gpu_binary']
 cache=(build/'CMakeCache.txt').read_text();assert re.search(r'^CMAKE_CUDA_ARCHITECTURES[^=]*=120$',cache,re.M)
 ninja=(build/'build.ninja').read_text();assert 'code=[compute_120,sm_120]' in ninja
 assert 'constexpr bool useIonCache=false;' in (build/'source/mvh_cuda/cuda/engine.cu').read_text()
 database=R/'raw/Marine_fw_3rev.fasta';assert sha(database)==provenance['fingerprints']['fasta']
 for sample in original['samples'].values():assert sha(Path(sample['path']))==sample['sha256']
 m={'status':'ready','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'binary':str(binary),'binary_sha256':sha(binary),'native_architecture':'sm_120','optix_ptx_targets':{p.name:re.search(r'^\.target (.+)$',p.read_text(),re.M).group(1) for p in build.glob('gpu_rt*.ptx')},'precursor':'multi-hypothesis','ion_cache':'compiled_off','database':str(database),'database_sha256':sha(database),'samples':original['samples'],'sources':original['sources'],'jobs':[],'selected_batch':4000000,'repeats':3,'capacity_reserve_gib':1.5,'host_rss_guard_gib':16,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 save(MP,m);return m
def measure(m,sample,backend,batch,repeat,phase):
 rid=f'marine_nested_sm120_{sample}_multi_nocache_b{batch}_{backend}_r{repeat}'
 scans=Path(m['samples'][sample]['path']);binary=Path(m['binary']);config=R/'experiments/Regular.cfg'
 parameters={'category':'scan_scaling','scans':str(scans),'fasta':m['database'],'config':str(config),'gpu_binary':str(binary),'backends':[backend],'batch':batch,'repeats':1,'warmups':0,'threads':1}
 pause=E/'manifests/LARGE_BENCHMARKS_PAUSED.json';paused=json.loads(pause.read_text());paused.setdefault('authorized_runs',{})[rid]={'parameters':parameters,'study':str(MP),'request':'User authorizes nested X1/X2/X3 Marine study, S3 capacity first, three repetitions, CUDA and RT-custom'};save(pause,paused)
 cmd=[sys.executable,'-B',str(R/'evaluation/scripts/run.py'),'--id',rid,'--category','scan_scaling','--scans',str(scans),'--fasta',m['database'],'--config',str(config),'--backends',backend,'--batch',str(batch),'--repeats','1','--warmups','0','--threads','1','--gpu-binary',str(binary),'--min-device-free-gib','1.5','--max-host-rss-gib','16']
 j={'id':rid,'sample':sample,'backend':backend,'batch':batch,'repeat':repeat,'phase':phase,'status':'running','command':cmd};m['jobs'].append(j);save(MP,m);report(m);print('START',rid,flush=True)
 with (D/(rid+'.log')).open('w') as log:p=subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT)
 path=E/'results/scan_scaling'/rid/'manifest.json';j['result_manifest']=str(path);n=json.loads(path.read_text()) if path.exists() else {};runs=n.get('runs',[])
 if p.returncode or not runs or runs[0]['status']!='completed':
  j.update(status='failed',exit_code=p.returncode)
  r=runs[0] if runs else {};text=(path.parent/(r.get('name',backend+'_r1')+'.log')).read_text(errors='replace') if path.exists() and (path.parent/(r.get('name',backend+'_r1')+'.log')).exists() else ''
  j['capacity_failure']=bool(r.get('resource_guard')) or any(v in text.lower() for v in ('out of memory','cudaerrormemoryallocation','optix_error_memory_allocation_failed','std::bad_alloc'))
  save(MP,m);report(m);print('FAILED',rid,flush=True)
  if not j['capacity_failure']:raise RuntimeError('Non-capacity failure; inspect preserved logs before continuing')
  return False
 r=runs[0];t=r['totals'];assert all(int(b['cached_ions'])==0 and int(b['cache_charge_stride'])==0 for b in r['batches'])
 assert r['generated_peptides']==464495485
 counts=m.setdefault('candidate_counts',{});c=int(t['candidates']);assert counts.setdefault(sample,c)==c
 j.update(status='completed',metrics={'fused_seconds':t['fused_device_seconds'],'rt_setup_seconds':r['rt_setup_seconds'],'complete_scoring_seconds':sum(t[k] for k in ('pack_seconds','gpu_service_seconds','replay_verify_seconds')),'search_seconds':float(r['summary']['search_seconds']),'device_peak_bytes':r['device_peak_bytes'],'host_peak_bytes':r['host_observed_hwm_bytes'],'candidates':c,'psm_sha256':r['psm_sha256']})
 save(MP,m);report(m);print('DONE',rid,j['metrics'],flush=True);return True
def main():
 lock=(E/'scan_scaling_marine_nested_multi_4m_sm120.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 m=prepare();disabled=set()
 try:
  m['status']='capacity_check';save(MP,m);report(m)
  for backend in ('cuda','rt-custom'):
   if not measure(m,'S3',backend,4000000,1,'capacity'):disabled.add(('S3',backend))
  m['status']='running';save(MP,m);report(m)
  for repeat in range(1,4):
   order=['S1','S2','S3'];shift=(repeat-1)%3;order=order[shift:]+order[:shift]
   backends=['cuda','rt-custom'];shift=(repeat-1)%2;backends=backends[shift:]+backends[:shift]
   for sample in order:
    for backend in backends:
     if repeat==1 and sample=='S3':continue
     if (sample,backend) in disabled:
      m.setdefault('skipped',[]).append({'sample':sample,'backend':backend,'repeat':repeat,'reason':'This cell failed capacity check; no automatic batch reduction'});continue
     if not measure(m,sample,backend,4000000,repeat,'formal'):disabled.add((sample,backend))
  m['status']='completed_with_capacity_failures' if disabled else 'completed'
 except BaseException as error:m.update(status='stopped_with_error',error=repr(error));raise
 finally:save(MP,m);report(m)
if __name__=='__main__':main()
