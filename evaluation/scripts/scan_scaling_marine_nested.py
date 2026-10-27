"""Nested X1/X2/X3, fixed Marine, primary-only, cache-off scaling study."""
import csv,datetime,fcntl,hashlib,json,re,subprocess,sys,statistics,shutil
from pathlib import Path
from prepare import records,recover
from inventory import sha,inspect
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';D=E/'scan_scaling_marine_nested';MP=D/'manifest.json'
def save(p,m):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(m,indent=2));t.replace(p)
def report(m):
 lines=['# Marine nested scan scaling','',f"Status: {m['status']}; selected batch: {m.get('selected_batch','capacity check pending')}",'','Primary-only; theory cache off; CUDA and RT-custom; seven complete searches per cell; capacity checks count when successful. All timings are unprofiled. Memory is sampled device-wide, including desktop.','']
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
   lines.append(f"|{sample}|{backend}|{len(js)}/7|{val('fused_seconds_median')}|{val('complete_scoring_seconds_median')}|{val('search_seconds_median')}|{max([j['metrics']['device_peak_bytes']/2**30 for j in js],default=0):.2f}|")
 (D/'progress.md').write_text('\n'.join(lines)+'\n')
 with (D/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def prepare():
 D.mkdir(exist_ok=False);inp=D/'inputs';inp.mkdir()
 cfg=json.loads((R/'evaluation/configs/workloads.json').read_text())['workloads'];sources=[R/cfg[k]['ft2'] for k in ('ecoli_x1','ecoli_x2','ecoli_x3')]
 missing=[k for k in ('ecoli_x2','ecoli_x3') if not (R/cfg[k]['ft2']).is_file()]
 if missing:
  (E/'tmp').mkdir(exist_ok=True);recover(missing)
 baseline=json.loads((E/'batch_sensitivity_marine_x1/manifest.json').read_text());binary=E/'batch_sensitivity_marine_x1/build/bin/sipros_mvh_cuda'
 assert sha(binary)==baseline['binary_sha256']
 m={'status':'preparing','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'binary':str(binary),'binary_sha256':sha(binary),'precursor':'primary-only','ion_cache':'compiled_off','database':str(R/'raw/Marine_fw_3rev.fasta'),'database_sha256':sha(R/'raw/Marine_fw_3rev.fasta'),'samples':{},'sources':[],'jobs':[],'capacity_reserve_gib':1.5,'host_rss_guard_gib':16,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 assert m['database_sha256']==baseline['inputs'][str(R/'raw/Marine_fw_3rev.fasta')]
 save(MP,m);header=[]
 with sources[0].open('rb') as f:
  for line in f:
   if line.startswith((b'S\t',b'S ')):break
   header.append(line)
 previous=None;gid=0
 for number,src in enumerate(sources,1):
  out=inp/f'S{number}.FT2';mapping=inp/f'S{number}_map.csv'
  if previous:
   shutil.copyfile(previous,out);shutil.copyfile(inp/f'S{number-1}_map.csv',mapping)
  else:
   out.write_bytes(b''.join(header));mapping.write_text('global_scan_id,source,original_scan_id,original_record_sha256,remapped_record_sha256\n')
  seen=set();n=0
  with out.open('ab') as f,mapping.open('a') as mapfile:
   writer=csv.writer(mapfile)
   for rec in records(src):
    sid=rec.splitlines()[0].split()[1].decode();assert sid not in seen;seen.add(sid);gid+=1;n+=1
    remap=re.sub(rb'^(S[ \t]+)\S+',lambda match:match.group(1)+str(gid).encode(),rec,count=1)
    assert re.sub(rb'^(S[ \t]+)\S+',b'S ID',rec,count=1)==re.sub(rb'^(S[ \t]+)\S+',b'S ID',remap,count=1)
    f.write(remap);writer.writerow([gid,str(src),sid,hashlib.sha256(rec).hexdigest(),hashlib.sha256(remap).hexdigest()])
  assert n==cfg[f'ecoli_x{number}']['expected_scans']
  info=inspect(out,'ft2');assert info['scans']==gid and info['duplicate_scan_ids']==0 and info['invalid_primary_charge_scans']==0
  if previous:
   with previous.open('rb') as a,out.open('rb') as b:
    while block:=a.read(1024*1024):assert block==b.read(len(block))
  m['sources'].append({'path':str(src),'sha256':sha(src),'scans':n});m['samples'][f'S{number}']={'path':str(out),'scan_count':gid,'sha256':sha(out),'map':str(mapping),'inspection':info};save(MP,m);previous=out
 return m
def measure(m,sample,backend,batch,repeat,phase):
 rid=f'marine_nested_{sample}_primary_nocache_b{batch}_{backend}_r{repeat}'
 scans=Path(m['samples'][sample]['path']);binary=Path(m['binary']);config=R/'experiments/Regular.cfg'
 parameters={'category':'scan_scaling','scans':str(scans),'fasta':m['database'],'config':str(config),'gpu_binary':str(binary),'backends':[backend],'batch':batch,'repeats':1,'warmups':0,'threads':1}
 pause=E/'manifests/LARGE_BENCHMARKS_PAUSED.json';paused=json.loads(pause.read_text());paused.setdefault('authorized_runs',{})[rid]={'parameters':parameters,'study':str(MP),'request':'User authorizes nested X1/X2/X3 Marine study, S3 capacity first, seven repetitions, no RT-instanced'};save(pause,paused)
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
 lock=(E/'scan_scaling_marine_nested.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if MP.exists():
  m=json.loads(MP.read_text());assert not m['jobs'] and len(m['samples'])==3
  for sample in m['samples'].values():assert sha(Path(sample['path']))==sample['sha256']
  m.pop('error',None)
 else:m=prepare()
 try:
  m['status']='capacity_check';save(MP,m);report(m)
  for batch in (8000000,4000000):
   ok=True
   for backend in ('cuda','rt-custom'):
    if not measure(m,'S3',backend,batch,1,'capacity'):ok=False;break
   if ok:m['selected_batch']=batch;break
  if 'selected_batch' not in m:raise RuntimeError('S3 capacity check failed at both batches; no formal measurements started')
  m['status']='running';save(MP,m);report(m)
  for repeat in range(1,8):
   order=['S1','S2','S3'];shift=(repeat-1)%3;order=order[shift:]+order[:shift]
   backends=['cuda','rt-custom'] if repeat%2 else ['rt-custom','cuda']
   for sample in order:
    for backend in backends:
     if repeat==1 and sample=='S3':continue
     if not measure(m,sample,backend,m['selected_batch'],repeat,'formal'):raise RuntimeError('Formal capacity failure; stop without silently changing batch')
  m['status']='completed'
 except BaseException as error:m.update(status='stopped_with_error',error=repr(error));raise
 finally:save(MP,m);report(m)
if __name__=='__main__':main()
