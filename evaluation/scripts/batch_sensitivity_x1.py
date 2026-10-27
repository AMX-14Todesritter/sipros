"""Explicitly authorized Marine/X1 single-precursor no-cache 4x3x3 serial study."""
import os,sys,json,subprocess,hashlib,shutil,datetime,statistics,csv,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';D=E/'batch_sensitivity_marine_x1';MP=D/'manifest.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,m):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(m,indent=2));t.replace(p)
def prepare():
 D.mkdir(exist_ok=False);S=D/'source';S.mkdir();B=D/'build'
 for name in ('mvh_cuda','MVH_RT'):shutil.copytree(R/name,S/name,ignore=shutil.ignore_patterns('__pycache__'))
 for name in ('MSToolkit','shared'):(S/name).symlink_to(R/name,target_is_directory=True)
 s=(S/'mvh_cuda/cuda/engine.cu').read_text();assert s.count('constexpr bool useIonCache=false;')==1, 'Cache-off baseline required; rebuild from current source'
 protected=['mvh_cuda/cuda/scoring.cuh','mvh_cuda/cuda/theoretical.cuh','MVH_RT/gpu_bridge/device.cu','MVH_RT/gpu_bridge/custom_device.cu','MVH_RT/gpu_bridge/custom_geometry.cu','mvh_cuda/src/spectrum_input.cpp']
 assert all(sha(R/f)==sha(S/f) for f in protected)
 cmds=[['cmake','-S',str(S/'mvh_cuda'),'-B',str(B),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DMVH_CUDA_ENABLE_RT=ON','-DMVH_ENABLE_EVALUATION_TIMING=ON','-DMVH_ENABLE_FLOW_COUNTERS=OFF','-DMVH_ENABLE_PROFILING=OFF','-DOPTIX_ROOT='+str(R/'build/mvh_rt/optix_sdk')],['cmake','--build',str(B),'--target','sipros_mvh_cuda','-j','4']]
 m={'status':'building','authorization':'User explicitly starts Marine full database + X1 batch sensitivity, cache off; other suites remain paused','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'precursor':'primary-only','ion_cache':'compiled_off','batch_unit':'generated peptide entries','scan_source':'Pan X1 full','repeats':3,'warmups':0,'profiling':False,'build_commands':cmds,'protected_sha256':{f:sha(S/f) for f in protected},'jobs':[]};save(MP,m)
 for i,cmd in enumerate(cmds):
  with (D/f'build_{i}.log').open('w') as log:p=subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:m.update(status='build_failed',exit_code=p.returncode);save(MP,m);raise RuntimeError('Build failed; preserved logs')
 binary=B/'bin/sipros_mvh_cuda';scans=R/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2';fasta=R/'raw/Marine_fw_3rev.fasta';config=R/'experiments/Regular.cfg'
 m.update(binary_sha256=sha(binary),inputs={str(p):sha(p) for p in (scans,fasta,config)})
 backend_orders=[['cuda','rt-instanced','rt-custom'],['rt-instanced','rt-custom','cuda'],['rt-custom','cuda','rt-instanced']];batches=[[2000000,4000000,8000000,12000000],[12000000,8000000,4000000,2000000],[8000000,12000000,2000000,4000000]]
 pause=E/'manifests/LARGE_BENCHMARKS_PAUSED.json';paused=json.loads(pause.read_text());allow=paused.setdefault('authorized_runs',{})
 for round in range(3):
  for batch in batches[round]:
   for backend in backend_orders[round]:
    rid=f'marine_x1_nocache_b{batch}_{backend}_r{round+1}'
    cmd=[sys.executable,'-B',str(R/'evaluation/scripts/run.py'),'--id',rid,'--category','batch_size','--scans',str(scans),'--fasta',str(fasta),'--config',str(config),'--backends',backend,'--batch',str(batch),'--repeats','1','--warmups','0','--threads','1','--gpu-binary',str(binary)]
    parameters={'category':'batch_size','scans':str(scans),'fasta':str(fasta),'config':str(config),'gpu_binary':str(binary),'backends':[backend],'batch':batch,'repeats':1,'warmups':0,'threads':1}
    allow[rid]={'parameters':parameters,'study':str(MP),'request':'Run the Marine/X1 batch-size experiment with the theoretical-ion cache disabled'}
    m['jobs'].append({'id':rid,'round':round+1,'batch':batch,'backend':backend,'status':'pending','command':cmd})
 paused['scoped_resumption']='Only the enumerated Marine/X1 batch study IDs are authorized; all other suites remain paused';save(pause,paused);m['status']='ready';save(MP,m);report(m);print(str(MP),flush=True)
def report(m):
 rows=[]
 for batch in (2000000,4000000,8000000,12000000):
  for backend in ('cuda','rt-instanced','rt-custom'):
   jobs=[j for j in m['jobs'] if j['batch']==batch and j['backend']==backend];complete=[j for j in jobs if j['status']=='completed'];row={'batch':batch,'backend':backend,'completed_repeats':len(complete),'failed':sum(j['status']=='failed' for j in jobs)}
   for metric in ('fused_seconds','rt_setup_seconds','complete_scoring_seconds','search_seconds','host_peak_bytes','device_wide_peak_bytes'):
    values=[j['metrics'][metric] for j in complete]
    if values:row.update({metric+'_median':statistics.median(values),metric+'_min':min(values),metric+'_max':max(values)})
   if complete:row['repeat_psm_identical']=len({j['metrics']['psm_sha256'] for j in complete})==1
   rows.append(row)
 with (D/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
 lines=['# Full Marine database + X1: batch-size sensitivity','',f"Queue status: {m['status']}. Completed {sum(j['status']=='completed' for j in m['jobs'])}/36; failed {sum(j['status']=='failed' for j in m['jobs'])}.",'','Use all X1 spectra and the full Marine database in primary-only mode. The theoretical-ion cache is explicitly disabled at compile time; verify cached_ions=0 and cache_charge_stride=0 for every batch. Peptide compaction is not enabled. Batch size counts generated peptide entries, rather than the assigned peptide entries used by the CPU. Run every configuration once before completing three independent-process repetitions, rotating batch and backend order. No additional warmups or profiling.','','| Batch size | Backend | Completed repetitions | Median fused scoring (s) | Median RT setup (s) | Median complete scoring (s) | Median search (s) |','|---|---|---:|---:|---:|---:|---:|']
 for r in rows:
  val=lambda k:'—' if k+'_median' not in r else f"{r[k+'_median']:.3f}"
  lines.append(f"| {r['batch']:,} | {r['backend']} | {r['completed_repeats']}/3 | {val('fused_seconds')} | {val('rt_setup_seconds')} | {val('complete_scoring_seconds')} | {val('search_seconds')} |")
 lines+=['','Complete scoring is the sum of adjacent host packing, GPU service, and host restoration intervals from the same run. RT setup is already included and must not be added again. Search also includes peptide generation and candidate assignment. Ordinary event intervals include first-launch effects; these are not warm-kernel measurements. See summary.csv for minimum/maximum values, memory observations, and repeated-run PSM checks.','Accuracy differences are deferred to a separate evaluation and do not block performance runs. Preserve PSMs and candidate counts. RT-instanced has known matching discrepancies. Completing a configuration does not establish strict matching equivalence.','A cache-policy or candidate-count violation stops this queue. If one configuration fails, preserve its logs, skip its remaining repetitions, and continue other configurations without changing batch sizes. All unrelated database, scan-count, and Soil queues remain paused.','']
 (D/'progress.md').write_text('\n'.join(lines))
def run():
 m=json.loads(MP.read_text());assert m['status']=='ready';env=os.environ.copy()
 for key in ('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR','SIPROS_DIAGNOSTIC_HOT_REPLAY','SIPROS_CONTINUOUS_REPLAY'):env.pop(key,None)
 lockfile=(D/'coordinator.lock').open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
 m.update(status='running',coordinator_pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(MP,m);failed=set()
 try:
  for job in m['jobs']:
   cell=(job['batch'],job['backend'])
   if cell in failed:job.update(status='skipped',reason='This exact configuration failed earlier; preserve logs, no automatic retry');save(MP,m);report(m);continue
   job['status']='running';save(MP,m);report(m);print('START',job['id'],flush=True)
   with (D/(job['id']+'.log')).open('w') as log:p=subprocess.run(job['command'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
   path=E/'results/batch_size'/job['id']/'manifest.json';n=json.loads(path.read_text()) if path.exists() else {};runs=n.get('runs',[])
   if p.returncode or not runs or runs[0]['status']!='completed':
    job.update(status='failed',exit_code=p.returncode,result_manifest=str(path));save(MP,m);report(m)
    if not runs:raise RuntimeError('Preflight/infrastructure failure; no measurement started')
    failed.add(cell);print('FAILED',job['id'],flush=True);continue
   r=runs[0];t=r['totals'];assert all(int(b['cached_ions'])==0 and int(b['cache_charge_stride'])==0 for b in r['batches']), 'Cache policy violated'
   generated=r['generated_peptides'];candidates=int(t['candidates'])
   expected=m.setdefault('workload_counts',{'generated':generated,'candidates':candidates})
   assert expected=={'generated':generated,'candidates':candidates},'Workload counts changed across configurations'
   job.update(status='completed',result_manifest=str(path),metrics={'fused_seconds':t['fused_device_seconds'],'rt_setup_seconds':r['rt_setup_seconds'],'complete_scoring_seconds':sum(t[k] for k in ('pack_seconds','gpu_service_seconds','replay_verify_seconds')),'search_seconds':float(r['summary']['search_seconds']),'generated':generated,'candidates':candidates,'psm_sha256':r['psm_sha256'],'host_peak_bytes':r.get('host_observed_hwm_bytes',0),'device_wide_peak_bytes':r.get('device_peak_bytes',0)});save(MP,m);report(m);print('DONE',job['id'],flush=True)
  m.update(status='completed' if not failed else 'completed_with_failures',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 except BaseException as error:m.update(status='stopped_with_error',error=repr(error));raise
 finally:save(MP,m);report(m)
if __name__=='__main__':
 if len(sys.argv)!=2 or sys.argv[1] not in ('prepare','run'):raise SystemExit('prepare|run')
 prepare() if sys.argv[1]=='prepare' else run()
