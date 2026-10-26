"""Persistent serial execution plan for the complete agreed evaluation matrix."""
import argparse,datetime,fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
from inventory import ROOT,OUT,sha
from run import save
CFG=json.loads((ROOT/'evaluation/configs/workloads.json').read_text());GPU=CFG['backends'];FIXED=CFG['real_world_workloads'];SCRIPTS=ROOT/'evaluation/scripts'
QPATH=OUT/'manifests/all_experiments.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def case(category,runid):return OUT/'results'/category/runid/'manifest.json'
def jobpath(job):return Path(job['case_manifest']) if job.get('case_manifest') else None

def create_plan():
 if QPATH.exists():raise FileExistsError('Queue exists; use --resume')
 jobs=[]
 def add(key,kind,**kwargs):
  j={'id':key,'kind':kind,'status':'planned','dependencies':[],**kwargs};jobs.append(j);return j
 for name,rid in [('ecoli_x1','main_20261026_ecoli_x1')]:
  path=case('real_world',rid);m=json.loads(path.read_text());assert m['status']=='completed'
  add('real_'+name,'existing',status='completed',case_manifest=str(path),dataset=name,category='real_world',evidence='Existing validated five-repeat result; do not rerun')
 for name in FIXED:
  add('prepare_'+name,'prepare',dataset=name)
 for name in ('human','marine','soil'):
  prefix=OUT/'inputs/pilots_x1_fixed_v2'/name
  add('resource_'+name,'run',dataset=name,category='pilot',run_id='x1v2_resource_'+name,scans=CFG['workloads'][name]['ft2'],fasta=str(prefix/'resource.fasta'),backends=(['cpu'] if name!='soil' else [])+GPU,repeats=1,warmups=0,batch=2000000,dependencies=['prepare_'+name])
  if name!='soil':
   add('real_'+name,'run',dataset=name,category='real_world',run_id='x1v2_real_'+name,scans=CFG['workloads'][name]['ft2'],fasta=CFG['workloads'][name]['fasta'],backends=['cpu']+GPU,repeats=5,warmups=1,batch=2000000,resource='resource_'+name,dependencies=['resource_'+name])
 # Build/capture profiling before long scaling sweeps; no overlap with measured work.
 add('build_profile','build',build_kind='profile')
 for name in ('ecoli_x1','marine'):
  for tool in ('nsys','ncu'):
   for backend in GPU:
    add('profile_'+name+'_'+tool+'_'+backend,'profile',dataset=name,backend=backend,tool=tool,dependencies=['build_profile','real_'+name])
 for batch in CFG['batch_sizes']:
  if batch==2000000:
   add('batch_'+str(batch),'alias',target='real_marine',category='batch_size',parameter=batch,dependencies=['real_marine'])
  else:add('batch_'+str(batch),'run',dataset='marine',category='batch_size',run_id='x1v2_batch_'+str(batch),scans=CFG['workloads']['marine']['ft2'],fasta=CFG['workloads']['marine']['fasta'],backends=GPU,repeats=5,warmups=1,batch=batch,parameter=batch,resource='resource_marine',dependencies=['resource_marine'])
 soil=json.loads((OUT/'inputs/soil_nested_seed20261026/manifest.json').read_text());assert soil['status']=='completed'
 for fraction in ('1','5','10','25','50','100'):
  add('soil_db_'+fraction,'run',dataset='soil',category='database_scaling',run_id='x1v2_soil_db_'+fraction,scans=CFG['workloads']['soil']['ft2'],fasta=soil['subsets'][fraction]['path'],backends=GPU,repeats=5,warmups=1,batch=2000000,parameter=int(fraction),protein_entries=soil['subsets'][fraction]['entries'],resource='resource_soil',dependencies=['resource_soil'])
 add('real_soil','alias',target='soil_db_100',category='real_world',dataset='soil',dependencies=['soil_db_100'],evidence='Same Soil FASTA/FT2, batch2M, five-repeat GPU results reused for DB-1 and DB-2 100%')
 scan=json.loads((OUT/'inputs/scan_scaling_seed20261026/manifest.json').read_text());assert scan['status']=='completed'
 for n,item in sorted(scan['outputs'].items(),key=lambda x:int(x[0])):
  add('resource_scans_'+n,'run',dataset='ecoli_x1',category='pilot',run_id='x1v2_resource_scans_'+n,scans=item['path'],fasta=str(OUT/'inputs/pilots_x1_fixed_v2/ecoli_x1/resource.fasta'),backends=GPU,repeats=1,warmups=0,batch=2000000,dependencies=['prepare_ecoli_x1'])
  add('scan_'+n,'run',dataset='ecoli_x1',category='scan_scaling',run_id='x1v2_scan_'+n,scans=item['path'],fasta=CFG['workloads']['ecoli_x1']['fasta'],backends=GPU,repeats=5,warmups=1,batch=2000000,parameter=int(n),resource='resource_scans_'+n,dependencies=['resource_scans_'+n])
 # Every actual GPU workload gets independent physical-work counters.
 for baseline in list(jobs):
  if baseline.get('category') not in ('real_world','batch_size','database_scaling','scan_scaling') or baseline['kind']=='alias':continue
  if baseline['id']=='real_ecoli_x1':
   add('flow_'+baseline['id'],'existing',baseline=baseline['id'],mode='flow',case_manifest=str(case('accuracy','x1_full_flow_20261026')),status='completed',evidence='Existing matched-counters diagnostic; reused')
  else:add('flow_'+baseline['id'],'derived_run',baseline=baseline['id'],mode='flow',dependencies=[baseline['id']])
 add('boundary_fused','boundary',dependencies=[])
 for name in FIXED:
  # Full fused peak audits; CPU oracle is independent double nearest selection in the diagnostic kernel.
  add('peak_'+name,'derived_run',baseline='real_'+name,mode='peak_audit',dependencies=['real_'+name,'boundary_fused'])
  add('score_'+name,'derived_run',baseline='real_'+name,mode='score_impact',dependencies=['real_'+name])
  if name!='soil':
   for backend in ('cuda','rt-custom'):
    add('verify_'+name+'_'+backend,'run',dataset=name,category='accuracy',run_id='x1v2_verify_'+name+'_'+backend,scans=str(OUT/'inputs/pilots_x1_fixed_v2'/name/'spectra256.FT2'),fasta=CFG['workloads'][name]['fasta'],backends=[backend],repeats=1,warmups=0,batch=2000000,verify=True,dependencies=['prepare_'+name])
 add('final_analysis','analyze')
 # Put demonstrated accuracy gaps and cheap existing-workload counters first.
 def priority(j):
  if j['kind']=='existing':return 0
  if j['kind']=='prepare':return 1
  if j['id']=='boundary_fused':return 2
  if j['id'] in ('peak_ecoli_x1','score_ecoli_x1'):return 3
  if j['id'].startswith('flow_real_ecoli_'):return 4
  return 10
 for j in jobs:
  if j['id']=='boundary_fused':
   artifact=OUT/'results/accuracy/all_fused_boundary/boundary_results.tsv'
   if artifact.is_file():j.update(status='completed',artifact=str(artifact),evidence='Reused unchanged fused boundary diagnostic')
  if j['id'] in ('peak_ecoli_x1','score_ecoli_x1'):
   old=case('accuracy','all_'+j['id'])
   if old.is_file() and json.loads(old.read_text()).get('status')=='completed':j.update(status='completed',case_manifest=str(old),evidence='Valid X1 diagnostic reused under the same scan policy')
  if j['kind']=='profile' and j['dataset']=='ecoli_x1':
   old=OUT/'results/profiling'/('all_'+j['id'])/'profile_manifest.json'
   if old.is_file() and json.loads(old.read_text()).get('status')=='completed':j.update(status='completed',artifact=str(old),evidence='Valid X1 profile reused')
 jobs.sort(key=priority)
 q={'schema_version':1,'status':'ready','created_utc':now(),'jobs':jobs,'policy':{'serial':True,'main_repetitions':5,'warmups':1,'fixed_scaling_batch_generated_peptides':2000000,'cpu_exclusions':['soil real-world','all database scaling','all scan scaling'],'host_rss_guard_gib':18,'minimum_free_disk_gib':32,'per_job_timeout_hours':0,'failed_backend_policy':'preserve failure and continue independent backends/jobs; never change requested batch size','resources':'Use full-spectra family probes; omit only backends with recorded failed probes; each requested main configuration still has its own warmup','reuse':'Existing X1 only; Marine2M also batch B1; Soil100% also real-world Soil','scope':'fused scoring, MVH application timing, fused-path identity correctness; no standalone matching benchmark'},'scan_policy_version':'x1_fixed_v2','fixed_scan_dataset':'Pan X1','scan_scaling_sources':['Pan X1','Pan X2','Pan X3'],'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'frozen_scientific_source':{str(p.relative_to(ROOT)):sha(p) for d in ('mvh','mvh_cuda','MVH_RT','shared') for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.cu','.cuh')},'source_templates':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'evaluation').rglob('*') if p.is_file()},'queue_history':[]}
 save(QPATH,q);return q

def resolve(job,lookup):
 seen=set()
 while job['kind']=='alias':
  if job['id'] in seen:raise RuntimeError('Alias cycle')
  seen.add(job['id']);job=lookup[job['target']]
 return job

def completed_backends(job,lookup):
 j=resolve(job,lookup);path=jobpath(j)
 if path and path.is_file():
  m=json.loads(path.read_text());return list(m.get('statistics',{}))
 return []

def run_command(job,cmd,q,expected=None,extra_env=None):
 logs=OUT/'results/queue';logs.mkdir(parents=True,exist_ok=True)
 job.update(command=cmd,started_utc=now(),status='running');save(QPATH,q)
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(OUT/'dependencies'),MPLCONFIGDIR=str(OUT/'cache/matplotlib'),TMPDIR=str(OUT/'tmp'))
 if extra_env:env.update(extra_env)
 with (logs/(job['id']+'.log')).open('a') as log:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  job['child_pid']=child.pid;save(QPATH,q);start=time.monotonic()
  while True:
   try:code=child.wait(timeout=5);break
   except subprocess.TimeoutExpired:
    job['elapsed_seconds']=round(time.monotonic()-start,1);q['updated_utc']=now();save(QPATH,q)
    if q['policy']['per_job_timeout_hours'] and job['elapsed_seconds']>q['policy']['per_job_timeout_hours']*3600:
     os.killpg(child.pid,signal.SIGTERM);job['termination_reason']='configured per-job wall-time watchdog';
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     code=child.returncode;break
 job.update(exit_code=code,finished_utc=now(),elapsed_seconds=round(time.monotonic()-start,1),status='completed' if code==0 else 'failed')
 if expected and expected.is_file():
  state=json.loads(expected.read_text());job['artifact_status']=state.get('status')
  if state.get('status')!='completed':job['status']='failed'
 save(QPATH,q)

def process(job,q,lookup):
 kind=job['kind'];python=sys.executable
 if kind=='existing':return
 if kind=='alias':
  target=resolve(job,lookup);p=jobpath(target)
  if p and p.is_file():job.update(status=target['status'],case_manifest=str(p),alias_of=target['id'])
  else:job.update(status='blocked',reason='Source experiment has no manifest')
  return
 if kind=='prepare':
  folder=OUT/'inputs/pilots_x1_fixed_v2'/job['dataset'];mp=folder/'manifest.json'
  if mp.is_file() and json.loads(mp.read_text()).get('status')=='completed':job.update(status='completed',artifact=str(mp));return
  run_command(job,[python,'-B',str(SCRIPTS/'queue_inputs.py'),'--dataset',job['dataset']],q,mp);return
 if kind=='build':
  run_command(job,[python,'-B',str(SCRIPTS/'build.py'),'--kind',job['build_kind']],q,OUT/'manifests'/('build_'+job['build_kind']+'.json'));return
 if kind=='profile':
  baseline=lookup['real_'+job['dataset']]
  if job['backend'] not in completed_backends(baseline,lookup):job.update(status='blocked',reason='Backend baseline/resource failed');return
  if lookup['build_profile']['status']!='completed':job.update(status='blocked',reason='Profile build failed');return
  run_command(job,[python,'-B',str(SCRIPTS/'profile_case.py'),'--id','x1v2_'+job['id'],'--dataset',job['dataset'],'--backend',job['backend'],'--tool',job['tool']],q);return
 if kind=='boundary':
  binary=OUT/'build/peak_audit/bin/evaluation_fused_boundary'
  if not binary.is_file():job.update(status='blocked',reason='Isolated fused audit build unavailable');return
  folder=OUT/'results/accuracy/x1v2_fused_boundary';job['artifact']=str(folder)
  run_command(job,[str(binary),str(folder)],q,extra_env={'LD_LIBRARY_PATH':str(ROOT/'build/mvh_rt/optix_runtime')+':/usr/local/cuda/lib64'});return
 if kind in ('run','derived_run'):
  args=job.copy();extra=[]
  if kind=='derived_run':
   source=resolve(lookup[job['baseline']],lookup);p=jobpath(source)
   if not p or not p.is_file():job.update(status='blocked',reason='Baseline has no manifest');return
   m=json.loads(p.read_text());available=[b for b in GPU if b in m.get('statistics',{})]
   if not available:job.update(status='blocked',reason='No completed GPU baseline');return
   args.update(category='accuracy',run_id='x1v2_'+job['id'],scans=m['arguments']['scans'],fasta=m['arguments']['fasta'],backends=available,repeats=1,warmups=0,batch=m['arguments']['batch'])
   if job['mode']=='flow':extra=['--gpu-binary',str(OUT/'build/flow/bin/sipros_mvh_cuda')]
   elif job['mode']=='peak_audit':
    if lookup['boundary_fused']['status']!='completed':job.update(status='blocked',reason='Fused audit boundary harness did not run successfully');return
    extra=['--gpu-binary',str(OUT/'build/peak_audit/bin/sipros_mvh_cuda'),'--peak-audit']
   else:extra=['--score-impact']
  backends=list(args['backends']);job['requested_backends']=backends.copy()
  if args.get('resource'):
   available=completed_backends(lookup[args['resource']],lookup)
   job['backend_exclusions']={b:'Full-spectra resource probe failed or unavailable; see '+args['resource'] for b in backends if b not in available}
   backends=[b for b in backends if b in available]
  if not backends:job.update(status='blocked',reason='No backend passed resource probe');return
  job['effective_backends']=backends
  path=case(args['category'],args['run_id']);job['case_manifest']=str(path)
  if path.exists():
   m=json.loads(path.read_text());job['status']=m.get('status','failed');job['reused_existing_manifest']=True;return
  cmd=[python,'-B',str(SCRIPTS/'run.py'),'--id',args['run_id'],'--category',args['category'],'--scans',str(ROOT/args['scans']),'--fasta',str(ROOT/args['fasta']),'--backends',*backends,'--batch',str(args['batch']),'--repeats',str(args['repeats']),'--warmups',str(args['warmups']),'--max-host-rss-gib','18','--continue-on-backend-failure',*extra]
  if args.get('verify'):cmd+=['--verify']
  run_command(job,cmd,q,path)
  if kind=='derived_run' and job['mode']=='peak_audit' and path.is_file():
   mapping=[python,'-B',str(SCRIPTS/'peak_raw_mapping.py'),str(path)]
   with (OUT/'results/queue'/(job['id']+'_raw_mapping.log')).open('w') as log:
    result=subprocess.run(mapping,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   job['analysis_steps']=[{'command':mapping,'exit_code':result.returncode}]
   if result.returncode:job.update(status='failed',reason='Original raw-peak identity mapping failed; search diagnostics retained')
  return
 if kind=='analyze':
  run_command(job,[python,'-B',str(SCRIPTS/'queue_analysis.py')],q);return
 raise RuntimeError('Unknown job type '+kind)

def tracker(q):
 counts={s:sum(j['status']==s for j in q['jobs']) for s in ('planned','running','completed','failed','blocked','invalidated')}
 lines=['# Serial evaluation queue','',f"Queue status: {q['status']}; updated UTC: {now()}",'',str(counts),'','| Job | State | Detail |','|---|---|---|']
 for j in q['jobs']:lines.append('| '+j['id']+' | '+j['status']+' | '+str(j.get('reason',j.get('elapsed_seconds','')))+' |')
 (OUT/'reports/queue_progress.md').write_text('\n'.join(lines)+'\n')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.execute and (OUT/'manifests/LARGE_BENCHMARKS_PAUSED.json').exists():raise RuntimeError('Large-scale experiments are paused by user; validate methodology before resuming')
 q=json.loads(QPATH.read_text()) if a.resume else create_plan();lookup={j['id']:j for j in q['jobs']}
 if not a.execute:tracker(q);print('Planned '+str(len(q['jobs']))+' serial jobs.');return
 with (OUT/'.all_experiments.lock').open('a') as lock,(OUT/'.suite.lock').open('a') as suite:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(suite,fcntl.LOCK_EX|fcntl.LOCK_NB)
  q.update(status='running',pid=os.getpid(),updated_utc=now());save(QPATH,q);tracker(q)
  try:
   for j in q['jobs']:
    if j['status'] in ('completed','failed','blocked','invalidated'):continue
    if j['status']=='running':
     pid=j.get('child_pid')
     if pid and Path('/proc/'+str(pid)).exists():raise RuntimeError('A child from the previous queue process still runs; refuse duplicate launch')
     path=jobpath(j)
     if path and path.is_file():j['status']=json.loads(path.read_text()).get('status','failed')
     if j['status']=='running':j.update(status='failed',reason='Interrupted earlier attempt retained; not silently overwritten')
     save(QPATH,q);continue
    if shutil.disk_usage(OUT).free<q['policy']['minimum_free_disk_gib']*2**30:q.update(status='blocked',reason='Free disk below configured 32GiB reserve');break
    for path,h in q['frozen_scientific_source'].items():
     if sha(ROOT/path)!=h:raise RuntimeError('Scientific source changed: '+path+'; queue stopped to avoid mixed versions')
    for path,h in q['source_templates'].items():
     if sha(ROOT/path)!=h:raise RuntimeError('Evaluation code changed after queue plan: '+path)
    print('QUEUE START '+j['id'],flush=True)
    try:process(j,q,lookup)
    except BaseException as e:
     if isinstance(e,(KeyboardInterrupt,SystemExit)):raise
     j.update(status='failed',error=repr(e),finished_utc=now());print('QUEUE FAILURE '+j['id']+' '+repr(e),flush=True)
    q['updated_utc']=now();save(QPATH,q);tracker(q);print('QUEUE '+j['status']+' '+j['id'],flush=True)
    if j.get('category') in ('real_world','batch_size','database_scaling','scan_scaling') or j.get('mode') in ('flow','peak_audit'):
     with (OUT/'results/queue/checkpoint_analysis.log').open('a') as log:
      env=os.environ.copy();env.update(PYTHONPATH=str(OUT/'dependencies'),MPLCONFIGDIR=str(OUT/'cache/matplotlib'))
      result=subprocess.run([sys.executable,'-B',str(SCRIPTS/'queue_analysis.py'),'--checkpoint'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
     if result.returncode:q['queue_history'].append({'utc':now(),'event':'checkpoint_analysis_failed','exit_code':result.returncode})
   else:q['status']='completed' if all(j['status']=='completed' for j in q['jobs']) else 'finished_with_failures'
  except BaseException as e:q.update(status='blocked',error=repr(e));raise
  finally:q['updated_utc']=now();save(QPATH,q);tracker(q)
if __name__=='__main__':main()
