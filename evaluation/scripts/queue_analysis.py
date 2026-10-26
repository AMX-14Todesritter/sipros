"""Consolidate queue evidence, link diagnostic counters, and plot agreed sweeps."""
import argparse,csv,json,math,statistics,subprocess,sys
from collections import defaultdict
from pathlib import Path
from inventory import ROOT,OUT,sha
from run import save
p=argparse.ArgumentParser();p.add_argument('--checkpoint',action='store_true');a=p.parse_args()
q=json.loads((OUT/'manifests/all_experiments.json').read_text());jobs={j['id']:j for j in q['jobs']}
def source(j):
 seen=set()
 while j['kind']=='alias':
  assert j['id'] not in seen;seen.add(j['id']);j=jobs[j['target']]
 return j
def read(j):
 j=source(j);path=Path(j['case_manifest']) if j.get('case_manifest') else None
 return (path,json.loads(path.read_text())) if path and path.is_file() else (None,None)
rows=[];accuracy=[];stages=[];notes=[]
for j in jobs.values():
 if j.get('category') not in ('real_world','batch_size','database_scaling','scan_scaling'):continue
 path,m=read(j)
 if not m:continue
 if m.get('status')!='completed' and not m.get('measurements_validated'):continue
 stats=m.get('statistics',{})
 flow=jobs.get('flow_'+source(j)['id']);fp,fm=read(flow) if flow else (None,None)
 for backend,s in stats.items():
  if backend=='cpu' and j['category']!='real_world':continue
  if s['search_seconds']['n']!=5 or not s.get('repeat_psm_identical'):continue
  runs=[r for r in m['runs'] if r['backend']==backend and not r['warmup'] and r.get('status')=='completed']
  row={'job':j['id'],'category':j['category'],'dataset':j.get('dataset',m.get('id')),'backend':backend,'parameter':j.get('parameter'),'protein_entries':j.get('protein_entries'),
       'scan_count':int(runs[0]['summary']['scan_count']),'batch_generated_peptides':m['arguments']['batch'],'repetitions':5,'source_manifest':str(path),'reused_from':j.get('alias_of',j.get('target')),'case_status':m['status']}
  for metric in ('fused_device_seconds','search_seconds','wall_seconds','theory_seconds','gpu_service_seconds'):
   for key in ('median','min','max','stdev'):row[metric+'_'+key]=s.get(metric,{}).get(key)
  row['gpu_memory_sampled_peak_bytes']=max(r['device_peak_bytes'] for r in runs);row['host_hwm_observed_bytes']=max(r['host_observed_hwm_bytes'] for r in runs)
  row.update(fragment_queries=None,query_throughput_per_second=None,geometric_hits=None,positive_class_hits=None,actual_mvh_operations=None,counters_validated=False)
  if fm and backend in fm.get('statistics',{}):
   dr=next(r for r in fm['runs'] if r['backend']==backend and not r['warmup'] and r.get('status')=='completed')
   equal=lambda r:[(b['candidates'],b['cached_ions'],b['cache_charge_stride']) for b in r['batches']]
   valid=m['inputs']==fm['inputs'] and m['source_sha256']==fm['source_sha256'] and all(r['psm_sha256']==dr['psm_sha256'] and equal(r)==equal(dr) for r in runs)
   file=fp.parent/dr['name']/'flow_counters.tsv'
   if valid and file.is_file():
    with file.open() as f:c={r['counter']:int(r['value']) for r in csv.DictReader(f,delimiter='\t')}
    assert c['total_fragment_hits']+c['total_fragment_misses']==c['total_fragment_queries']
    row.update(fragment_queries=c['total_fragment_queries'],geometric_hits=c['total_fragment_hits'],positive_class_hits=c['total_backend_scored_fragment_hits'],actual_mvh_operations=c['total_mvh_scores_computed'],counters_validated=True,precursor_associations=c['total_precursor_associations'],generated_peptide_entries=c['total_peptide_entries'])
    if row['fused_device_seconds_median']:row['query_throughput_per_second']=c['total_fragment_queries']/row['fused_device_seconds_median']
   elif file.is_file():notes.append(j['id']+'/'+backend+': counter run cache, input or output mismatch; do not link.')
  rows.append(row)
  comp=m.get('comparisons',{}).get(backend)
  if comp:accuracy.append({'job':j['id'],'backend':backend,'reference':m.get('comparison_reference','cuda'),**comp})
  for r in runs:
   summary=r['summary'];timing=r['totals'];item={'job':j['id'],'backend':backend,'repeat':r['repeat'],'wall_seconds':r['wall_seconds']}
   for k in ('config_and_load_seconds','preprocess_seconds','search_seconds','export_seconds'):item[k]=float(summary[k])
   item['process_other_seconds']=r['wall_seconds']-sum(item[k] for k in ('config_and_load_seconds','preprocess_seconds','search_seconds','export_seconds'))
   if backend!='cpu':
    for k in ('pack_seconds','gpu_service_seconds','allocation_upload_seconds','theory_seconds','kernel_seconds','fused_device_seconds','retention_seconds','compaction_seconds','download_seconds','replay_verify_seconds'):item[k]=timing.get(k)
    item['rt_setup_seconds']=r['rt_setup_seconds']
    item['search_other_seconds']=item['search_seconds']-sum(timing.get(k,0) for k in ('pack_seconds','gpu_service_seconds','replay_verify_seconds'))
   stages.append(item)
report=OUT/'reports';report.mkdir(parents=True,exist_ok=True)
def write(name,data):
 with (report/name).open('w') as f:
  if data:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)
write('all_experiments_performance.csv',rows);write('all_experiments_psm_consistency.csv',accuracy);write('all_experiments_stage_breakdown.csv',stages)
save(report/'all_experiments_summary.json',{'queue_status':q['status'],'performance':rows,'consistency':accuracy,'notes':notes,'limits':['No FDR-filtered biological accuracy without a common validated downstream pipeline.','Fused event interval is distinct from profiler kernel duration.','Resource/OOM/timeout and all failed attempts are preserved in queue logs.']})
# Full fused peak audit counters, keeping instrumentation/output equality explicit.
peak_rows=[]
for j in jobs.values():
 if j.get('mode')!='peak_audit':continue
 ap,am=read(j);bp,bm=read(jobs[j['baseline']])
 if not am or not bm:continue
 for ar in am['runs']:
  if ar.get('status')!='completed' or ar['warmup']:continue
  backend=ar['backend'];br=next((r for r in bm['runs'] if r['backend']==backend and not r['warmup'] and r.get('status')=='completed'),None)
  folder=ap.parent/ar['name']/'peak_audit';file=folder/'peak_audit_batches.tsv'
  if not file.is_file():continue
  with file.open() as f:data=list(csv.DictReader(f,delimiter='\t'))
  totals={k:sum(int(r[k]) for r in data) for k in data[0] if k!='batch'} if data else {}
  peak_rows.append({'dataset':j['id'][5:],'backend':backend,**totals,'normal_psm_equal':bool(br and br['psm_sha256']==ar['psm_sha256']),'peak_mismatch_rate':totals.get('peak_identity_mismatches',0)/totals['queries'] if totals.get('queries') else None,'class_mismatch_rate':totals.get('class_mismatches',0)/totals['queries'] if totals.get('queries') else None,'examples_limit_per_batch':128,'peak_identity_scope':'common retained peak ordinal mapped to original m/z; raw duplicate memberships retained separately','audit_folder':str(folder)})
write('all_experiments_peak_accuracy.csv',peak_rows)
try:
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 for category,label in [('batch_size','Requested generated peptide batch size'),('database_scaling','Soil FASTA entries'),('scan_scaling','MS2 scans')]:
  subset=[r for r in rows if r['category']==category and r['backend']!='cpu' and r['fused_device_seconds_median'] is not None]
  if not subset:continue
  fig,axes=plt.subplots(1,2,figsize=(10,3.5))
  for backend in ['cuda','rt-instanced','rt-custom']:
   points=sorted([r for r in subset if r['backend']==backend],key=lambda r:r['protein_entries'] if category=='database_scaling' else (r['scan_count'] if category=='scan_scaling' else r['batch_generated_peptides']))
   x=[r['protein_entries'] if category=='database_scaling' else (r['scan_count'] if category=='scan_scaling' else r['batch_generated_peptides']) for r in points]
   y=[r['fused_device_seconds_median'] for r in points];axes[0].plot(x,y,'o-',label=backend)
   ratios=[];xs=[]
   for point in points:
    ref=next((r for r in subset if r['backend']=='cuda' and r['job']==point['job']),None)
    if ref:xs.append(point['protein_entries'] if category=='database_scaling' else (point['scan_count'] if category=='scan_scaling' else point['batch_generated_peptides']));ratios.append(ref['fused_device_seconds_median']/point['fused_device_seconds_median'])
   if backend!='cuda' and xs:axes[1].plot(xs,ratios,'o-',label=backend)
  for ax in axes:ax.set_xlabel(label);ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
  axes[0].set_ylabel('Median fused scoring event time (s)');axes[1].set_ylabel('CUDA time / RT time');axes[1].axhline(1,color='gray',linestyle='--',linewidth=.7);fig.tight_layout()
  for ext in ('pdf','svg','png'):fig.savefig(OUT/'figures'/('all_'+category+'.'+ext),dpi=200)
  plt.close(fig)
except ImportError:notes.append('Plotting dependency unavailable')
subprocess.run([sys.executable,'-B',str(ROOT/'evaluation/scripts/summarize.py')],cwd=ROOT,check=True)
print('Queue analysis: '+str(len(rows))+' performance rows; '+str(len(peak_rows))+' fused audit rows.')
