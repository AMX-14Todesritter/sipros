"""Continuous wall intervals and independent full-service measurements."""
import json,csv,re,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';D=E/'diagnostics/continuous_control'
m=json.loads((E/'results/accuracy/marine_continuous_replay/manifest.json').read_text());rows=[]
for run in m['runs']:
 groups={cache:{i:[] for i in range(3,10)} for cache in ('auto','disabled')}
 for l in (E/'results/accuracy/marine_continuous_replay'/(run['name']+'.log')).read_text().splitlines():
  if not l.startswith('[CONTINUOUS replay]'):continue
  v=dict(re.findall(r'(\w+)=([^\s]+)',l))
  if v['warmup']=='0':groups[v['cache']][int(v['trial'])].append(v)
 for cache,g in groups.items():
  assert all(len(v)==len(run['batches']) for v in g.values())
  series={k:[sum(float(b[k])*1000 for b in g[i]) for i in range(3,10)] for k in ('theory_seconds','fused_seconds','continuous_seconds')}
  row={'backend':run['backend'],'cache':cache,'measured_repetitions':7,'warmups':3}
  for k,a in series.items():
   label=k.replace('_seconds','');row.update({label+'_median_ms':statistics.median(a),label+'_min_ms':min(a),label+'_max_ms':max(a)})
  rows.append(row)
service=[]
for record in json.loads((D/'runs.json').read_text()):
 if record['id']=='marine_continuous_replay':continue
 p=E/'results/accuracy'/record['id']/'manifest.json';n=json.loads(p.read_text())
 for run in n['runs']:
  t=run['totals'];service.append({'id':record['id'],'backend':run['backend'],'cache':record['cache'],'gpu_service_ms':t['gpu_service_seconds']*1000,'pack_service_restore_ms':sum(t[k] for k in ('pack_seconds','gpu_service_seconds','replay_verify_seconds'))*1000,'search_ms':float(run['summary']['search_seconds'])*1000,'psm_sha256':run['psm_sha256']})
summary=[]
for backend in ('cuda','rt-instanced','rt-custom'):
 for cache in ('auto','disabled'):
  selected=[s for s in service if s['backend']==backend and s['cache']==cache]
  row={'backend':backend,'cache':cache,'fresh_process_repeats':len(selected),'psm_identical':len({x['psm_sha256'] for x in selected})==1}
  for k in ('gpu_service_ms','pack_service_restore_ms','search_ms'):row[k]=statistics.median(x[k] for x in selected)
  summary.append(row)
for filename,data in [('continuous_timings.csv',rows),('service_runs.csv',service),('service_summary.csv',summary)]:
 with (D/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(D/'analysis.json').write_text(json.dumps({'continuous':rows,'service':summary},indent=2))
print(json.dumps({'continuous':rows,'service':summary},indent=2))
