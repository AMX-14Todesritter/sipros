"""Export every completed nested-scan run, distributions and paired speedups."""
import csv,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'output/evaluation/scan_scaling_marine_nested'
m=json.loads((D/'manifest.json').read_text());jobs=sorted([j for j in m['jobs'] if j['status']=='completed'],key=lambda j:(j['sample'],j['backend'],j['repeat']))
rows=[]
for j in jobs:
 x=j['metrics'];rows.append({'sample':j['sample'],'scans':m['samples'][j['sample']]['scan_count'],'backend':j['backend'],'repeat':j['repeat'],**x,'device_peak_gib':x['device_peak_bytes']/2**30,'host_peak_gib':x['host_peak_bytes']/2**30})
def csvout(name,rows):
 with (D/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
csvout('all_24_runs.csv',rows)
stats=[];pairs=[]
for s in ('S1','S2','S3'):
 for b in ('cuda','rt-custom'):
  rs=[r for r in rows if r['sample']==s and r['backend']==b];out={'sample':s,'backend':b,'n':len(rs)}
  for k in ('fused_seconds','rt_setup_seconds','complete_scoring_seconds','search_seconds','device_peak_gib','host_peak_gib'):
   v=[r[k] for r in rs];avg=statistics.mean(v);sd=statistics.stdev(v)
   out.update({k+'_mean':avg,k+'_median':statistics.median(v),k+'_min':min(v),k+'_max':max(v),k+'_stdev':sd,k+'_cv_percent':100*sd/avg if avg else 0})
  out['search_scans_per_second']=rs[0]['scans']/out['search_seconds_mean'];out['fused_candidates_per_second']=rs[0]['candidates']/out['fused_seconds_mean'];stats.append(out)
 for rep in (1,2,3,4):
  c=next(r for r in rows if r['sample']==s and r['backend']=='cuda' and r['repeat']==rep);rt=next(r for r in rows if r['sample']==s and r['backend']=='rt-custom' and r['repeat']==rep)
  pairs.append({'sample':s,'repeat':rep,**{k+'_speedup':c[k]/rt[k] for k in ('fused_seconds','complete_scoring_seconds','search_seconds')}})
csvout('distribution_statistics.csv',stats);csvout('paired_speedups.csv',pairs)
for r in stats:print(r['sample'],r['backend'],*[f"{r[k+'_mean']:.3f} [{r[k+'_min']:.3f},{r[k+'_max']:.3f}] CV{r[k+'_cv_percent']:.2f}%" for k in ('fused_seconds','complete_scoring_seconds','search_seconds','device_peak_gib')])
for s in ('S1','S2','S3'):
 a,b=[next(r for r in stats if r['sample']==s and r['backend']==back) for back in ('cuda','rt-custom')]
 print('SPEED',s,{k:round(a[k+'_mean']/b[k+'_mean'],4) for k in ('fused_seconds','complete_scoring_seconds','search_seconds')})
 print('PAIRS',s,[{k:round(v,4) for k,v in r.items() if k.endswith('speedup')} for r in pairs if r['sample']==s])
