"""Link counters-only runs to baseline timings only after workload/output checks."""
import argparse,csv,json
from pathlib import Path
from inventory import OUT
from run import save
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True);a=p.parse_args()
b=json.loads(a.baseline.read_text());d=json.loads(a.diagnostic.read_text())
assert b['status']==d['status']=='completed' and not b['diagnostic'] and d['diagnostic']
assert b['inputs']==d['inputs'] and b['source_sha256']==d['source_sha256']
rows=[];link={'status':'completed','baseline_manifest':str(a.baseline),'diagnostic_manifest':str(a.diagnostic),'checks':{}}
for backend in ('cuda','rt-instanced','rt-custom'):
 br=next(r for r in b['runs'] if r['backend']==backend and not r['warmup']);dr=next(r for r in d['runs'] if r['backend']==backend and not r['warmup'])
 assert br['psm_sha256']==dr['psm_sha256'],backend+' counters changed PSMs'
 cache=lambda r:[(x['candidates'],x['cached_ions'],x['cache_charge_stride']) for x in r['batches']]
 assert all(cache(r)==cache(dr) for r in b['runs'] if r['backend']==backend and not r['warmup']),backend+' cache/input workload changed'
 assert all(r['psm_sha256']==dr['psm_sha256'] for r in b['runs'] if r['backend']==backend and not r['warmup']),backend+' repeat outputs changed'
 counters=a.diagnostic.parent/dr['name']/'flow_counters.tsv'
 with counters.open() as f:c={row['counter']:int(row['value']) for row in csv.DictReader(f,delimiter='\t')}
 assert c['total_fragment_hits']+c['total_fragment_misses']==c['total_fragment_queries']
 assert c['total_precursor_associations']==int(br['totals']['candidates'])
 item={'backend':backend,'fragment_queries':c['total_fragment_queries'],'geometric_hits':c['total_fragment_hits'],'positive_class_hits':c['total_backend_scored_fragment_hits'],
       'precursor_associations':c['total_precursor_associations'],'generated_peptide_entries':c['total_peptide_entries'],'mvh_scores_computed':c['total_mvh_scores_computed'],
       'fused_seconds_median':b['statistics'][backend]['fused_device_seconds']['median'],'psm_diagnostic_equal':True,'cache_decisions_equal':True,'baseline_repetitions_checked':5,'counters_file':str(counters)}
 item['query_throughput_per_second']=item['fragment_queries']/item['fused_seconds_median'];rows.append(item);link['checks'][backend]=item
link['query_denominators_identical']=len({r['fragment_queries'] for r in rows})==1
save(a.baseline.parent/'linked_counters.json',link)
with (a.baseline.parent/'workload_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(link,indent=2))
