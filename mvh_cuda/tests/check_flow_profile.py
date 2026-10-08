"""Validate full-flow count identities on cached and direct search fixtures."""
import argparse,csv,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);a=p.parse_args()
data=Path(__file__).resolve().parent/'data'
with tempfile.TemporaryDirectory() as tmp:
 for backend in ['cuda','rt-custom']:
  out=Path(tmp)/backend
  r=subprocess.run([str(a.binary),'-f',str(data/'sample.ft2'),'-c',str(data/'search.cfg'),'-fasta',str(data/'proteins.fasta'),'-o',str(out),'--match-backend',backend,'--peptide-batch-size','3'],capture_output=True,text=True)
  assert r.returncode==0,r.stdout+r.stderr
  c={k:int(v) for k,v in list(csv.reader((out/'flow_counters.tsv').open(),delimiter='\t'))[1:]}
  assert c['total_precursor_associations']==c['total_associations_entering_fragment_stage']+c['total_associations_rejected_before_fragment_stage']
  assert c['total_fragment_queries']==c['total_fragment_hits']+c['total_fragment_misses']
  if backend == 'cuda':
   assert c['cuda_search_calls'] == c['total_fragment_queries']
   assert c['rt_trace_calls'] == 0
  else:
   assert c['rt_trace_calls'] <= c['total_fragment_queries']
   assert c['rt_closest_hit_calls'] + c['rt_miss_calls'] == c['rt_trace_calls']
   assert c['cuda_search_calls'] == 0
  assert c['total_associations_entering_fragment_stage']==c['total_associations_with_at_least_one_fragment_hit']+c['total_associations_with_zero_fragment_hits']
  assert c['total_ions_offered_to_associations']==c['total_fragment_queries']+c['total_ions_outside_scan_range']
  assert c['total_mvh_scores_computed']==c['total_associations_entering_mvh_scoring']
  assert c['total_candidates_retained_after_mvh']<=c['total_mvh_scores_computed']
  assert c['total_final_psm_candidates']==len((out/'mvh_psms.tsv').read_text().splitlines())-1
  assert c['total_theoretical_ion_generation_calls']==sum(c[k] for k in ['total_direct_generation_calls','total_cache_count_generation_calls','total_cache_store_generation_calls'])
  assert c['total_theoretical_fragment_ions_generated']==sum(c[k] for k in ['total_direct_generated_ions','total_cache_count_ions','total_cache_stored_ions'])
  assert c['total_cache_count_generation_calls']>0 and c['total_cache_store_generation_calls']>0
  print('PASS',backend,'flow identities and cache accounting')
