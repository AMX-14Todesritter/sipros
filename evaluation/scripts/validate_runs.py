"""Check recorded pilot invariants; no searches or timing runs."""
import json,sys
from pathlib import Path
from inventory import OUT,sha
ROOT=OUT/'results'

def main():
    base=json.loads((ROOT/'pilot/x1_first256_fused_20261026/manifest.json').read_text())
    diag=json.loads((ROOT/'accuracy/x1_first256_scoreimpact_20261026/manifest.json').read_text())
    small=json.loads((ROOT/'accuracy/x1_first256_smallbatch_20261026/manifest.json').read_text())
    sphere=json.loads((ROOT/'accuracy/x1_first256_sphere_verified_20261026/manifest.json').read_text())
    result={'status':'completed','checks':{}}
    for b in ('cuda','rt-instanced','rt-custom'):
        pick=lambda m:next(r for r in m['runs'] if not r['warmup'] and r['backend']==b)
        a,c,d=pick(base),pick(diag),pick(small)
        equal=a['psm_sha256']==c['psm_sha256']==d['psm_sha256']
        counts=a['totals']['candidates']==c['totals']['candidates']==d['totals']['candidates']
        result['checks'][b]={'normal_diagnostic_smallbatch_psms_identical':equal,'candidate_totals_identical':counts,
                             'normal_candidates':a['totals']['candidates'],'normal_batches':len(a['batches']),'small_batches':len(d['batches'])}
        if not equal or not counts:result['status']='failed'
    result['checks']['sphere_strict_verification_psms_identical']=sphere['runs'][0]['psm_sha256']==next(r for r in base['runs'] if r['backend']=='rt-custom')['psm_sha256']
    if not result['checks']['sphere_strict_verification_psms_identical']:result['status']='failed'
    result['limitations']=['No complete per-query original peak-ID proof.','Instanced is known to differ from CPU/CUDA; same-backend invariance is not cross-backend equivalence.']
    path=OUT/'reports/pilot_invariants.json';path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if result['status']!='completed':sys.exit(1)
if __name__=='__main__':main()
