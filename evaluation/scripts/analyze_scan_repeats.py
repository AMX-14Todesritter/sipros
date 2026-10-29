"""Audit scan-study CSVs and export descriptive, matched-round comparisons."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics as st


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=read(a.input/'measurements.csv');provided=read(a.input/'summary.csv');m=json.loads((a.input/'manifest.json').read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    source_hashes={}
    for name in ('measurements.csv','summary.csv','manifest.json'):
        raw=(a.input/name).read_bytes();source_hashes[name]=hashlib.sha256(raw).hexdigest();(a.output/name).write_bytes(raw)
    groups={};keys=set()
    for r in rows:
        key=(int(r['scan_count']),r['backend'],int(r['round']))
        assert key not in keys,'Duplicate outer-round identity';keys.add(key)
        groups.setdefault(key[:2],[]).append(r)
    assert m['status']=='completed' and all(j['status']=='completed' for j in m['jobs'])
    assert len(rows)==len(groups)*m['repeats']
    points=sorted({k[0] for k in groups});summary=[];pairs=[]
    for (n,b),xs in sorted(groups.items()):
        assert sorted(int(r['round']) for r in xs)==list(range(1,m['repeats']+1))
        assert len({(r['generated'],r['candidates']) for r in xs})==1
        assert len({r['psm_sha256'] for r in xs})==1
        row={'scan_count':n,'backend':b,'repeats':len(xs),'generated':int(xs[0]['generated']),'candidates':int(xs[0]['candidates'])}
        original=next(r for r in provided if int(r['scan_count'])==n and r['backend']==b)
        for k in ('fused_seconds','complete_scoring_seconds','search_seconds','device_peak_bytes','host_peak_bytes'):
            vs=[float(r[k]) for r in xs];stats={'median':st.median(vs),'mean':st.mean(vs),'min':min(vs),'max':max(vs),'stdev':st.stdev(vs)}
            for stat,val in stats.items():
                assert math.isclose(val,float(original[k+'_'+stat]),rel_tol=1e-9,abs_tol=1e-9),(n,b,k,stat)
                row[k+'_'+stat]=val
            row[k+'_cv_percent']=stats['stdev']/stats['mean']*100
        summary.append(row)
    for n in points:
        all_at_n=[r for r in rows if int(r['scan_count'])==n]
        assert len({(r['generated'],r['candidates']) for r in all_at_n})==1
        for b in ('rt-instanced','rt-custom'):
            for round_number in range(1,m['repeats']+1):
                c=next(r for r in groups[(n,'cuda')] if int(r['round'])==round_number)
                t=next(r for r in groups[(n,b)] if int(r['round'])==round_number)
                row={'scan_count':n,'backend':b,'round':round_number}
                for metric in ('fused_seconds','complete_scoring_seconds','search_seconds'):
                    cv,tv=float(c[metric]),float(t[metric]);row[metric+'_cuda']=cv;row[metric+'_rt']=tv
                    row[metric+'_speedup']=cv/tv;row[metric+'_saved_seconds']=cv-tv;row[metric+'_time_reduction_percent']=(cv-tv)/cv*100
                row['rt_setup_seconds']=float(t['rt_setup_seconds'])
                row['nonfused_extra_seconds']=(float(t['complete_scoring_seconds'])-float(t['fused_seconds']))-(float(c['complete_scoring_seconds'])-float(c['fused_seconds']))
                row['other_nonfused_delta_excluding_setup']=row['nonfused_extra_seconds']-row['rt_setup_seconds']
                pairs.append(row)
    matched=[]
    for n in points:
        for b in ('rt-instanced','rt-custom'):
            xs=[r for r in pairs if r['scan_count']==n and r['backend']==b]
            row={'scan_count':n,'backend':b}
            for metric in ('fused_seconds','complete_scoring_seconds','search_seconds'):
                ratios=[r[metric+'_speedup'] for r in xs];savings=[r[metric+'_saved_seconds'] for r in xs]
                row.update({metric+'_speedup_median':st.median(ratios),metric+'_speedup_min':min(ratios),metric+'_speedup_max':max(ratios),metric+'_faster_rounds':sum(v>0 for v in savings),metric+'_saved_mean':st.mean(savings),metric+'_saved_min':min(savings),metric+'_saved_max':max(savings)})
            matched.append(row)
    write(a.output/'recomputed_statistics.csv',summary);write(a.output/'matched_rounds.csv',pairs);write(a.output/'matched_round_summary.csv',matched)
    result={'source_sha256':source_hashes,'validated_runs':len(rows),'cells':len(groups),'outer_rounds':m['repeats'],
            'all_summary_statistics_reproduced':True,'workload_counts_equal_within_each_scan_level':True,'same_backend_psm_stable_within_each_scan_level':True,
            'sum_process_wall_hours':sum(float(r['wall_seconds']) for r in rows)/3600,
            'caveats':['Match by outer round, not repeat: child repeat is always 1.',
                       'Round-matched runs were sequential, not simultaneous or independently randomized pairs.',
                       'Nested scan levels and rounds are not independent replicates; do not pool 21 comparisons as independent trials.',
                       'Three rounds are descriptive evidence, not a strong confidence interval or a universal significance claim.',
                       'Different-backend PSM hashes differ; hash stability is not cross-backend correctness.',
                       'Complete scoring, fused and RT setup are nested intervals; compare same-run components before attribution.']}
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['5090 SCAN-SCALING REPEATED-MEASUREMENT ANALYSIS','',json.dumps(result,indent=2),'',
           'Matched-round descriptive comparisons (CUDA time / RT time):']
    for r in matched:
        lines.append(f"{r['scan_count']:6d} {r['backend']:12s} fused={r['fused_seconds_speedup_min']:.4f}..{r['fused_seconds_speedup_max']:.4f}x; complete faster={r['complete_scoring_seconds_faster_rounds']}/3; search faster={r['search_seconds_faster_rounds']}/3; search saved={r['search_seconds_saved_min']:.3f}..{r['search_seconds_saved_max']:.3f}s")
    (a.output/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
