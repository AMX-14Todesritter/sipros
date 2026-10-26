"""Consolidate finished and failed manifests without inventing missing metrics."""
import csv,json,statistics
from pathlib import Path
from inventory import ROOT, OUT

def main():
    summaries=[];statuses=[];comparison_rows=[];scope_exclusions=[]
    cfg=json.loads((ROOT/'evaluation/configs/workloads.json').read_text())
    fixed=(ROOT/cfg['workloads']['ecoli_x1']['ft2']).resolve()
    for path in sorted((OUT/'results').rglob('manifest.json')):
        m=json.loads(path.read_text());statuses.append({'path':str(path.relative_to(OUT)),'status':m.get('status'),'error':m.get('error')})
        if (m.get('status')!='completed' and not m.get('measurements_validated')) or 'statistics' not in m:continue
        if m.get('category') in ('real_world','batch_size','database_scaling','end_to_end') and Path(m['arguments']['scans']).resolve()!=fixed:
            scope_exclusions.append({'manifest':str(path.relative_to(OUT)),'reason':'Superseded fixed-scan policy: all fixed workloads must use Pan X1','scans':m['arguments']['scans']})
            continue
        for backend,stats in m['statistics'].items():
            if m['diagnostic'] or m['category'] in ('accuracy','pilot'):continue
            runs=[r for r in m['runs'] if r['backend']==backend and not r['warmup']]
            n=stats['search_seconds']['n'];totals=runs[0]['totals'];candidates=totals.get('candidates')
            row={'case_status':m['status'],'comparison_reference':m.get('comparison_reference','cuda'),'experiment':m['id'],'category':m['category'],'backend':backend,'diagnostic':m['diagnostic'],'repetitions':n,'scan_count':runs[0]['summary']['scan_count'],
                 'retained_psms':runs[0]['summary']['retained_psm_count'],'generated_peptides':runs[0]['generated_peptides'] if backend!='cpu' else None,
                 'candidates':candidates,'batch_generated_peptides':m['arguments']['batch'],'repeat_psm_identical':stats['repeat_psm_identical'],'manifest':str(path.relative_to(OUT))}
            for metric in ('search_seconds','wall_seconds','kernel_seconds','fused_device_seconds','theory_seconds','gpu_service_seconds'):
                for stat in ('median','min','max','stdev'):row[metric+'_'+stat]=stats.get(metric,{}).get(stat)
            duration=row.get('fused_device_seconds_median')
            row['candidate_throughput_per_second']=candidates/duration if candidates is not None and duration else None
            row['fragment_queries']=None # Never substitute logical post-merge inrange counts.
            linked=path.parent/'linked_counters.json'
            if linked.is_file():
                counters=json.loads(linked.read_text())['checks'].get(backend)
                if counters:
                    row.update(fragment_queries=counters['fragment_queries'],query_throughput_per_second=counters['query_throughput_per_second'],
                               geometric_hits=counters['geometric_hits'],positive_class_hits=counters['positive_class_hits'],mvh_scores_computed=counters['mvh_scores_computed'])
            row['device_sampled_peak_bytes']=max(r['device_peak_bytes'] for r in runs)
            row['host_observed_hwm_bytes']=max(r['host_observed_hwm_bytes'] for r in runs)
            row['fused_cuda_over_backend_ratio']=None
            if backend.startswith('rt-') and duration and 'cuda' in m['statistics']:
                ref=m['statistics']['cuda'].get('fused_device_seconds',{}).get('median')
                if ref:row['fused_cuda_over_backend_ratio']=ref/duration
            row['psm_consistency_vs_cuda']=m.get('comparisons',{}).get(backend,{}).get('spectrum_groups_differ') if m.get('comparison_reference','cuda')=='cuda' else None
            row['equivalent_peak_results_established']=False
            summaries.append(row)
        for backend,c in m.get('comparisons',{}).items():comparison_rows.append({'experiment':m['id'],'backend':backend,**c})
    reports=OUT/'reports';reports.mkdir(parents=True,exist_ok=True)
    (reports/'scope_exclusions.json').write_text(json.dumps(scope_exclusions,indent=2))
    (reports/'run_status.json').write_text(json.dumps(statuses,indent=2));(reports/'consolidated.json').write_text(json.dumps(summaries,indent=2))
    def table(name,rows):
        with (reports/name).open('w') as f:
            if rows:
                w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    table('consolidated.csv',summaries);table('psm_consistency.csv',comparison_rows)
    lines=['# Preliminary evaluation findings','','Current scope: Pan X1 for all fixed FASTA workloads; Pan X2/X3 only for scan scaling. Superseded native-spectrum and standalone X2/X3 real-world results are excluded. No pure matching measurements are included.','',
           '## Verified observations','','- Initial 6/7 checks passed; conversion-source provenance test failed because pre-existing CPU_BASELINE.json contains merge markers.',
           '- X1/X2/X3/Marine/Soil counts are independently inspected. RAW conversion manifests document recovered FT2 inputs.',
           '- Strict verification on real X1 first256 aborted for rt-instanced. The incomplete diagnostic is retained, not counted as a completed benchmark.',
           '', '| Experiment | Backend | n | Search median (s) | Fused median (s) | PSM scan groups differing vs CUDA |','|---|---|---:|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{x:.6g}'
    for r in summaries:lines.append(f"| {r['experiment']} | {r['backend']} | {r['repetitions']} | {fmt(r['search_seconds_median'])} | {fmt(r['fused_device_seconds_median'])} | {r['psm_consistency_vs_cuda'] if r['psm_consistency_vs_cuda'] is not None else '—'} |")
    lines += ['', '## Interpretation limits','','Performance ratios remain measurements of distinct implementations. They are not evidence of equivalent results. Small prefixes are not representative full-dataset performance. Cache-policy differences must be reported with their timings. No conclusion about biological accuracy, FDR, full peak-ID agreement or RT-core utilization is justified.',
              '', '## Remaining work','','Complete fused-path peak-ID/boundary diagnostics, diagnostic/normal PSM parity, workload counters, resource pilots, full real-world/batch/scaling runs and profiler analysis. Missing values remain blank.']
    for path in sorted((OUT/'results/real_world').glob('*/linked_counters.json')):
        link=json.loads(path.read_text());checks=link['checks'];ref=checks['cuda']
        lines += ['', '## Linked full-workload counters', '',
                  f"{path.parent.name}: {ref['fragment_queries']:,} fragment queries per GPU backend, {ref['precursor_associations']:,} associations, {ref['generated_peptide_entries']:,} generated peptide entries. Independent diagnostic counts were linked only after source/input, five-repeat cache policy and PSM checks."]
        for backend in ('rt-instanced','rt-custom'):
            delta=checks[backend]['positive_class_hits']-ref['positive_class_hits']
            lines.append(f"- {backend}: positive-class hit count net difference vs CUDA = {delta:+,}. This aggregate is not an exact per-query mismatch count. Equal retained PSMs do not establish peak-level identity agreement.")
    (reports/'findings.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; consolidated tables saved; figures pending');return
    for exp in sorted({r['experiment'] for r in summaries if not r['diagnostic']}):
        rows=[r for r in summaries if r['experiment']==exp and r['fused_device_seconds_median'] is not None]
        if not rows:continue
        fig,ax=plt.subplots(figsize=(6,3.5));x=list(range(len(rows)));y=[r['fused_device_seconds_median'] for r in rows]
        errors=[[r['fused_device_seconds_median']-r['fused_device_seconds_min'] for r in rows],[r['fused_device_seconds_max']-r['fused_device_seconds_median'] for r in rows]]
        ax.bar(x,y,yerr=errors,capsize=4,color=['#4675a4','#d99b45','#62966e'][:len(rows)]);ax.set_xticks(x,[r['backend'] for r in rows]);ax.set_ylabel('Fused scoring event time (s)');ax.set_title(exp.replace('_',' '),fontsize=9)
        ax.spines[['top','right']].set_visible(False);fig.tight_layout()
        for ext in ('pdf','svg','png'):fig.savefig(OUT/'figures'/f'{exp}_fused.{ext}',dpi=180)
        plt.close(fig)
    for exp in sorted({r['experiment'] for r in summaries if r['category']=='real_world'}):
        rows=[r for r in summaries if r['experiment']==exp]
        fig,ax=plt.subplots(figsize=(6,3.5));x=list(range(len(rows)));width=.35
        ax.bar([i-width/2 for i in x],[r['search_seconds_median'] for r in rows],width,label='MVH search')
        ax.bar([i+width/2 for i in x],[r['wall_seconds_median'] for r in rows],width,label='Process wall time')
        ax.set_xticks(x,[r['backend'] for r in rows]);ax.set_ylabel('Median runtime (s)');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);fig.tight_layout()
        for ext in ('pdf','svg','png'):fig.savefig(OUT/'figures'/f'{exp}_application.{ext}',dpi=180)
        plt.close(fig)
    print('Consolidated '+str(len(summaries))+' backend summaries.')
if __name__=='__main__':main()
