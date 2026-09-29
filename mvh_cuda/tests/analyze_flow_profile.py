"""Join exact flow counts with a COUNTERS-OFF NSYS search timeline; never sum overlapping scopes."""
import argparse,csv,hashlib,json,sqlite3,re
from collections import defaultdict
from pathlib import Path

def table(path):
    with path.open() as f:return dict(list(csv.reader(f,delimiter='\t'))[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def union_ns(intervals):
    end=-1;total=0
    for a,b in sorted(intervals):
        total+=max(0,b-max(a,end));end=max(end,b)
    return total

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counted',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--sqlite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    c={k:int(v) for k,v in table(a.counted/'flow_counters.tsv').items()}
    ref=table(a.reference/'run_summary.tsv');observed=table(a.counted/'run_summary.tsv')
    assert c['total_precursor_associations']==c['total_associations_entering_fragment_stage']+c['total_associations_rejected_before_fragment_stage']
    assert c['total_fragment_queries']==c['total_fragment_hits']+c['total_fragment_misses']
    assert c['total_associations_entering_fragment_stage']==c['total_associations_with_at_least_one_fragment_hit']+c['total_associations_with_zero_fragment_hits']
    assert c['total_ions_offered_to_associations']==c['total_fragment_queries']+c['total_ions_outside_scan_range']
    assert c['total_mvh_scores_computed']==c['total_associations_entering_mvh_scoring']
    assert c['total_theoretical_ion_generation_calls']==sum(c[k] for k in ['total_direct_generation_calls','total_cache_count_generation_calls','total_cache_store_generation_calls'])
    assert c['total_theoretical_fragment_ions_generated']==sum(c[k] for k in ['total_direct_generated_ions','total_cache_count_ions','total_cache_stored_ions'])
    assert c['total_final_psm_candidates']==int(ref['retained_psm_count'])==int(observed['retained_psm_count'])
    hashes={k:sha(path/'mvh_psms.tsv') for k,path in [('counted',a.counted),('reference',a.reference)]}
    assert len(set(hashes.values()))==1,'Counter instrumentation changed PSM output'
    batch_traces={}
    for label,path in [('counted',a.counted),('reference',a.reference)]:
        log=(path.parent/(path.name+'.log')).read_text()
        assignment=re.findall(r'\[CUDA assignment\] generated=(\d+) assigned=(\d+) associations=(\d+)',log)
        cache=re.findall(r'cached_ions=(\d+) cache_charge_stride=(\d+)',log)
        legacy=re.findall(r'\[CUDA scoring\] candidates=(\d+) calls=(\d+) success=(\d+) inrange=(\d+) matched=(\d+)',log)
        legacy_totals=dict(zip(['candidates','calls','success','inrange','matched'],
                              [sum(int(row[i]) for row in legacy) for i in range(5)]))
        batch_traces[label]={'assignment':assignment,'cache':cache,'legacy_postmerge_totals':legacy_totals}
    assert batch_traces['counted']==batch_traces['reference'],'Batch association/cache decisions changed'
    assert sum(int(x[0]) for x in batch_traces['counted']['assignment'])==c['total_peptide_entries']
    assert sum(int(x[2]) for x in batch_traces['counted']['assignment'])==c['total_precursor_associations']
    db=sqlite3.connect(f'file:{a.sqlite}?mode=ro',uri=True)
    strings=dict(db.execute('select id,value from StringIds'))
    nvtx=[(s,e,t or strings.get(tid,'')) for s,e,t,tid in db.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')]
    ranges=[(s,e) for s,e,n in nvtx if n=='mvh/search/database'];assert len(ranges)==1,ranges
    begin,end=ranges[0];seconds=float(ref['search_seconds'])
    stages=[]
    def add(kind,name,intervals):
        if not intervals:return
        elapsed=sum(b-a for a,b in intervals)/1e9
        stages.append(dict(stage_name=name,timing_kind=kind,total_time_seconds=elapsed,
            percentage_of_search_time=100*elapsed/seconds,invocation_count=len(intervals),average_time_per_invocation=elapsed/len(intervals)))
    groups=defaultdict(list)
    for s,e,n in nvtx:
        if n.startswith('mvh/') and s>=begin and e<=end:groups[n].append((s,e))
    for name,intervals in groups.items():add('host_inclusive',name,intervals)
    kernels=defaultdict(list);gpu_intervals=[]
    for s,e,n in db.execute('select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(begin,end)):
        kernels[strings.get(n,str(n))].append((s,e));gpu_intervals.append((s,e))
    for name,intervals in kernels.items():add('gpu_kernel',name,intervals)
    tables={x[0] for x in db.execute("select name from sqlite_master where type='table'")}
    for t in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
        if t in tables:
            intervals=list(db.execute(f'select start,end from {t} where start>=? and end<=?',(begin,end)))
            add('gpu_memory',t,intervals);gpu_intervals+=intervals
    for t in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
        if t not in tables:continue
        api=defaultdict(list)
        for s,e,n in db.execute(f'select start,end,nameId from {t} where start>=? and end<=?',(begin,end)):
            api[strings.get(n,str(n))].append((s,e))
        for name,intervals in api.items():add('host_cuda_api_inclusive',name,intervals)
    for name,count in [('theoretical_generation_inside_scoring',c['total_direct_generation_calls']),
                       ('fragment_lookup_inside_scoring',c['total_fragment_queries']),
                       ('mvh_arithmetic_inside_scoring',c['total_mvh_scores_computed'])]:
        stages.append(dict(stage_name=name,timing_kind='fused_unavailable',total_time_seconds=None,
            percentage_of_search_time=None,invocation_count=count,average_time_per_invocation=None))
    stages.append(dict(stage_name='XCorr_WDP',timing_kind='not_executed',total_time_seconds=0,
        percentage_of_search_time=0,invocation_count=0,average_time_per_invocation=None))
    with (a.output/'stage_timings.tsv').open('w') as f:
        fields=list(stages[0]);w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows({k:('NA' if v is None else v) for k,v in row.items()} for row in stages)
    gen=c['total_theoretical_ion_generation_calls']
    report={'batch_traces_identical':True,'batch_traces':batch_traces,'counters':c,'psm_sha256':hashes,'identical':True,'reference_search_seconds':seconds,
      'counted_search_seconds':float(observed['search_seconds']),
      'observed_search_delta_percent':100*(float(observed['search_seconds'])/seconds-1),
      'overhead_caveat':'Single runs; reference includes NSYS capture. Delta is not isolated counter overhead or a speedup.',
      'generation_calls_per_peptide_entry':gen/c['total_peptide_entries'],
      'generation_calls_per_precursor_association':gen/c['total_precursor_associations'],
      'gpu_active_union_seconds_in_search':union_ns(gpu_intervals)/1e9,
      'fused_components_separate_seconds':None,'stages':stages}
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Full database-search flow measurement','',f"Backend: `{ref['match_backend']}`; batch: {ref['peptide_batch_size']}; cache: {ref['spectrum_cache']}.",
      f"PSM files identical: **yes** (`{hashes['reference']}`).",'',
      '| Counter | Count |','|---|---:|']
    lines += [f'| {k} | {v:,} |' for k,v in c.items()]
    lines += ['',f"Legacy post-merge success: **{batch_traces['counted']['legacy_postmerge_totals']['success']:,}**, versus **{c['total_mvh_scores_computed']:,}** actual speculative MVH evaluations.",'',f'Generator calls / entry: **{gen/c["total_peptide_entries"]:.6f}**; calls / association: **{gen/c["total_precursor_associations"]:.6f}**.',
      '',f'Reference search: **{seconds:.3f} s**; counted search: **{float(observed["search_seconds"]):.3f} s**.',
      'Single-run comparison; reference includes NSYS and counted run includes geometric-hit observation. Do not interpret the delta as isolated instrumentation overhead.',
      '', '| Stage | Seconds | % search | Calls | Avg s | Kind |','|---|---:|---:|---:|---:|---|']
    wanted=['mvh/search/generate_peptides','mvh/batch/assign_scans','mvh/batch/preprocess_peptides','mvh/pack/all','mvh/pack/sequence_ids','mvh/pack/spectra_and_top','mvh/rt/prepare_or_reuse','mvh/gpu/theory_cache','mvh/gpu/retain_top','mvh/gpu/compact_results','mvh/gpu/download_results','mvh/host/restore_results','mvh/batch/delete_peptides']
    for r in stages:
        if r['stage_name'] in wanted or (r['timing_kind']=='gpu_kernel' and ('optixLaunch' in r['stage_name'] or 'ScoreSequenceVsSpectrum(' in r['stage_name'] or 'countTheoreticalIons(' in r['stage_name'] or 'generateTheoreticalIons(' in r['stage_name'])) or r['timing_kind']=='gpu_memory' or r['stage_name'].split('_v')[0] in ['cudaMemcpy','cudaDeviceSynchronize','cudaEventSynchronize','cudaMalloc','cudaFree']:
            lines.append(f"| {r['stage_name']} | {r['total_time_seconds']:.6f} | {r['percentage_of_search_time']:.3f} | {r['invocation_count']} | {r['average_time_per_invocation']:.6f} | {r['timing_kind']} |")
    for r in stages:
        if r['timing_kind']=='fused_unavailable':
            lines.append(f"| {r['stage_name']} | NA | NA | {r['invocation_count']} | NA | fused, source-level calls |")
    lines.append('| XCorr/WDP | 0 | 0 | 0 | NA | not executed |')
    host={r['stage_name']:r['total_time_seconds'] for r in stages if r['timing_kind']=='host_inclusive'}
    cpu_main=host.get('mvh/pack/all',0)+host.get('mvh/search/generate_peptides',0)
    lines += ['',f"CPU-side packing plus peptide generation: **{cpu_main:.3f} s ({100*cpu_main/seconds:.2f}% of search)**. These two host scopes are disjoint. They dominate end-to-end time in this measurement; high generation-call counts alone do not establish a GPU arithmetic bottleneck."]
    lines += ['',f"Union of recorded GPU kernel/memory intervals during search: **{report['gpu_active_union_seconds_in_search']:.3f} s**.",
      'Host scopes are inclusive. CUDA waits overlap GPU execution; neither these rows nor parent/child stages can be summed.',
      'Direct theoretical generation, fragment lookup and MVH arithmetic are fused. Separate seconds are **not measurable with event boundaries in the unchanged kernel** and are reported as unavailable. XCorr/WDP are not executed.',
      'Fragment hits are tolerance-existence observations on preprocessed peaks. Backend scoring hits are separate. Retained-after-MVH counts insertion events, not final survivors.',
      '', 'Source/function and counter definitions: `mvh_cuda/FLOW_PROFILING.md`. Full timing rows: `stage_timings.tsv`.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(a.output/'REPORT.md')
if __name__=='__main__':main()
