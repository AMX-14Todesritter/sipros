"""Read-only source/data audit; all outputs live under output/evaluation."""
import collections, hashlib, json, platform, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/evaluation'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def command(args):
    p=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    return {'command':args,'exit_code':p.returncode,'output':p.stdout}

def inspect(path,kind):
    result={'path':str(path.relative_to(ROOT)),'exists':path.is_file()}
    if not result['exists']:return result
    result.update(bytes=path.stat().st_size,sha256=sha(path))
    if kind=='fasta':
        n=0; prefixes=collections.Counter(); examples=[]
        with path.open(errors='strict') as f:
            for line in f:
                if line.startswith('>'):
                    n+=1; ident=line[1:].split()[0]
                    prefix=next((x for x in ['Rev_1_','Rev_2_','TestRev_','Rev1_','Rev2_','Rev3_','Rev_','rev_','rev','DECOY_','decoy_'] if ident.startswith(x)),'other')
                    prefixes[prefix]+=1
                    if len(examples)<12:examples.append(ident)
        result.update(entries=n,prefix_counts=dict(prefixes),first_ids=examples)
    else:
        scans=0; peak_count=0; ids=set(); duplicates=0; first_z=None; missing=0; counts=[]; current=0; extra_z=0
        with path.open() as f:
            for line in f:
                if line.startswith('S\t') or line.startswith('S '):
                    if scans:
                        counts.append(current)
                        if first_z is None or first_z<=0:missing+=1
                    parts=line.split(); ident=parts[1]
                    duplicates+=ident in ids;ids.add(ident);scans+=1;current=0;first_z=None
                elif line.startswith('Z\t') or line.startswith('Z '):
                    if first_z is None:first_z=int(line.split()[1])
                    else:extra_z+=1
                elif line and (line[0].isdigit() or line[0] in '.-'):
                    current+=1;peak_count+=1
        if scans:
            counts.append(current)
            if first_z is None or first_z<=0:missing+=1
        result.update(scans=scans,raw_peaks=peak_count,duplicate_scan_ids=duplicates,invalid_primary_charge_scans=missing,extra_z_lines=extra_z,
                      raw_peak_count_min=min(counts,default=0),raw_peak_count_max=max(counts,default=0),raw_peaks_per_scan=peak_count/scans if scans else None)
    return result

def main():
    (OUT/'reports').mkdir(parents=True,exist_ok=True);(OUT/'manifests').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'evaluation/configs/workloads.json').read_text());files={}
    for w in cfg['workloads'].values():
        for key in ('fasta','ft2'):
            if w[key] not in files:files[w[key]]=inspect(ROOT/w[key],key)
        if w.get('raw'):files[w['raw']]={'path':w['raw'],'exists':(ROOT/w['raw']).is_file()}
    env={name:command(args) for name,args in {
        'git_commit':['git','rev-parse','HEAD'],'git_status':['git','status','--short','--branch'],
        'gpu':['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu,utilization.gpu','--format=csv'],
        'nvcc':['/usr/local/cuda/bin/nvcc','--version'],'compiler':['c++','--version'],
        'cpu':['lscpu'],'memory':['free','-b'],'nsys':['nsys','--version'],'ncu':['/usr/local/cuda/bin/ncu','--version']}.items()}
    tracked=command(['git','ls-files','*.cpp','*.cu','*.cuh','*.h','CMakeLists.txt'])['output'].splitlines()
    source_hashes={p:sha(ROOT/p) for p in tracked if (ROOT/p).is_file()}
    merged=[str(p.relative_to(ROOT)) for base in ('raw','test_output','output') for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix.lower()=='.ft2' and any(s in p.name.lower() for s in ('merge','joint','combined'))]
    report={'files':files,'environment':env,'source_sha256':source_hashes,'merged_ft2_candidates':merged,'disk':dict(zip(('total','used','free'),shutil.disk_usage(ROOT)))}
    (OUT/'manifests/inventory.json').write_text(json.dumps(report,indent=2))
    rows=['# Repository and environment inventory','','Primary branch: profiling_0.0. Actual source commit: '+env['git_commit']['output'].strip(),'',
          '| Input | Available | Count |','|---|---|---|']
    for path,item in files.items():rows.append(f"| {path} | {item['exists']} | {item.get('entries',item.get('scans','—'))} |")
    rows += ['', '## Available implementations', '', 'CPU: mvh; CUDA: mvh_cuda; RT V1: rt-instanced; RT V2: rt-custom (split-m/z sphere).',
      'Batch size counts generated peptide entries including PTM variants and duplicates. It does not count scans, candidates or fragment queries.',
      '', '## Timing audit', '',
      'kernel_seconds is a CUDA-event interval around fused scoring. RT launch also stages Params and synchronizes; therefore this interval can include transfer/submission gaps. theory_seconds separately covers optional cache construction. gpu_service_seconds includes allocations/uploads, setup, theory, scoring, retention and downloads; search_seconds includes host generation/packing/restoration. Neither is the full Python SIP/FDR workflow.',
      '', '## Reusable tools', '',
      '- MVH_RT/gpu_bridge/benchmark.py: rotating backend order, log metrics and memory polling; requires replacement of hardcoded binary/input assumptions and mean-only statistics.',
      '- MVH_RT/gpu_bridge/validate_score_impact.py, score_impact_report.py: diagnostic candidate score/count/top impact.',
      '- MVH_RT/gpu_bridge/profile.py and mvh_cuda/tests/profile_gpu.py: NSYS/NCU commands, inspect paths before reuse.',
      '- mvh_cuda/tests and MVH_RT/gpu_bridge contract binaries: small fused-scoring semantic checks.',
      '- tools/raxport and script33/search.py: RAW conversion.',
      '', '## Missing evidence/instrumentation', '',
      '- Exact per-query original peak-ID agreement and all specified float/tie boundary cases within the fused path; existing PSM equality is insufficient.',
      '- Complete fragment-query/hit counters require separate diagnostic flow-counter build; ordinary inrange/matched values describe logically replayed scoring, not all GPU operations.',
      '- Kernel-only RT timing must exclude Params staging; preserve the existing event interval as a separate scope.',
      '- Retained peak distribution and target-decoy family mapping need validation before scaling.',
      '- Existing MVH-only outputs do not establish comparable FDR-filtered PSM sets; no biological accuracy/FDR claims yet.',
      '', '## Data recovery and risks', '',
      'Current scan policy: Pan X1 for every fixed-workload FASTA; Pan X2/X3 only for source-stratified scan scaling. Prepared Pan inputs and nested scan inputs are retained and validated. Native Human/Marine/Soil FT2 files are out of scope.',
      'Known CPU_BASELINE.json contains unresolved merge markers. Record provenance test failure; do not fabricate a corrected baseline.',
      'Independent profiling/flow builds are required; production timing runs use both switches OFF.',
      '', '## Environment', '', '```',env['gpu']['output'],env['nvcc']['output'],'```',
      '', 'Machine-readable details, source hashes, file hashes and input header conventions: ../manifests/inventory.json.']
    (OUT/'reports/inventory.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps({'files':len(files),'merged_ft2':merged,'commit':env['git_commit']['output'].strip()}))
if __name__=='__main__':main()
