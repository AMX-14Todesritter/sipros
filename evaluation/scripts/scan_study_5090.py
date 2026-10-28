"""Time-bounded Marine/X1-X3 multi-precursor scan scaling for the 5090."""
import argparse
import csv
import datetime
import fcntl
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

from inventory import ROOT, OUT, sha
from batch_study import save, utc

POINTS = [20000, 40000, 60000, 80000, 100000, 120000, 140297]
SOURCES = [OUT / f'inputs/ecoli_x{x}/Pan_062822_X{x}iso5.FT2' for x in (1, 2, 3)]


def report(d, m):
    records = []
    for job in m['jobs']:
        p = OUT / 'batch_studies' / job['id'] / 'measurements.csv'
        if p.exists():
            with p.open() as f:
                records.extend(dict(scan_count=job['scans'], round=job['round'], child_study=job['id'], **r)
                               for r in csv.DictReader(f))
    lines = ['# 5090 Marine scan scaling', '', 'Status: ' + m['status'],
             'Updated UTC: ' + utc(), 'Dispatch deadline UTC: ' + m.get('deadline_utc', 'not started'), '',
             'Multi-precursor, sm120, theoretical-ion cache off. Fixed Marine database, batch ' + str(m['batch']) + '.',
             'Nested source-stratified X1/X2/X3 samples; rounds rotate sample/backend order. No CPU or profiler.', '',
             '| Scans | Backend | Completed | Median fused (s) | Median complete scoring (s) | Median search (s) | Max GPU (GiB) |',
             '|---|---|---:|---:|---:|---:|---:|']
    summary = []
    for n in POINTS:
        for b in m['backends']:
            rs = [r for r in records if int(r['scan_count']) == n and r['backend'] == b]
            row = {'scan_count': n, 'backend': b, 'completed_repeats': len(rs)}
            if rs:
                row['workload_identical'] = len({(r['generated'], r['candidates']) for r in rs}) == 1
            for k in ('fused_seconds', 'complete_scoring_seconds', 'search_seconds', 'device_peak_bytes', 'host_peak_bytes'):
                vs = [float(r[k]) for r in rs]
                if vs:
                    row.update({k+'_median': statistics.median(vs), k+'_mean': statistics.mean(vs),
                                k+'_min': min(vs), k+'_max': max(vs),
                                k+'_stdev': statistics.stdev(vs) if len(vs)>1 else None})
            if rs: row['repeat_psm_identical'] = len({r['psm_sha256'] for r in rs}) == 1
            summary.append(row)
            val = lambda k: f'{row[k]:.3f}' if k in row else '—'
            memory = f"{row['device_peak_bytes_max']/2**30:.2f}" if rs else '—'
            lines.append(f"| {n} | {b} | {len(rs)}/{m['repeats']} | {val('fused_seconds_median')} | {val('complete_scoring_seconds_median')} | {val('search_seconds_median')} | {memory} |")
    for j in m['jobs']:
        if j['status'] in ('running','failed','paused'):
            lines += ['', f"{j['status']}: {j['id']}; see `../../batch_studies/{j['id']}/progress.md`."]
    lines += ['', 'Deadline/PAUSE finishes the current native measurement before stopping dispatch; it is not a hard shutdown timer.',
              'Failed/partial measurements and warmups are excluded. Guards sample whole-device memory and cannot prevent every transient OOM.',
              'Create PAUSE in this directory to stop after the active child study. To pause before the next backend, use PAUSE in the active child study directory.',
              'Nested sample maps preserve source and original IDs; only S-line scan IDs are remapped, with peak data unchanged.', '']
    for name, rows in [('measurements.csv', records), ('summary.csv', summary)]:
        temp = d / (name + '.tmp')
        with temp.open('w', newline='') as f:
            keys = list(dict.fromkeys(k for row in rows for k in row))
            w = csv.DictWriter(f, fieldnames=keys or ['scan_count','backend']); w.writeheader(); w.writerows(rows)
        temp.replace(d / name)
    temp = d / 'progress.tmp'; temp.write_text('\n'.join(lines)); temp.replace(d / 'progress.md')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id',required=True); p.add_argument('--hours',type=float,default=6.5)
    p.add_argument('--repeats',type=int,default=3); p.add_argument('--batch',type=int,default=4000000)
    p.add_argument('--backends',nargs='+',choices=['cuda','rt-instanced','rt-custom'],default=['cuda','rt-instanced','rt-custom'])
    p.add_argument('--resume',action='store_true'); p.add_argument('--retry-failed',action='store_true',help='Explicitly retry inspected failures in new attempt directories'); a=p.parse_args()
    if a.retry_failed and not a.resume:p.error('--retry-failed requires --resume')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,45}',a.id):p.error('Invalid study ID')
    if a.hours<=0 or a.repeats<1 or a.batch<1:p.error('Positive hours/repeats/batch required')
    if len(set(a.backends))!=len(a.backends):p.error('Duplicate backends')
    d=OUT/'scan_studies'/a.id;d.parent.mkdir(parents=True,exist_ok=True)
    with (d.parent/'.coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(d,a,p)


def execute(d,a,p):
    binary=OUT/'build/gpu_multi/bin/sipros_mvh_cuda';meta=binary.parents[1]/'build_manifest.json'
    for f in [*SOURCES, ROOT/'raw/Marine_fw_3rev.fasta',ROOT/'experiments/Regular.cfg',binary,meta]:
        if not f.is_file():p.error('Missing input: '+str(f))
    bm=json.loads(meta.read_text())
    cache=(binary.parents[1]/'CMakeCache.txt').read_text()
    if (bm['status']!='completed' or bm['precursor_mode']!='multi' or bm['ion_cache']!='compiled_off'
        or bm['binary_sha256']!=sha(binary) or not re.search(r'^CMAKE_CUDA_ARCHITECTURES:[^=]+=120$',cache,re.M)):
        p.error('Verified sm120/cache-off/multi binary required at build/gpu_multi')
    fingerprints={str(f):sha(f) for f in [*SOURCES,ROOT/'raw/Marine_fw_3rev.fasta',ROOT/'experiments/Regular.cfg',binary]}
    if a.resume:
        m=json.loads((d/'manifest.json').read_text())
        if fingerprints!=m['fingerprints']:p.error('Inputs or binary changed; use a new ID')
        for j in m['jobs']:
            if j['status'] in ('running','paused') or (a.retry_failed and j['status']=='failed'):j['status']='pending'
        if any(j['status']=='failed' for j in m['jobs']):p.error('Failed child requires inspection before starting a new study')
    else:
        d.mkdir(exist_ok=False)
        jobs=[]
        for r in range(1,a.repeats+1):
            shift=(r-1)%len(POINTS); points=POINTS[shift:]+POINTS[:shift]
            bshift=(r-1)%len(a.backends);backends=a.backends[bshift:]+a.backends[:bshift]
            for n in points:jobs.append({'id':f'{a.id}_s{n}_r{r}','scans':n,'round':r,'backends':backends,'status':'pending'})
        m={'status':'preparing','created_utc':utc(),'batch':a.batch,'repeats':a.repeats,
           'backends':a.backends,'fingerprints':fingerprints,'jobs':jobs}
    def checkpoint(): save(d/'manifest.json',m);report(d,m)
    checkpoint()
    # Preparation is separate from the 6.5-hour measurement dispatch budget.
    seed=20261028;inp=OUT/f'inputs/scan_scaling_seed{seed}'
    if not inp.exists():
        cmd=[sys.executable,'-B',str(ROOT/'evaluation/scripts/scaling_inputs.py'),'scans','--seed',str(seed),
             '--points',*[str(n) for n in POINTS],'--sources',*[str(f) for f in SOURCES]]
        with (d/'prepare.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    inputs=json.loads((inp/'manifest.json').read_text())
    if inputs['status']!='completed' or inputs['source_sha256']!={f'ecoli_x{i}':fingerprints[str(f)] for i,f in enumerate(SOURCES,1)}:
        raise RuntimeError('Nested input manifest invalid or source hashes differ')
    for n in POINTS:
        path=inp/f'scans_{n}.FT2'
        if sha(path)!=inputs['outputs'][str(n)]['sha256']:raise RuntimeError('Nested sample changed')
    deadline=datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=a.hours)
    m.update(status='running',deadline_utc=deadline.isoformat());checkpoint()
    for j in m['jobs']:
        if j['status']=='completed':continue
        if datetime.datetime.now(datetime.timezone.utc)>=deadline or (d/'PAUSE').exists():
            m['status']='paused';checkpoint();return
        childdir=OUT/'batch_studies'/j['id']
        cmd=[sys.executable,'-B',str(ROOT/'evaluation/scripts/batch_study.py'),'--id',j['id'],
             '--stop-at-utc',deadline.isoformat()]
        if (childdir/'manifest.json').exists():
            cmd+=['--resume']
            if a.retry_failed:cmd+=['--retry-failed']
        else:cmd+=['--category','scan_scaling','--precursor-mode','multi','--scans',str(inp/f"scans_{j['scans']}.FT2"),
                   '--fasta',str(ROOT/'raw/Marine_fw_3rev.fasta'),'--config',str(ROOT/'experiments/Regular.cfg'),
                   '--gpu-binary',str(binary),'--batches',str(m['batch']),'--backends',*j['backends'],
                   '--repeats','1','--warmups','0','--threads','1','--min-device-free-gib','3','--max-host-rss-gib','55','--failure-policy','stop']
        j.update(status='running',command=cmd);checkpoint();print('START '+j['id'],flush=True)
        with (d/(j['id']+'.log')).open('a') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:checkpoint()
        n=json.loads((childdir/'manifest.json').read_text()) if (childdir/'manifest.json').exists() else {}
        j['status']=n.get('status','failed');checkpoint()
        if child.returncode or j['status'] not in ('completed','paused'):
            j['status']='failed';m['status']='failed';checkpoint();raise RuntimeError('Child failed; no more experiments dispatched')
        if j['status']=='paused':m['status']='paused';checkpoint();return
        summary=report(d,m)
        for row in summary:
            if row.get('repeat_psm_identical') is False or row.get('workload_identical') is False:
                m.update(status='failed',error='Same-backend PSM or workload changed across rounds');checkpoint();raise RuntimeError(m['error'])
    m.update(status='completed',finished_utc=utc());checkpoint()


if __name__=='__main__':main()
