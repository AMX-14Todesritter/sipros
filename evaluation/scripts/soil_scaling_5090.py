"""Unattended Soil scaling: X1+X2, multi, sm120, cache off; no time deadline."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

from inventory import ROOT, OUT, sha
from prepare import records
from batch_study import save, utc
def read_json(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}

PERCENTAGES = [1, 5, 10, 25, 50, 100]
BACKENDS = ['cuda', 'rt-custom']
SOURCE_COUNTS = [46066, 46862]


def prepare_x1_x2(sources, destination):
    """Concatenate physical scans, remapping only S-line IDs; preserve provenance."""
    source_hashes = {str(p): sha(p) for p in sources}
    manifest_path = destination / 'manifest.json'
    if destination.exists():
        m = read_json(manifest_path)
        if (m.get('status') != 'completed' or m.get('source_sha256') != source_hashes
                or m.get('scans') != sum(SOURCE_COUNTS)
                or m.get('sha256') != sha(destination / 'scans_92928.FT2')
                or m.get('map_sha256') != sha(destination / 'scan_map.csv')):
            raise RuntimeError('Existing X1+X2 input is incomplete or changed; inspect before retrying')
        return destination / 'scans_92928.FT2'
    destination.mkdir(parents=True)
    m = {'status': 'preparing', 'source_sha256': source_hashes,
         'method': 'All X1 followed by all X2; globally unique S IDs; other record bytes preserved'}
    save(manifest_path, m)
    try:
        header = []
        with sources[0].open('rb') as f:
            for line in f:
                if line.startswith((b'S\t', b'S ')): break
                header.append(line)
        total = 0; counts = []
        with (destination / 'scans_92928.FT2').open('wb') as out, (destination / 'scan_map.csv').open('w', newline='') as mapping:
            out.write(b''.join(header)); writer = csv.writer(mapping)
            writer.writerow(['global_scan_id', 'source', 'original_scan_id', 'original_record_sha256', 'remapped_record_sha256'])
            for path, expected in zip(sources, SOURCE_COUNTS):
                seen = set(); count = 0
                for record in records(path):
                    sid = record.splitlines()[0].split()[1].decode()
                    if sid in seen: raise RuntimeError('Duplicate source scan ID')
                    seen.add(sid); count += 1; total += 1
                    remapped = re.sub(rb'^(S[ \t]+)\S+', lambda _: b'S\t' + str(total).encode(), record, count=1)
                    # Only the first S-line ID/whitespace prefix changes; all other lines are identical.
                    normalize = lambda b: re.sub(rb'^S[ \t]+\S+', b'S\tID', b, count=1)
                    if normalize(record) != normalize(remapped): raise RuntimeError('Peak/precursor record changed')
                    out.write(remapped)
                    writer.writerow([total, str(path), sid, hashlib.sha256(record).hexdigest(), hashlib.sha256(remapped).hexdigest()])
                if count != expected: raise RuntimeError(f'Unexpected source count: {path}: {count}')
                counts.append(count)
        m.update(status='completed', scans=total, source_counts=counts,
                 sha256=sha(destination / 'scans_92928.FT2'), map_sha256=sha(destination / 'scan_map.csv'))
        save(manifest_path, m)
        return destination / 'scans_92928.FT2'
    except BaseException as error:
        m.update(status='failed', error=repr(error)); save(manifest_path, m); raise


def report(directory, m):
    measured = []; details = []
    states = {key: 0 for key in ('completed', 'running', 'pending', 'failed', 'skipped')}
    for job in m['jobs']:
        childdir = OUT / 'batch_studies' / job['id']
        child = read_json(childdir / 'manifest.json')
        childjobs = {j['backend']: j for j in child.get('jobs', [])}
        csv_path = childdir / 'measurements.csv'
        if csv_path.exists():
            with csv_path.open() as f:
                measured.extend(dict(soil_percent=job['percent'], round=job['round'], child_study=job['id'], **r) for r in csv.DictReader(f))
        for backend in job['backends']:
            j = childjobs.get(backend, {}); status = j.get('status', 'pending')
            states[status] = states.get(status, 0) + 1
            details.append(f"| {job['round']} | {job['percent']}% | {backend} | {status} | {len(j.get('attempts', []))} |")
    lines = ['# 5090 Soil database scaling', '', 'Status: ' + m['status'], 'Updated UTC: ' + utc(), '',
             f"Completed measured processes: {len(measured)}/{len(m['jobs'])*2}; states: `{json.dumps(states)}`", '',
             'All 92,928 X1+X2 spectra; multi-precursor; sm120; theoretical-ion cache off.',
             'Fixed 8M generated-peptide batch. CUDA and RT-custom only. No time deadline, CPU or profiler.',
             'Rounds rotate database/backend order. No additional warmup processes.',
             'Guards: GPU free 3 GiB; host RSS 55 GiB; output free 10 GiB before each process.', '',
             '| Soil | Backend | Completed | Fused median (s) | Complete scoring median (s) | Search median (s) | Max GPU (GiB) |',
             '|---|---|---:|---:|---:|---:|---:|']
    summary = []
    for pct in PERCENTAGES:
        for backend in BACKENDS:
            rs = [r for r in measured if int(r['soil_percent']) == pct and r['backend'] == backend]
            row = {'soil_percent': pct, 'backend': backend, 'completed_repeats': len(rs)}
            if rs:
                row['workload_identical'] = len({(r['generated'], r['candidates']) for r in rs}) == 1
                row['repeat_psm_identical'] = len({r['psm_sha256'] for r in rs}) == 1
            for metric in ('fused_seconds', 'rt_setup_seconds', 'complete_scoring_seconds', 'search_seconds', 'device_peak_bytes', 'host_peak_bytes'):
                values = [float(r[metric]) for r in rs]
                if values:
                    row.update({metric+'_median': statistics.median(values), metric+'_mean': statistics.mean(values),
                                metric+'_min': min(values), metric+'_max': max(values),
                                metric+'_stdev': statistics.stdev(values) if len(values) > 1 else None})
            summary.append(row)
            val = lambda k: f'{row[k]:.3f}' if k in row else '—'
            memory = f"{row['device_peak_bytes_max']/2**30:.2f}" if rs else '—'
            lines.append(f"| {pct}% | {backend} | {len(rs)}/{m['repeats']} | {val('fused_seconds_median')} | {val('complete_scoring_seconds_median')} | {val('search_seconds_median')} | {memory} |")
    lines += ['', '## Current task and errors', '']
    for job in m['jobs']:
        if job['status'] not in ('running', 'failed', 'paused'): continue
        childdir = OUT / 'batch_studies' / job['id']; child = read_json(childdir / 'manifest.json')
        lines += [f"Soil {job['percent']}%, round {job['round']}: {job['status']}."]
        active = childdir / 'progress.md'
        if active.exists(): lines.append(active.read_text())
        elif (directory / (job['id'] + '.log')).exists():
            lines += ['```text', '\n'.join((directory / (job['id']+'.log')).read_text(errors='replace').splitlines()[-20:]), '```']
    if m.get('error'): lines += ['', 'Last error: ' + m['error']]
    lines += ['', '## All rounds', '', '| Round | Soil | Backend | Latest status | Attempts |', '|---|---|---|---|---:|', *details, '',
              'Only validated successful measurements enter summaries. Identical PSM exports share disk blocks; all paths remain present.',
              'Percentage means a nested SHA256-threshold target/decoy-family fraction; actual entry counts are in the input manifest.',
              'Create PAUSE here for a safe stop after the current two-backend case. Use PAUSE in its child study to stop after the current backend.', '']
    for name, rows in [('measurements.csv', measured), ('summary.csv', summary)]:
        temporary = directory / (name+'.tmp')
        with temporary.open('w', newline='') as f:
            keys = list(dict.fromkeys(k for row in rows for k in row))
            w = csv.DictWriter(f, fieldnames=keys or ['soil_percent','backend']); w.writeheader(); w.writerows(rows)
        temporary.replace(directory / name)
    temporary = directory / 'progress.tmp'; temporary.write_text('\n'.join(lines)); temporary.replace(directory / 'progress.md')
    save(directory / 'manifest.json', m)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id', default='soil_5090_x1x2_8m_v1'); p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--resume', action='store_true'); p.add_argument('--retry-failed', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,45}', a.id) or a.repeats < 1: p.error('Invalid ID/repeats')
    if a.retry_failed and not a.resume: p.error('--retry-failed requires --resume')
    directory = OUT / 'soil_studies' / a.id; directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / '.coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); execute(directory, a, p)


def execute(directory, a, parser):
    sources = [OUT / f'inputs/ecoli_x{x}/Pan_062822_X{x}iso5.FT2' for x in (1, 2)]
    soil = ROOT / 'raw/Soil_fw_3rev.fasta'; config = ROOT / 'experiments/Regular.cfg'
    binary = OUT / 'build/gpu_multi/bin/sipros_mvh_cuda'; build = binary.parents[1]
    for f in [*sources, soil, config, binary, build/'build_manifest.json']:
        if not f.is_file(): parser.error('Missing input: ' + str(f))
    meta = read_json(build / 'build_manifest.json')
    if (meta.get('status') != 'completed' or meta.get('precursor_mode') != 'multi' or meta.get('ion_cache') != 'compiled_off'
            or meta.get('binary_sha256') != sha(binary)
            or not re.search(r'^CMAKE_CUDA_ARCHITECTURES:[^=]+=120$', (build/'CMakeCache.txt').read_text(), re.M)):
        parser.error('Verified sm120/multi/cache-off binary required')
    inputs = {str(f): sha(f) for f in [*sources, soil, config, binary, ROOT/'evaluation/scripts/run.py']}
    if a.resume:
        m = read_json(directory / 'manifest.json')
        if not m or m['fingerprints'] != inputs: parser.error('Saved inputs/binary/runner changed; new study required')
        if any(j['status'] == 'failed' for j in m['jobs']) and not a.retry_failed:
            parser.error('Inspect failure first; explicit --retry-failed is required')
        for job in m['jobs']:
            if job['status'] in ('running', 'paused') or (a.retry_failed and job['status'] == 'failed'): job['status'] = 'pending'
    else:
        directory.mkdir(exist_ok=False); jobs = []
        for r in range(1, a.repeats + 1):
            shift = (r-1) % len(PERCENTAGES)
            for pct in PERCENTAGES[shift:]+PERCENTAGES[:shift]:
                jobs.append({'id':f'{a.id}_p{pct:03d}_r{r}', 'percent':pct, 'round':r,
                             'backends':BACKENDS if r % 2 else BACKENDS[::-1], 'status':'pending'})
        m = {'status':'preparing','created_utc':utc(),'repeats':a.repeats,'batch':8000000,'fingerprints':inputs,'jobs':jobs}
    report(directory, m)
    try:
        print('Preparing/verifying all X1+X2 scans', flush=True)
        scans = prepare_x1_x2(sources, OUT/'inputs/x1_x2_full')
        subsetdir = OUT/'inputs/soil_nested_seed20261028'
        if not subsetdir.exists():
            with (directory/'prepare.log').open('w') as log:
                subprocess.run([sys.executable,'-B',str(ROOT/'evaluation/scripts/scaling_inputs.py'),'soil','--seed','20261028'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        preparation = read_json(subsetdir/'manifest.json')
        if preparation.get('status') != 'completed' or preparation['source_sha256'] != inputs[str(soil)]:
            raise RuntimeError('Soil preparation incomplete or source changed')
        subsets = {}
        for pct in PERCENTAGES:
            path = soil if pct == 100 else subsetdir/f'soil_{pct:03d}pct.fasta'
            if sha(path) != preparation['subsets'][str(pct)]['sha256']: raise RuntimeError('Subset hash mismatch')
            subsets[pct] = path
        m.update(status='running', scans_sha256=sha(scans), database_preparation=preparation); report(directory,m)
        for job in m['jobs']:
            if job['status']=='completed': continue
            if (directory/'PAUSE').exists(): m['status']='paused';report(directory,m);return
            childdir = OUT/'batch_studies'/job['id']
            cmd=[sys.executable,'-u','-B',str(ROOT/'evaluation/scripts/batch_study.py'),'--id',job['id']]
            if (childdir/'manifest.json').exists():
                cmd+=['--resume']
                if a.retry_failed:cmd+=['--retry-failed']
            else:
                cmd+=['--category','database_scaling','--precursor-mode','multi','--scans',str(scans),
                      '--fasta',str(subsets[job['percent']]),'--config',str(config),'--gpu-binary',str(binary),
                      '--batches','8000000','--backends',*job['backends'],'--repeats','1','--warmups','0',
                      '--threads','1','--min-device-free-gib','3','--max-host-rss-gib','55','--failure-policy','stop']
            job.update(status='running',command=cmd);report(directory,m);print('START '+job['id'],flush=True)
            with (directory/(job['id']+'.log')).open('a') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:report(directory,m)
            result=read_json(childdir/'manifest.json');job['status']=result.get('status','failed');job['exit_code']=child.returncode
            summary=report(directory,m)
            if child.returncode or job['status'] not in ('completed','paused'):
                job['status']='failed';raise RuntimeError(result.get('error','Child failed; inspect '+str(directory/(job['id']+'.log'))))
            if job['status']=='paused':m['status']='paused';report(directory,m);return
            if any(row.get('workload_identical') is False or row.get('repeat_psm_identical') is False for row in summary):
                raise RuntimeError('Workload or same-backend PSM changed across rounds')
            print('DONE '+job['id'],flush=True)
        m.update(status='completed',finished_utc=utc())
    except BaseException as error:
        m.update(status='failed',error=repr(error));raise
    finally:report(directory,m)


if __name__ == '__main__': main()
