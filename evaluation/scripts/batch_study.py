"""Configurable serial GPU batch studies. All artifacts stay in output/evaluation."""
import argparse
import collections
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

from inventory import ROOT, OUT, sha

RUNNER = ROOT / 'evaluation/scripts/run.py'
BACKENDS = ('cuda', 'rt-instanced', 'rt-custom')
METRICS = ('fused_seconds', 'rt_setup_seconds', 'complete_scoring_seconds',
           'search_seconds', 'wall_seconds', 'device_peak_bytes', 'host_peak_bytes')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id', required=True, help='Unique study ID; reuse only with --resume')
    p.add_argument('--category', choices=['batch_size', 'scan_scaling'], default='batch_size')
    p.add_argument('--stop-at-utc', help='ISO-8601 UTC deadline: finish current child, then pause dispatch')
    p.add_argument('--scans', type=Path)
    p.add_argument('--fasta', type=Path)
    p.add_argument('--config', type=Path, default=ROOT / 'experiments/Regular.cfg')
    p.add_argument('--gpu-binary', type=Path, default=OUT / 'build/gpu/bin/sipros_mvh_cuda')
    p.add_argument('--batches', type=int, nargs='+')
    p.add_argument('--backends', nargs='+', choices=BACKENDS, default=['cuda', 'rt-custom'])
    p.add_argument('--precursor-mode', choices=['primary', 'multi'], default='primary', help='Multi requires a mode-verified binary from prepare_batch_binary.py')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--warmups', type=int, default=0, help='Excluded processes per cell before its first measured repetition')
    p.add_argument('--order', choices=['ascending', 'round-robin'], default='ascending')
    p.add_argument('--failure-policy', choices=['stop', 'continue', 'skip-larger'], default='stop',
                   help='stop: preserve pending jobs; continue: skip failed cell; skip-larger: cancel this and larger batches')
    p.add_argument('--min-device-free-gib', type=float, default=3)
    p.add_argument('--max-host-rss-gib', type=float, default=0, help='0 disables the host RSS guard')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--resume', action='store_true', help='Resume saved settings; never overwrite completed runs')
    p.add_argument('--retry-failed', action='store_true', help='With --resume, retry failures in new attempt directories')
    p.add_argument('--dry-run', action='store_true', help='Write a plan/report without launching measurements')
    p.add_argument('--authorize-paused', action='store_true', help='Explicitly authorize only this study when the legacy pause guard exists')
    return p


def make_jobs(settings):
    batches = sorted(settings['batches'])
    cells = []
    if settings['order'] == 'ascending':
        for batch in batches:
            for repeat in range(1, settings['repeats'] + 1):
                cells.append((batch, repeat))
    else:
        for repeat in range(1, settings['repeats'] + 1):
            shift = (repeat - 1) % len(batches)
            cells.extend((b, repeat) for b in batches[shift:] + batches[:shift])
    jobs = []
    for batch, repeat in cells:
        backends = settings['backends']
        shift = (repeat - 1) % len(backends)
        for backend in backends[shift:] + backends[:shift]:
            jobs.append({'key': f'b{batch}_{backend}_r{repeat}', 'batch': batch,
                         'backend': backend, 'repeat': repeat, 'status': 'pending', 'attempts': []})
    return jobs


def extract_metrics(run):
    batches = run.get('batches', [])
    if not batches or any(int(b.get('cached_ions', -1)) != 0 or
                          int(b.get('cache_charge_stride', -1)) != 0 for b in batches):
        raise ValueError('Cache-off baseline violated or counters missing')
    totals = run['totals']
    return {'fused_seconds': totals['fused_device_seconds'],
            'rt_setup_seconds': run['rt_setup_seconds'],
            'complete_scoring_seconds': sum(totals[k] for k in
                ('pack_seconds', 'gpu_service_seconds', 'replay_verify_seconds')),
            'search_seconds': float(run['summary']['search_seconds']),
            'wall_seconds': run['wall_seconds'],
            'device_peak_bytes': run['device_peak_bytes'],
            'host_peak_bytes': run['host_observed_hwm_bytes'],
            'generated': run['generated_peptides'], 'candidates': int(totals['candidates']),
            'psm_sha256': run['psm_sha256']}


def report(directory, manifest):
    rows = []
    for batch in sorted(manifest['settings']['batches']):
        for backend in manifest['settings']['backends']:
            jobs = [j for j in manifest['jobs'] if j['batch'] == batch and j['backend'] == backend]
            done = [j['metrics'] for j in jobs if j['status'] == 'completed']
            row = {'batch': batch, 'backend': backend, 'completed_repeats': len(done),
                   'requested_repeats': manifest['settings']['repeats'],
                   'failed': sum(j['status'] == 'failed' for j in jobs),
                   'skipped': sum(j['status'] == 'skipped' for j in jobs)}
            for metric in METRICS:
                values = [d[metric] for d in done]
                if values:
                    for name, value in [('median', statistics.median(values)), ('mean', statistics.mean(values)),
                                        ('min', min(values)), ('max', max(values)),
                                        ('stdev', statistics.stdev(values) if len(values) > 1 else None)]:
                        row[metric + '_' + name] = value
            if done:
                row['repeat_psm_identical'] = len({d['psm_sha256'] for d in done}) == 1
                row['generated'] = done[0]['generated']; row['candidates'] = done[0]['candidates']
            rows.append(row)
    for name, data in [('summary.csv', rows), ('measurements.csv', [
            dict(batch=j['batch'], backend=j['backend'], repeat=j['repeat'], **j['metrics'])
            for j in manifest['jobs'] if j['status'] == 'completed'])]:
        path = directory / name; temp = path.with_suffix('.tmp')
        with temp.open('w', newline='') as f:
            keys = list(dict.fromkeys(k for row in data for k in row))
            w = csv.DictWriter(f, fieldnames=keys or ['batch', 'backend', 'repeat'])
            w.writeheader(); w.writerows(data)
        temp.replace(path)
    counts = collections.Counter(j['status'] for j in manifest['jobs'])
    lines = ['# Batch study: ' + manifest['id'], '', 'Updated UTC: ' + utc(), '',
             'Status: ' + manifest['status'] + '; jobs: ' + json.dumps(dict(counts)), '',
             'Settings: `' + json.dumps(manifest['settings']) + '`', '',
             '| Batch | Backend | Completed | Fused median (s) | Complete scoring median (s) | Search median (s) | GPU peak median (GiB) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        def value(k, scale=1):
            return f"{r[k] / scale:.4f}" if k in r else '—'
        lines.append(f"| {r['batch']} | {r['backend']} | {r['completed_repeats']}/{r['requested_repeats']} | "
                     f"{value('fused_seconds_median')} | {value('complete_scoring_seconds_median')} | "
                     f"{value('search_seconds_median')} | {value('device_peak_bytes_median', 2**30)} |")
    for j in manifest['jobs']:
        if j['status'] == 'running':
            a = j['attempts'][-1]
            lines += ['', 'Active job: `' + j['key'] + '`; attempt ' + str(len(j['attempts'])),
                      'Child result directory: `' + a['result_directory'] + '`']
            child = Path(a['result_directory']) / 'manifest.json'
            try:
                n = json.loads(child.read_text())
                lines.append('Child processes: ' + json.dumps([(r['name'], r['status']) for r in n['runs']]))
                for r in n['runs']:
                    if r['status'] != 'running': continue
                    log = child.parent / (r['name'] + '.log')
                    if log.exists():
                        count = sum(x.startswith('[CUDA scoring]') for x in log.read_text(errors='replace').splitlines())
                        lines.append(f'Completed scoring batches in active process: {count} (partial).')
                    memory = child.parent / (r['name'] + '.memory.jsonl')
                    if memory.exists():
                        with memory.open('rb') as f:
                            f.seek(max(0, memory.stat().st_size - 8192)); tail = f.read().splitlines()
                        sample = json.loads(tail[-1]) if tail else {}
                        lines.append('Latest saved memory sample: `' + json.dumps(sample) + '`')
            except (OSError, ValueError, KeyError): pass
    if manifest.get('error'): lines += ['', 'Last error: ' + manifest['error']]
    lines += ['', 'Complete scoring = host packing + GPU service + host restoration; RT setup is already included.',
              'Only validated completed measurements enter the tables; warmups, failed and interrupted attempts are excluded.',
              'GPU memory is sampled device-wide usage, not process-only usage. Resource guards cannot catch every transient allocation.',
              'The binary determines precursor semantics; --precursor-mode checks mode metadata and does not rewrite the reader.',
              'Compilation caches are distinct from the disabled theoretical-ion cache. No profiler is attached.',
              'All detailed native logs/configurations/PSMs are retained in the child result directories.',
              'To pause safely, create `PAUSE` here. The current child case finishes before dispatch stops.', '']
    temp = directory / 'progress.tmp'; temp.write_text('\n'.join(lines)); temp.replace(directory / 'progress.md')


def fingerprints(settings):
    return {k: sha(Path(settings[k])) for k in ('scans', 'fasta', 'config', 'gpu_binary')} | {'runner': sha(RUNNER)}


def child_command(manifest, job, attempt):
    s = manifest['settings']; rid = f"{manifest['id']}_{job['key']}_a{attempt}"
    cmd = [sys.executable, '-B', str(RUNNER), '--id', rid, '--category', s.get('category', 'batch_size'),
           '--scans', s['scans'], '--fasta', s['fasta'], '--config', s['config'],
           '--gpu-binary', s['gpu_binary'], '--batch', str(job['batch']), '--backends', job['backend'],
           '--repeats', '1', '--warmups', str(s['warmups'] if job['repeat'] == 1 else 0),
           '--threads', str(s['threads']), '--min-device-free-gib', str(s['min_device_free_gib']),
           '--max-host-rss-gib', str(s['max_host_rss_gib'])]
    return rid, cmd


def authorize(manifest, job, rid, allowed):
    p = OUT / 'manifests/LARGE_BENCHMARKS_PAUSED.json'
    if not p.exists(): return
    if not allowed: raise RuntimeError('Legacy pause guard exists. Use --authorize-paused to authorize only this explicit study.')
    n = json.loads(p.read_text()); s = manifest['settings']
    parameters = {k: s[k] for k in ('scans', 'fasta', 'config', 'gpu_binary', 'threads')}
    parameters.update(category=s.get('category', 'batch_size'), backends=[job['backend']], batch=job['batch'], repeats=1,
                      warmups=s['warmups'] if job['repeat'] == 1 else 0)
    n.setdefault('authorized_runs', {})[rid] = {'parameters': parameters, 'request': 'Explicit batch_study invocation', 'study': manifest['id']}
    save(p, n)


def main(argv=None):
    p = parser(); arguments = sys.argv[1:] if argv is None else argv; a = p.parse_args(arguments)
    if a.resume:
        immutable = {'--scans', '--fasta', '--config', '--gpu-binary', '--batches', '--backends',
                     '--repeats', '--warmups', '--order', '--failure-policy', '--threads', '--precursor-mode',
                     '--min-device-free-gib', '--max-host-rss-gib', '--category'}
        if any(token.split('=')[0] in immutable for token in arguments):
            p.error('Resume uses saved settings; changing experiment options requires a new study ID')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', a.id) or a.id in ('.', '..'): p.error('Invalid study ID')
    if a.stop_at_utc:
        try:
            a.deadline = datetime.datetime.fromisoformat(a.stop_at_utc.replace('Z', '+00:00'))
            if a.deadline.tzinfo is None: raise ValueError('Timezone required')
        except ValueError: p.error('--stop-at-utc must be an ISO-8601 timestamp with timezone')
    else: a.deadline = None
    if a.retry_failed and not a.resume: p.error('--retry-failed requires --resume')
    directory = OUT / 'batch_studies' / a.id
    (OUT / 'batch_studies').mkdir(parents=True, exist_ok=True)
    with (OUT / 'batch_studies/.coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute_study(a, p, directory)


def execute_study(a, p, directory):
    if a.resume:
        if a.scans or a.fasta or a.batches: p.error('Resume uses saved settings; omit --scans/--fasta/--batches')
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['fingerprints'] != fingerprints(manifest['settings']): p.error('Input/binary/runner changed; create a new study')
        for job in manifest['jobs']:
            if job['status'] == 'running' and job['attempts']:
                pid = job['attempts'][-1].get('runner_pid')
                try:
                    command_line = Path(f'/proc/{pid}/cmdline').read_bytes()
                except OSError:
                    command_line = b''
                if job['attempts'][-1]['id'].encode() in command_line:
                    p.error('Previous child runner is still alive; wait or inspect it before resuming')
            if job['status'] in ('running', 'interrupted') or (a.retry_failed and job['status'] == 'failed'):
                if job['attempts']: job['attempts'][-1].setdefault('recovered_as', 'interrupted' if job['status'] in ('running', 'interrupted') else 'failed')
                job['status'] = 'pending'
    else:
        if not a.scans or not a.fasta or not a.batches: p.error('New studies require --scans, --fasta and --batches')
        if min(a.batches) < 1 or len(set(a.batches)) != len(a.batches): p.error('Batch values must be positive and unique')
        if a.repeats < 1 or a.warmups < 0 or a.threads < 1 or min(a.min_device_free_gib, a.max_host_rss_gib) < 0: p.error('Invalid repetition/resource settings')
        if len(set(a.backends)) != len(a.backends): p.error('Duplicate backends')
        s = {k: getattr(a, k) for k in ('batches', 'backends', 'repeats', 'warmups', 'order', 'failure_policy', 'threads', 'min_device_free_gib', 'max_host_rss_gib', 'precursor_mode', 'category')}
        for k in ('scans', 'fasta', 'config', 'gpu_binary'):
            path = getattr(a, k).resolve()
            if not path.is_file(): p.error('Missing input: ' + str(path))
            s[k] = str(path)
        mode_path = Path(s['gpu_binary']).parents[1] / 'build_manifest.json'
        if mode_path.exists():
            mode = json.loads(mode_path.read_text())
            if (mode.get('status') != 'completed' or mode.get('precursor_mode') != a.precursor_mode
                    or mode.get('ion_cache') != 'compiled_off' or mode.get('binary_sha256') != sha(Path(s['gpu_binary']))):
                p.error('Binary mode/hash/cache metadata mismatch; rebuild or select the correct binary')
        elif a.precursor_mode == 'multi':
            p.error('Multi mode requires a verified binary built by prepare_batch_binary.py')
        directory.mkdir(exist_ok=False)
        manifest = {'id': a.id, 'status': 'planned', 'created_utc': utc(), 'settings': s,
                    'fingerprints': fingerprints(s), 'jobs': make_jobs(s)}
    mp = directory / 'manifest.json'
    def checkpoint(): save(mp, manifest); report(directory, manifest)
    checkpoint()
    if a.dry_run:
        print(directory / 'progress.md'); return 0
    if (directory / 'PAUSE').exists():
        manifest['status'] = 'paused'; checkpoint(); return 0
    stop_requested = False
    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True  # Do not interrupt the child measurement.
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
    for key in ('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR', 'SIPROS_DIAGNOSTIC_HOT_REPLAY', 'SIPROS_CONTINUOUS_REPLAY', 'SIPROS_PEAK_AUDIT_DIR'):
        env.pop(key, None)
    try:
        manifest.update(status='running', coordinator_pid=os.getpid(), stop_at_utc=a.stop_at_utc); manifest.pop('error', None); checkpoint()
        for job in manifest['jobs']:
            if job['status'] != 'pending': continue
            if stop_requested or (directory / 'PAUSE').exists() or (a.deadline and datetime.datetime.now(datetime.timezone.utc) >= a.deadline):
                manifest['status'] = 'paused'; checkpoint(); return 0
            if any(j['status'] == 'failed' and j['batch'] == job['batch'] and j['backend'] == job['backend'] for j in manifest['jobs']):
                job.update(status='skipped', reason='Earlier repetition of this cell failed'); checkpoint(); continue
            if manifest['fingerprints'] != fingerprints(manifest['settings']):
                raise RuntimeError('Input/binary/runner changed during study; create a new study')
            rid, cmd = child_command(manifest, job, len(job['attempts']) + 1)
            authorize(manifest, job, rid, a.authorize_paused)
            attempt = {'id': rid, 'command': cmd, 'started_utc': utc(),
                       'result_directory': str(OUT / 'results' / manifest['settings'].get('category', 'batch_size') / rid)}
            job['attempts'].append(attempt); job['status'] = 'running'; checkpoint()
            print('START ' + rid, flush=True)
            with (directory / (rid + '.log')).open('w') as log:
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                attempt['runner_pid'] = child.pid; checkpoint()
                while child.poll() is None:
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: checkpoint()
            attempt.update(exit_code=child.returncode, finished_utc=utc())
            try:
                n = json.loads((Path(attempt['result_directory']) / 'manifest.json').read_text())
                if child.returncode or n['status'] != 'completed': raise ValueError(n.get('error', 'Child case failed; inspect logs'))
                if n.get('diagnostic'): raise ValueError('Diagnostic/profiling build cannot supply baseline timings')
                runs = [r for r in n['runs'] if r['status'] == 'completed' and not r['warmup']]
                if len(runs) != 1: raise ValueError('Expected one validated measured process')
                metrics = extract_metrics(runs[0])
                counts = {k: metrics[k] for k in ('generated', 'candidates')}
                if manifest.setdefault('workload_counts', counts) != counts: raise ValueError('Workload totals changed across configurations')
                if any(j['status'] == 'completed' and j['backend'] == job['backend'] and j['metrics']['psm_sha256'] != metrics['psm_sha256'] for j in manifest['jobs']):
                    raise ValueError('Same-backend PSM changed across batches/repetitions')
                job.update(status='completed', metrics=metrics); print('DONE ' + rid, flush=True)
            except (OSError, ValueError, KeyError) as error:
                detail = str(error)
                if not (Path(attempt['result_directory']) / 'manifest.json').exists():
                    runner_log = directory / (rid + '.log')
                    tail = '\n'.join(runner_log.read_text(errors='replace').splitlines()[-20:]) if runner_log.exists() else ''
                    detail = f'Child runner exited {child.returncode} before manifest creation. Runner log: {runner_log}\n{tail}'
                job.update(status='failed', error=detail); manifest['error'] = rid + ': ' + detail
                # Methodology violations always stop, regardless of resource-failure policy.
                violation = any(x in str(error) for x in ('Cache-off', 'counters missing', 'Workload totals', 'PSM changed', 'Diagnostic/'))
                policy = manifest['settings']['failure_policy']
                if policy == 'skip-larger' and not violation:
                    for pending in manifest['jobs']:
                        if pending['status'] == 'pending' and pending['batch'] >= job['batch']:
                            pending.update(status='skipped', reason='Capacity failure at batch ' + str(job['batch']))
                checkpoint()
                if policy == 'stop' or violation:
                    manifest['status'] = 'failed'; checkpoint(); return 1
            checkpoint()
        manifest['status'] = 'completed_with_failures' if any(j['status'] == 'failed' for j in manifest['jobs']) else 'completed'
        manifest['finished_utc'] = utc(); checkpoint()
        return int(manifest['status'] != 'completed')
    except BaseException as error:
        manifest.update(status='stopped_with_error', error=repr(error)); checkpoint(); raise
    finally:
        for sig, handler in old.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
