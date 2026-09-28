"""Profile the unchanged CPU baseline with gperftools in the GPU container.

Sampling measures active CPU stacks, not exact per-function wall time. The
profiler is preloaded only into the search process; no allocator is replaced.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.output_paths import resolve_output


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['smoke', 'ecoli', 'marine'], default='marine')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--frequency', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.frequency <= 1000:
        parser.error('threads must be positive; frequency must be in 1..1000 Hz')
    binary = ROOT / 'build/mvh/bin/sipros_mvh'
    pprof = ROOT / 'build/tools/cpu_profile/pprof'
    library = Path('/usr/lib/x86_64-linux-gnu/libprofiler.so.0')
    smoke = args.dataset == 'smoke'
    scans = ROOT / ('mvh/tests/data/sample.ft2' if smoke else
                    'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2')
    config = ROOT / ('mvh/tests/data/search.cfg' if smoke else 'experiments/Regular.cfg')
    fasta = ROOT / ('mvh/tests/data/proteins.fasta' if smoke else
                    'raw/Marine_fw_3rev.fasta' if args.dataset == 'marine' else 'raw/Ecoli.fasta')
    for path in [binary, pprof, library, scans, config, fasta]:
        if not path.is_file():
            parser.error(f'Missing required file: {path}')
    output = resolve_output(args.output, 'benchmarks', 'cpu_profiling', args.dataset,
                            project_root=ROOT)
    output.mkdir(parents=True, exist_ok=False)
    command = [str(binary), '-f', str(scans), '-c', str(config), '-fasta', str(fasta),
               '-t', str(args.threads), '-o', str(output / 'results')]
    manifest = {
        'complete': False, 'command': command, 'threads': args.threads,
        'frequency_hz': args.frequency, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'sha256': {str(p): sha256(p) for p in [binary, scans, config, fasta, library, pprof]},
        'measurement': 'gperftools CPU-time stack sampling; cumulative rows overlap; '
                       '100 Hz is the requested default, not per-function exact timing. '
                       'External process memory/CPU sampling every second. No tcmalloc preload.',
        'steps': {},
    }
    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    child = None
    def interrupt(signum, frame):
        manifest['interrupted'] = True
        if child is not None and child.poll() is None:
            child.terminate()
        raise KeyboardInterrupt
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, interrupt)
    save()
    print('OUTPUT:', output, flush=True)
    try:
        env = os.environ.copy()
        env['LD_PRELOAD'] = str(library)
        env['CPUPROFILE'] = str(output / 'cpu.prof')
        env['CPUPROFILE_FREQUENCY'] = str(args.frequency)
        # CPU-time sampling is intentional; do not inherit a realtime timer mode.
        env.pop('CPUPROFILE_REALTIME', None)
        env.pop('CPUPROFILESIGNAL', None)
        ticks = os.sysconf('SC_CLK_TCK')
        started = time.monotonic()
        peak_hwm = 0
        last_cpu = 0.0
        with (output / 'search.log').open('w') as log, (output / 'process_samples.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['elapsed_seconds', 'cpu_seconds', 'rss_kib', 'hwm_kib', 'threads'])
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            manifest['pid'] = child.pid
            save()
            while child.poll() is None:
                try:
                    proc = Path(f'/proc/{child.pid}')
                    status = dict(line.split(':', 1) for line in (proc/'status').read_text().splitlines() if ':' in line)
                    stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
                    last_cpu = (int(stat[11]) + int(stat[12])) / ticks
                    hwm = int(status.get('VmHWM', '0').split()[0])
                    peak_hwm = max(peak_hwm, hwm)
                    writer.writerow([round(time.monotonic()-started, 3), last_cpu,
                                     status.get('VmRSS', '0').split()[0], hwm,
                                     status.get('Threads', '0').strip()])
                    stream.flush()
                except (OSError, ValueError, IndexError):
                    pass  # Process can exit between polling and reading /proc.
                time.sleep(1)
        manifest.update(exit_code=child.returncode, wall_seconds=time.monotonic()-started,
                        observed_cpu_seconds=last_cpu, observed_hwm_kib=peak_hwm)
        save()
        if child.returncode:
            raise RuntimeError(f'Search failed: exit {child.returncode}; see search.log')
        manifest['psm_sha256'] = sha256(output/'results/mvh_psms.tsv')
        for name, flags in [('flat.txt', ['--text']), ('cumulative.txt', ['--text', '--cum']),
                            ('stacks.folded', ['--collapsed']), ('callgrind.out', ['--callgrind'])]:
            with (output/name).open('w') as report, (output/(name+'.log')).open('w') as errors:
                result = subprocess.run(['perl', str(pprof), *flags, str(binary), str(output/'cpu.prof')],
                                        stdout=report, stderr=errors)
            manifest['steps'][name] = result.returncode
            save()
            if result.returncode:
                raise RuntimeError(f'Analysis failed for {name}')
        subprocess.run([sys.executable, str(ROOT/'mvh/analyze_cpu_profile.py'), str(output)],
                       check=True)
        manifest['complete'] = True
    except BaseException as error:
        manifest['error'] = str(error) or type(error).__name__
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        save()
        if os.geteuid() == 0:
            owner = ROOT.stat()
            for path in [output, *output.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid)
    print('FINISHED:', output, flush=True)


if __name__ == '__main__':
    main()
