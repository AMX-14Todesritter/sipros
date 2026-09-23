#!/usr/bin/env python3
"""Compare CPU and CUDA outputs and timings sequentially in the existing container."""
import argparse
import sys
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

# Locate the shared path policy independently of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.output_paths import resolve_output

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read_scoring_profile(log):
    """Keep individual batch counters/timings instead of mixing timer scopes."""
    batches = []
    for line in log.read_text().splitlines():
        if line.startswith('[CUDA scoring]'):
            batches.append({name: float(value) for name, value in
                            re.findall(r'(\w+)=([0-9.eE+-]+)', line)})
    return batches


def write_report(output, report):
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


def archive_identical_psms(output, runs, expected_hash):
    first = output / runs[0]['name'] / 'mvh_psms.tsv'
    archive = output / 'mvh_psms.tsv.gz'
    with first.open('rb') as source, gzip.open(archive, 'wb') as target:
        shutil.copyfileobj(source, target)
    checksum = hashlib.sha256()
    with gzip.open(archive, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != expected_hash:
        raise RuntimeError('Compressed PSM validation failed; originals retained')
    for run in runs:
        (output / run['name'] / 'mvh_psms.tsv').unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New output directory (default: project output tree)')
    parser.add_argument('--real', action='store_true', help='Use retained E. coli inputs')
    parser.add_argument('--verify-cuda', action='store_true',
                        help='CPU verification mode; not a performance measurement')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--keep-psms', action='store_true')
    parser.add_argument('--cuda-baseline-binary', type=Path,
                        help='Also run a saved CUDA binary for before/after comparison')
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.repeats < 1:
        parser.error('threads and repeats must be positive')

    binaries = {'cpu': ROOT / 'build/mvh/bin/sipros_mvh'}
    if args.cuda_baseline_binary:
        binaries['cuda_before'] = args.cuda_baseline_binary.resolve()
    binaries['cuda'] = ROOT / 'build/mvh_cuda/bin/sipros_mvh_cuda'
    for binary in binaries.values():
        if not binary.is_file():
            parser.error(f'Binary not found: {binary}')

    if args.real:
        spectrum = ROOT / 'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2'
        config = ROOT / 'experiments/Regular.cfg'
        fasta = ROOT / 'raw/Ecoli.fasta'
    else:
        data = ROOT / 'mvh_cuda/tests/data'
        spectrum, config, fasta = (data / name for name in
                                   ('sample.ft2', 'search.cfg', 'proteins.fasta'))
    output = resolve_output(args.output, "validation", "mvh_cuda", "ecoli" if args.real else "smoke")
    output.mkdir(parents=True, exist_ok=False)
    report = {'inputs': {str(path): digest(path) for path in (spectrum, config, fasta)},
              'verification_mode': args.verify_cuda, 'runs': [], 'all_equal': False}
    reference = None
    backend_names = list(binaries)

    for repeat in range(1, args.repeats + 1):
        # Rotate execution order to reduce a fixed warm-cache/thermal advantage.
        # Runs remain strictly sequential; no compilation or tests run alongside.
        offset = (repeat - 1) % len(backend_names)
        order = backend_names[offset:] + backend_names[:offset]
        for backend in order:
            binary = binaries[backend]
            name = f'{backend}_{repeat}'
            run_output = output / name
            command = [str(binary), '-f', str(spectrum), '-c', str(config),
                       '-fasta', str(fasta), '-o', str(run_output),
                       '-t', str(args.cpu_threads)]
            if backend != 'cpu' and args.verify_cuda:
                command.append('--verify-cuda')
            print('Running', name, flush=True)
            log_path = output / f'{name}.log'
            with log_path.open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            sha = digest(run_output / 'mvh_psms.tsv')
            if reference is None:
                reference = sha
            with (run_output / 'run_summary.tsv').open() as stream:
                summary = dict(list(csv.reader(stream, delimiter='\t'))[1:])
            report['runs'].append({
                'name': name, 'backend': backend, 'repeat': repeat, 'command': command,
                'binary_sha256': digest(binary), 'psm_sha256': sha,
                'summary': summary, 'scoring_batches': read_scoring_profile(log_path)})
            write_report(output, report)
            if sha != reference:
                raise RuntimeError('PSM mismatch; results and logs retained')

    report['all_equal'] = True
    write_report(output, report)
    with (output / 'comparison.tsv').open('w') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['backend', 'repeat', 'search_seconds', 'preprocess_seconds',
                         'retained_psm_count', 'sha256'])
        for run in report['runs']:
            summary = run['summary']
            writer.writerow([run['backend'], run['repeat'], summary['search_seconds'],
                             summary['preprocess_seconds'], summary['retained_psm_count'],
                             run['psm_sha256']])
    if not args.keep_psms:
        archive_identical_psms(output, report['runs'], reference)
    print(f"PASS: {len(report['runs'])} searches; {output}", flush=True)


if __name__ == '__main__':
    main()
