#!/usr/bin/env python3
"""Collect NCU kernel metrics in the existing container, without changing clocks.

A separate unprofiled run supplies wall time. Never treat replay-instrumented
execution time as application performance. Refuse concurrent Sipros workloads.
"""
import argparse
import sys
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

# Locate the shared path policy independently of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.output_paths import resolve_output

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ('preprocessMvh|sumIntensity|preprocessingMVH|GetAllRangeFromMass|'
           'assignPeptides2Scans|setCandidateRanges|countTheoreticalIons|'
           'generateTheoreticalIons|ScoreSequenceVsSpectrum|scorePeptidesMVH|'
           'gatherScoringEvents')


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def check_idle():
    busy = []
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():
            continue
        try:
            name = (process / 'comm').read_text().strip()
            if name.startswith(('sipros', 'mvh_cuda_contr')):
                busy.append(f'{process.name}: {name}')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    if busy:
        raise RuntimeError('Other Sipros workloads are running; leave them untouched: '
                           + ', '.join(busy))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New output directory (default: project output tree)')
    parser.add_argument('--set', choices=['basic', 'detailed'], default='detailed')
    parser.add_argument('--peptide-batch-size', type=int, default=2000000)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists():
        parser.error('Run this script inside the existing Docker container')
    ncu = shutil.which('ncu')
    if ncu is None:
        parser.error('NCU is not installed in this container; no installation attempted')
    if args.peptide_batch_size < 1:
        parser.error('Batch size must be positive')
    check_idle()
    output = resolve_output(args.output, "profiling", "mvh_cuda", "ecoli")
    output.mkdir(parents=True, exist_ok=False)
    binary = ROOT / 'build/mvh_cuda/bin/sipros_mvh_cuda'
    inputs = [ROOT / 'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2',
              ROOT / 'experiments/Regular.cfg', ROOT / 'raw/Ecoli.fasta']
    report = {'binary': str(binary), 'binary_sha256': digest(binary),
              'inputs': {str(p): digest(p) for p in inputs}, 'runs': [],
              'ncu_version': subprocess.check_output([ncu, '--version'], text=True),
              'kernel_filter': KERNELS, 'set': args.set, 'complete': False,
              'note': 'NCU replay perturbs timings; use unprofiled run_summary for wall time.'}

    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    common = [str(binary), '-f', str(inputs[0]), '-c', str(inputs[1]),
              '-fasta', str(inputs[2]), '--peptide-batch-size', str(args.peptide_batch_size)]
    commands = [
        ('unprofiled', common + ['-o', str(output / 'unprofiled')]),
        ('ncu', [ncu, '--set', args.set, '--clock-control', 'none',
                 '--replay-mode', 'kernel', '--kernel-name-base', 'function',
                 '--kernel-name', 'regex:.*(' + KERNELS + ').*',
                 '--import-source', 'yes', '--export', str(output / 'kernels')]
         + common + ['-o', str(output / 'ncu')])]
    try:
        reference = None
        for name, command in commands:
            check_idle()
            print('Running', name, flush=True)
            with (output / f'{name}.log').open('w') as log:
                start = time.perf_counter()
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                elapsed = time.perf_counter() - start
            record = {'name': name, 'command': command, 'returncode': result.returncode,
                      'process_wall_seconds': elapsed, 'instrumented': name == 'ncu'}
            report['runs'].append(record)
            save()
            if result.returncode:
                raise RuntimeError(f'{name} failed; inspect {output / (name + ".log")}. '
                                   'No permissions, driver settings, or dependencies were changed.')
            psm = output / name / 'mvh_psms.tsv'
            record['psm_sha256'] = digest(psm)
            with (output / name / 'run_summary.tsv').open() as stream:
                record['summary'] = dict(list(csv.reader(stream, delimiter='\t'))[1:])
            if reference is None:
                reference = record['psm_sha256']
            elif record['psm_sha256'] != reference:
                raise RuntimeError('Profiled/unprofiled PSM mismatch; all results retained')
            save()
        # Export both human-readable analysis and raw metrics for independent review.
        ncu_report = output / 'kernels.ncu-rep'
        if not ncu_report.is_file():
            raise RuntimeError('No NCU report produced; inspect kernel filter/permissions')
        for page in ['details', 'raw']:
            with (output / f'{page}.csv').open('w') as stream:
                subprocess.run([ncu, '--import', str(ncu_report), '--page', page, '--csv'],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        with (output / 'unprofiled/mvh_psms.tsv').open('rb') as source:
            with gzip.open(output / 'mvh_psms.tsv.gz', 'wb') as target:
                shutil.copyfileobj(source, target)
        checksum = hashlib.sha256()
        with gzip.open(output / 'mvh_psms.tsv.gz', 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(chunk)
        if checksum.hexdigest() != reference:
            raise RuntimeError('Compressed PSM failed validation; originals retained')
        for name, _ in commands:
            (output / name / 'mvh_psms.tsv').unlink()
        report['complete'] = True
        save()
        print('PASS: NCU metrics and matching outputs in', output, flush=True)
    except Exception as error:
        report['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    main()
