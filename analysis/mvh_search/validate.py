#!/usr/bin/env python3
"""Run inside the existing container. Compare complete, ordered MVH snapshots."""
import argparse
import csv
import filecmp
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def read_tsv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fixture(out, case):
    config = out / 'search.cfg'
    config.write_text((ROOT / 'configTemplates/Regular.cfg').read_text())
    fasta = out / 'database.fasta'
    fasta.write_text('>plain\nPEPTIDEK\n>methionine\nPEMPTIDEK\n>duplicate\nPEPTIDEK\n')
    # Monoisotopic elemental masses from the supplied configuration; b/y ions.
    masses = dict(P=97.052764, E=129.042594, T=101.047679, I=113.084064,
                  D=115.026944, K=128.094963, M=131.040485)
    water, proton = 18.010565, 1.007276466
    def spectrum(seq, oxidized=False):
        residues = [masses[a] for a in seq]
        if oxidized:
            residues[seq.index('M')] += 15.994915
        total = sum(residues) + water
        ions = []
        for i in range(1, len(residues)):
            ions.extend([sum(residues[:i]) + proton, sum(residues[i:]) + water + proton])
        return total, [(mz, 10000 - i * 101) for i, mz in enumerate(ions)] + [(90.0, 200), (1100.0, 100)]
    mass, peaks = spectrum('PEPTIDEK')
    modified_mass, modified_peaks = spectrum('PEMPTIDEK', True)
    cases = [(1, mass, peaks), (2, 10000.0, peaks), (3, mass, [(150.0, 100)]),
             (4, modified_mass, modified_peaks)]
    if case == 'unmatched':
        cases = [cases[1]]
    elif case == 'skipped':
        cases = [cases[2]]
    ft2 = out / 'synthetic.ft2'
    with ft2.open('w') as f:
        f.write('H\tExtractor\tMVH validation fixture\n')
        for scan, neutral, ions in cases:
            mz = neutral / 2 + proton
            f.write(f'S\t{scan}\t{mz:.9f}\t10000\nZ\t2\t{neutral + proton:.9f}\nI\tRetentionTime\t1.0\n')
            for mz, intensity in reversed(ions):
                f.write(f'{mz:.9f}\t{intensity}\n')
    return ft2, config, fasta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT / 'build-mvh-only')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--input', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--fasta', type=Path)
    p.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--standalone-only', action='store_true', help='Repeat the independent entry for timing after reference validation')
    p.add_argument('--case', choices=['mixed', 'unmatched', 'skipped'], default='mixed')
    args = p.parse_args()
    require(args.repeats > 0 and all(t > 0 for t in args.threads), 'Positive repeats/threads required')
    supplied = [args.input, args.config, args.fasta]
    require(all(supplied) or not any(supplied), 'Supply all of --input --config --fasta, or none for synthetic')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ft2, config, fasta = [f.resolve() for f in supplied] if all(supplied) else fixture(out, args.case)
    records = []
    baseline = None
    for threads in args.threads:
        for repeat in range(args.repeats):
            for target in (('sipros_mvh_search',) if args.standalone_only else ('sipros_mvh_search', 'sipros_mvh_reference')):
                run = out / f'{target}_t{threads}_r{repeat}'
                if target.endswith('reference'):
                    run.mkdir()
                command = [str(args.build.resolve() / 'bin' / target), '-f', str(ft2),
                           '-c', str(config), '-fasta', str(fasta), '-o', str(run), '-t', str(threads)]
                print('Running', target, threads, repeat, flush=True)
                with (out / (run.name + '.log')).open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                snapshot = run / 'mvh_psms.tsv'
                digest = hashlib.sha256()
                with snapshot.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                rows = read_tsv(run / 'mvh_psms.tsv')
                for row in rows:
                    require(math.isfinite(float(row['mvh_score'])), 'Nonfinite MVH score')
                    require(1 <= int(row['mvh_rank']) <= 50, 'Invalid retained rank')
                if baseline is None:
                    baseline = snapshot
                require(filecmp.cmp(snapshot, baseline, shallow=False), f'Ordered snapshot mismatch: {run}; files preserved for diagnosis')
                record = dict(target=target, threads=threads, repeat=repeat, rows=len(rows),
                              sha256=digest.hexdigest(), command=command)
                if target.endswith('search'):
                    summary = {r['metric']: r['value'] for r in read_tsv(run / 'run_summary.tsv')}
                    require(int(summary['retained_psm_count']) == len(rows), 'Summary row count mismatch')
                    require(int(summary['peptide_batch_size']) == 2000000, 'Production batch changed')
                    record['summary'] = summary
                    if not any(supplied):
                        require(summary['scan_count'] == ('4' if args.case == 'mixed' else '1'), 'Scan count mismatch')
                        require(summary['skipped_scan_count'] == ('0' if args.case == 'unmatched' else '1'), 'Skipped count mismatch')
                if not any(supplied) and args.case != 'mixed':
                    require(not rows, 'Expected header-only output')
                if not any(supplied) and args.case == 'mixed':
                    require({r['scan_id'] for r in rows} == {'1', '4'}, 'Match/no-match/skip coverage failed')
                    require(any('~' in r['peptide'] for r in rows if r['scan_id'] == '4'), 'PTM was not retained')
                    require(any(float(r['mvh_score']) > 0 for r in rows), 'No positive match score')
                records.append(record)
                (out / 'validation.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'PASS: {len(records)} runs; exact ordered TSV equality; {out}', flush=True)


if __name__ == '__main__':
    main()
