#!/usr/bin/env python3
"""B integration validation. Execute only in the existing Sipros container."""
import argparse
import csv
import filecmp
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import zlib

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summary(path):
    with path.open(newline='') as f:
        return {r['metric']: r['value'] for r in csv.DictReader(f, delimiter='\t')}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT / 'build-mvh-only')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--input', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--fasta', type=Path)
    p.add_argument('--reference', type=Path, help='Previously validated A TSV, required for supplied real data')
    p.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    p.add_argument('--raw-threads', type=int, nargs='+', help='Defaults to all --threads; first thread must be included')
    p.add_argument('--repeat', type=int, default=2)
    a = p.parse_args()
    raw_threads = a.raw_threads or a.threads
    require(a.repeat >= 2 and all(t > 0 for t in a.threads), 'Use positive threads and at least two repeats')
    require(a.threads[0] in raw_threads, 'First thread must produce a raw-path snapshot')
    provided = [a.input, a.config, a.fasta, a.reference]
    require(all(provided) or not any(provided), 'Supply --input --config --fasta --reference together')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    binary = a.build.resolve() / 'bin/sipros_mvh_snapshot'
    records = []

    def run(command, log, good=True):
        print('Running', log.name, flush=True)
        with log.open('w') as f:
            result = subprocess.run([str(x) for x in command], stdout=f, stderr=subprocess.STDOUT)
        require((result.returncode == 0) == good, f'Unexpected exit {result.returncode}: {log}')

    def compare(directory, baseline, expected):
        info = summary(directory / 'preparation.tsv')
        for i in range(1, a.repeat + 1):
            current = directory / f'run_{i}'
            require(filecmp.cmp(current / 'mvh_psms.tsv', baseline, shallow=False), f'Ordered TSV mismatch: {current}')
            result = summary(current / 'run_summary.tsv')
            for key in ('scan_count', 'precursor_entry_count', 'skipped_scan_count', 'retained_psm_count', 'peptide_batch_size'):
                require(result[key] == expected[key], f'Summary mismatch {key}: {current}')
            require(result['peptide_batch_size'] == '2000000', 'Production batch changed')
            if info['input_mode'] == 'snapshot':
                require(info['spectrum_load_seconds'] == '0' and info['preprocess_seconds'] == '0', 'Snapshot path repeated preprocessing')
            records.append(dict(run=str(current), sha256=digest(current / 'mvh_psms.tsv'), preparation=info, summary=result))
        (out / 'validation.json').write_text(json.dumps(records, indent=2) + '\n')

    if all(provided):
        cases = [('real', tuple(x.resolve() for x in provided))]
    else:
        spec = importlib.util.spec_from_file_location('a_validation', ROOT / 'analysis/mvh_search/validate.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cases = []
        for case in ('mixed', 'unmatched', 'skipped', 'multi_precursor'):
            folder = out / case; folder.mkdir()
            ft2, config, fasta = module.fixture(folder, 'mixed' if case == 'multi_precursor' else case)
            if case == 'multi_precursor':
                lines = ft2.read_text().splitlines()
                for i, line in enumerate(lines):
                    if line.startswith('S\t1\t'):
                        mz = line.split('\t')[2]
                        lines[i+1] = f'Z\t0\t0\t2\t{mz}\t2\t{mz}\t3\t{mz}'
                # Duplicate instrument scan IDs must not merge object identity.
                ft2.write_text('\n'.join(lines).replace('S\t4\t', 'S\t1\t') + '\n')
            cases.append((case, (ft2, config, fasta, None)))
    for case, (ft2, config, fasta, reference) in cases:
        folder = out / case
        folder.mkdir(exist_ok=True)
        if reference is None:
            a_out = folder / 'A'
            run([a.build.resolve() / 'bin/sipros_mvh_search', '-f', ft2, '-c', config,
                 '-fasta', fasta, '-o', a_out, '-t', '1'], folder / 'A.log')
            reference = a_out / 'mvh_psms.tsv'
        expected = summary(reference.parent / 'run_summary.tsv')
        base = [binary, '-c', config, '-fasta', fasta, '--repeat', str(a.repeat)]
        snapshot = None
        for threads in a.threads:
            if threads in raw_threads:
                raw = folder / f'raw_t{threads}'
                run(base + ['-f', ft2, '-o', raw, '-t', str(threads)], folder / f'raw_t{threads}.log')
                compare(raw, reference, expected)
                candidate_snapshot = raw / 'preprocessed.mvh'
                if snapshot:
                    require(filecmp.cmp(snapshot, candidate_snapshot, shallow=False), 'Snapshots differ across preprocessing threads')
                snapshot = candidate_snapshot
            loaded = folder / f'loaded_t{threads}'
            hidden = ft2.with_suffix('.hidden_for_B_test')
            # Hide only our own synthetic input, never move the existing real data.
            if not all(provided):
                ft2.rename(hidden)
            try:
                run(base + ['--snapshot', snapshot, '-o', loaded, '-t', str(threads)], folder / f'loaded_t{threads}.log')
            finally:
                if not all(provided):
                    hidden.rename(ft2)
            compare(loaded, reference, expected)
        if not all(provided) and case == 'mixed':
            data = snapshot.read_bytes()
            damaged = {
                'truncated': data[:-17],
                'crc': data[:-5] + bytes([data[-5] ^ 1]) + data[-4:],
            }
            for label, offset in [('version', 8), ('build', 20)]:
                v = bytearray(data); v[offset] ^= 0x10
                v[-4:] = struct.pack('<I', zlib.crc32(v[:-4]))
                damaged[label] = v
            # Header: magic8, version4, build length8+build, config length8+config.
            v = bytearray(data); v[12:20] = struct.pack('<Q', 2**63)
            v[-4:] = struct.pack('<I', zlib.crc32(v[:-4])); damaged['length'] = v
            for label, content in damaged.items():
                path = folder / (label + '.mvh'); path.write_bytes(content)
                run(base + ['--snapshot', path, '-o', folder / label], folder / (label + '.log'), False)
            wrong = folder / 'different.cfg'; wrong.write_text(config.read_text() + '\n# changed configuration\n')
            run([binary, '--snapshot', snapshot, '-c', wrong, '-fasta', fasta, '-o', folder / 'config_mismatch'], folder / 'config_mismatch.log', False)
            before = digest(snapshot)
            run(base + ['--snapshot', snapshot, '-o', snapshot.parent], folder / 'overwrite.log', False)
            require(digest(snapshot) == before, 'Existing snapshot overwritten')
            run([binary, '-f', ft2, '-c', config, '-o', folder / 'prepare_only', '--prepare-only'], folder / 'prepare_only.log')
            require(filecmp.cmp(snapshot, folder / 'prepare_only/preprocessed.mvh', shallow=False), 'Prepare-only snapshot differs')
            require(not (folder / 'prepare_only/run_1').exists(), 'Prepare-only ran a search')
    print(f'PASS: {len(records)} B searches equal A, including same-process reuse; {out}', flush=True)


if __name__ == '__main__':
    main()
