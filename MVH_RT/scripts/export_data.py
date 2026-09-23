#!/usr/bin/env python3
"""Container-only exporter for scan 1004 and its validated [LDNM~ATK] PSM."""
import argparse
import sys
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

# Locate the shared path policy independently of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.output_paths import resolve_output

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/mvh_rt/bin/export_mvh_sample')
    parser.add_argument('--output', type=Path, help='New output directory (default: project output tree)')
    args = parser.parse_args()
    output = resolve_output(args.output, "exports", "mvh_rt", "scan_1004")
    if output.exists():
        parser.error('Output already exists; choose a new directory')
    spectrum = ROOT / 'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2'
    config = ROOT / 'experiments/Regular.cfg'
    reference = ROOT / 'output/reference/historical_mvh_psms.tsv.gz'
    with gzip.open(reference, 'rt') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        selected = next(row for row in reader
                        if row['scan_id'] == '1004' and row['peptide'] == '[LDNM~ATK]'
                        and row['precursor_charge'] == '2')
        fields = reader.fieldnames
    with tempfile.TemporaryDirectory(prefix='mvh-rt-export-') as temporary:
        psm = Path(temporary) / 'selected_psm.tsv'
        with psm.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerow(selected)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [str(args.binary.resolve()), str(spectrum), str(config), str(psm), str(output)]
        run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if output.exists():
            (output / 'export.log').write_text(run.stdout)
        print(run.stdout, end='')
        run.check_returncode()

    # Retain the original scan text as provenance, not as the preprocessing input.
    headers, scan_lines, active = [], [], False
    with spectrum.open() as stream:
        for line in stream:
            if line.startswith('H\t'):
                headers.append(line)
            if line.startswith('S\t'):
                if active:
                    break
                active = line.split()[1] == '1004'
            if active:
                scan_lines.append(line)
    (output / 'raw_scan.ft2.txt').write_text(''.join(headers + scan_lines))
    sources = [spectrum, config, reference, args.binary.resolve(),
               ROOT / 'mvh/original/src/MVH.cpp', ROOT / 'mvh/original/src/ms2scan.cpp']
    (output / 'provenance.json').write_text(json.dumps({
        'method': 'Original CPU functions; snapshot before constructing PeakList',
        'source_sha256': {str(path): sha256(path) for path in sources},
        'git_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT}', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'full_spectrum_read_for_global_bounds': True,
        'table_used_only_for_reference_validation': 'ln_factorial_table.bin',
    }, indent=2) + '\n')
    print('Saved:', output)


if __name__ == '__main__':
    main()
