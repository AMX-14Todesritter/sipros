#!/usr/bin/env python3
"""Check CLI rejection and peak-export copy hooks inside the container."""
import argparse
import importlib.util
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = a.fixture.resolve()
    base = [str(a.binary.resolve()), '-f', str(fixture / 'synthetic.ft2'), '-c',
            str(fixture / 'search.cfg'), '-fasta', str(fixture / 'database.fasta')]
    existing = out / 'existing'
    existing.mkdir()
    sentinel = existing / 'sentinel'
    sentinel.write_text('preserve\n')
    invalid_config = out / 'sip.cfg'
    invalid_config.write_text((fixture / 'search.cfg').read_text().replace('Search_Type = Regular', 'Search_Type = SIP'))
    cases = [base + ['-o', str(existing)], base + ['-o', str(out / 'bad-thread'), '-t', '0'],
             base + ['-o', str(out / 'bad-type'), '-c', str(invalid_config)],
             base + ['-o', str(out / 'bad-input'), '-f', str(out / 'missing.ft2')]]
    for i, command in enumerate(cases):
        with (out / f'case_{i}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            raise RuntimeError(f'Invalid input accepted: {command}')
    if sentinel.read_text() != 'preserve\n' or list(existing.iterdir()) != [sentinel]:
        raise RuntimeError('Existing output directory changed')
    for name in ('bad-thread', 'bad-type', 'bad-input'):
        if (out / name).exists():
            raise RuntimeError(f'Rejected command created output: {name}')
    exporter = Path(__file__).resolve().parents[1] / 'peak_export/export_one.py'
    spec = importlib.util.spec_from_file_location('export_one', exporter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.prepare(out / 'export_copy')
    if not (source / 'MSToolkit/src/expat-2.2.9/lib/xmlparse.c').is_file():
        raise RuntimeError('Expat source missing')
    print('PASS: four CLI rejection cases, preserved output, peak-export copy and unique hooks')


if __name__ == '__main__':
    main()
