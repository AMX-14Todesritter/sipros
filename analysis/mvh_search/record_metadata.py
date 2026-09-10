#!/usr/bin/env python3
"""Capture input/build identity after validation; never change environment settings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='Existing validated run directory')
    p.add_argument('--build', type=Path, default=ROOT / 'build-mvh-only')
    a = p.parse_args()
    runs = json.loads((a.output / 'validation.json').read_text())
    command = runs[0]['command']
    inputs = [Path(command[command.index(flag) + 1]) for flag in ('-f', '-c', '-fasta')]
    sources = [ROOT / 'CMakeLists.txt', ROOT / 'MSToolkit/CMakeLists.txt']
    for folder in ('src', 'include', 'openmp', 'analysis/mvh_search'):
        sources.extend(f for f in (ROOT / folder).iterdir() if f.is_file())
    builds = [a.build / f for f in ('CMakeCache.txt', 'compile_commands.json',
              'bin/sipros_mvh_search', 'bin/sipros_mvh_reference')]
    result = dict(platform=platform.platform(), processor=platform.processor(),
                  environment={k: v for k, v in os.environ.items()
                               if k.startswith(('OMP_', 'SIPROS_MVH_PROFILE'))},
                  inputs={str(f): dict(bytes=f.stat().st_size, sha256=sha256(f)) for f in inputs},
                  sources={str(f.relative_to(ROOT)): sha256(f) for f in sorted(sources)},
                  build={str(f): sha256(f) for f in builds},
                  tools={cmd: subprocess.check_output([cmd, '--version'], text=True).splitlines()[0]
                         for cmd in ('cmake', 'c++')})
    for name in ('cpu.max', 'memory.max', 'cpuset.cpus.effective'):
        path = Path('/sys/fs/cgroup') / name
        result[name] = path.read_text().strip() if path.exists() else None
    result['lscpu'] = subprocess.check_output(['lscpu'], text=True)
    with (a.output / 'metadata.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('Recorded metadata:', a.output / 'metadata.json')


if __name__ == '__main__':
    main()
