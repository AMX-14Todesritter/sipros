#!/usr/bin/env python3
"""Record completed B validation provenance and timing summaries in its output folder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--synthetic', type=Path, required=True)
    p.add_argument('--build', type=Path, default=ROOT / 'build-mvh-only')
    a = p.parse_args()
    runs = json.loads((a.output / 'validation.json').read_text())
    synthetic = json.loads((a.synthetic / 'validation.json').read_text())
    if len(runs) != 6 or len(synthetic) != 32 or len({r['sha256'] for r in runs}) != 1:
        raise RuntimeError('Incomplete validation records')
    identity = {r['preparation']['build_identity'] for r in runs + synthetic}
    if len(identity) != 1:
        raise RuntimeError('Mixed build identities')
    groups = {}
    for r in runs:
        groups.setdefault((r['preparation']['input_mode'], r['summary']['omp_max_threads']), []).append(r)
    timing = {}
    for (mode, threads), members in groups.items():
        timing[mode + '_t' + threads] = dict(
            preparation=members[0]['preparation'],
            **{metric: dict(values=[float(r['summary'][metric]) for r in members],
                            median=statistics.median(float(r['summary'][metric]) for r in members))
               for metric in ('state_prepare_seconds', 'search_seconds', 'export_seconds')})
    prep = runs[0]['preparation']
    files = [Path(prep[key]) for key in ('input_file', 'config_file', 'fasta_file', 'snapshot_file')]
    files += [a.build / name for name in ('bin/sipros_mvh_snapshot', 'bin/sipros_mvh_search',
                                         'CMakeCache.txt', 'compile_commands.json')]
    result = dict(build_identity=next(iter(identity)), result_sha256=runs[0]['sha256'], timing=timing,
                  synthetic_searches=len(synthetic), real_searches=len(runs),
                  files={str(f): dict(bytes=f.stat().st_size, sha256=digest(f)) for f in files},
                  platform=platform.platform(),
                  lscpu=subprocess.check_output(['lscpu'], text=True),
                  tools={tool: subprocess.check_output([tool, '--version'], text=True).splitlines()[0]
                         for tool in ('cmake', 'c++')},
                  environment={k: v for k, v in os.environ.items() if k.startswith(('OMP_', 'SIPROS_MVH_PROFILE'))})
    for name in ('cpu.max', 'memory.max', 'cpuset.cpus.effective'):
        path = Path('/sys/fs/cgroup') / name
        result[name] = path.read_text().strip() if path.exists() else None
    result['source_hashes'] = {str(f.relative_to(ROOT)): digest(f)
                              for folder in ('src', 'include', 'mvh')
                              for f in sorted((ROOT / folder).rglob('*'))
                              if f.is_file() and '__pycache__' not in str(f)}
    with (a.output / 'metadata.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print('build_identity:', result['build_identity'])
    print('result_sha256:', result['result_sha256'])
    print('snapshot_bytes:', files[3].stat().st_size)
    print('| 路径 | 线程 | 准备/加载 (s) | 状态准备中位数 (s) | 搜索两次 (s) |')
    print('|---|---|---|---|---|')
    for key, t in timing.items():
        p = t['preparation']
        seconds = float(p['snapshot_load_seconds']) if p['input_mode'] == 'snapshot' else float(p['spectrum_load_seconds']) + float(p['preprocess_seconds'])
        searches = ', '.join('{:.5f}'.format(v) for v in t['search_seconds']['values'])
        print('| {} | {} | {:.5f} | {:.5f} | {} |'.format(p['input_mode'], p['omp_max_threads'], seconds, t['state_prepare_seconds']['median'], searches))


if __name__ == '__main__':
    main()
