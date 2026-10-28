"""Build isolated cache-off primary/multi-precursor binaries; never run benchmarks."""
import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys

from inventory import ROOT, OUT, sha

PROTECTED = ['mvh_cuda/cuda/scoring.cuh', 'mvh_cuda/cuda/theoretical.cuh',
             'MVH_RT/gpu_bridge/device.cu', 'MVH_RT/gpu_bridge/custom_device.cu',
             'MVH_RT/gpu_bridge/custom_geometry.cu', 'mvh_cuda/include/peptide_batch.h']


def restore_multi(current, historical):
    sys.path.insert(0, str(ROOT / 'mvh/tests'))
    from source_methods import methods
    now, old = methods(current, 'MvhScanVector'), methods(historical, 'MvhScanVector')
    for name in ('ReadFT2File', 'loadFT2File', 'saveFT2Scan'):
        new, previous = now[name], old[name]
        replacement = new[:new.index('{')] + '{' + previous[previous.index('{') + 1:-1] + '}'
        if current.count(new) != 1: raise ValueError('Ambiguous reader method: ' + name)
        current = current.replace(new, replacement)
    return current


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precursor-mode', choices=['primary', 'multi'], required=True)
    p.add_argument('--cuda-architectures', default='120')
    p.add_argument('--build-name', help='Unique directory name under output/evaluation/build')
    p.add_argument('--optix-root', type=Path, default=ROOT / 'build/mvh_rt/optix_sdk')
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    if a.jobs < 1: p.error('--jobs must be positive')
    if not (a.optix_root / 'include/optix.h').is_file(): p.error('Missing OptiX SDK headers')
    if 'constexpr bool useIonCache=false;' not in (ROOT / 'mvh_cuda/cuda/engine.cu').read_text():
        p.error('Current source must use the cache-off baseline')
    historical = None
    if a.precursor_mode == 'multi':
        try:
            historical = subprocess.check_output(['git', 'show', '82522a4:mvh_cuda/src/spectrum_input.cpp'], cwd=ROOT, text=True)
        except subprocess.CalledProcessError:
            p.error('Historical reader unavailable; fetch full repository history first (git fetch --unshallow if shallow)')
    build_name = a.build_name or ('gpu_' + a.precursor_mode)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', build_name): p.error('Invalid build name')
    d = OUT / 'build' / build_name
    if d.exists(): p.error('Build directory already exists; preserve it and choose a clean setup: ' + str(d))
    d.mkdir(parents=True); source = d / 'source'; source.mkdir()
    for name in ('mvh_cuda', 'MVH_RT'):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('MSToolkit', 'shared'): (source / name).symlink_to(ROOT / name, target_is_directory=True)
    reader = source / 'mvh_cuda/src/spectrum_input.cpp'
    if historical is not None: reader.write_text(restore_multi(reader.read_text(), historical))
    assert all(sha(ROOT / rel) == sha(source / rel) for rel in PROTECTED)
    m = {'status': 'building', 'precursor_mode': a.precursor_mode, 'ion_cache': 'compiled_off',
         'build_name': build_name, 'cuda_architectures': a.cuda_architectures,
         'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'source_diff': subprocess.check_output(['git', 'diff', 'HEAD', '--', 'mvh_cuda', 'MVH_RT', 'shared'], cwd=ROOT, text=True),
         'reader_commit': '82522a4' if historical is not None else None,
         'reader_sha256': sha(reader), 'engine_sha256': sha(source / 'mvh_cuda/cuda/engine.cu'),
         'protected_sha256': {rel: sha(source / rel) for rel in PROTECTED}}
    mp = d / 'build_manifest.json'
    commands = [['cmake', '-S', str(source / 'mvh_cuda'), '-B', str(d), '-G', 'Ninja',
                 '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=' + a.cuda_architectures,
                 '-DMVH_CUDA_ENABLE_RT=ON', '-DMVH_ENABLE_EVALUATION_TIMING=ON',
                 '-DMVH_ENABLE_FLOW_COUNTERS=OFF', '-DMVH_ENABLE_PROFILING=OFF',
                 '-DOPTIX_ROOT=' + str(a.optix_root.resolve())],
                ['cmake', '--build', str(d), '--target', 'sipros_mvh_cuda', '-j', str(a.jobs)]]
    m['commands'] = commands
    try:
        for i, cmd in enumerate(commands):
            mp.write_text(json.dumps(m, indent=2))
            with (d / f'build_{i}.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        m.update(status='completed', binary_sha256=sha(d / 'bin/sipros_mvh_cuda'))
        print(d / 'bin/sipros_mvh_cuda')
    except BaseException as error:
        m.update(status='failed', error=repr(error)); raise
    finally: mp.write_text(json.dumps(m, indent=2))


if __name__ == '__main__': main()
