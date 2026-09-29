"""Snapshot, build and compare host-packing changes without application profiling.
Run inside the project container. Does not change the working tree or run git checkout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['smoke', 'ecoli', 'marine', 'soil'], default='ecoli')
    p.add_argument('--baseline', default='b8262427cbe8393acaae34e6f5319a2b5e11715e')
    p.add_argument('--batch', type=int, default=4000000)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--backends', nargs='+', choices=['cuda', 'rt-custom', 'rt-triangle', 'rt-instanced'], default=['cuda', 'rt-custom'])
    a = p.parse_args()
    if a.batch < 1 or a.repeats < 1 or len(a.backends) != len(set(a.backends)):
        p.error('positive batch/repeats and unique backends required')
    root = Path(__file__).resolve().parents[2]
    git = ['git', '-c', f'safe.directory={root}', '-C', str(root)]
    baseline = subprocess.check_output(git + ['rev-parse', a.baseline + '^{commit}'], text=True).strip()
    data = root / 'mvh_cuda/tests/data'
    smoke = a.dataset == 'smoke'
    scans = data/'sample.ft2' if smoke else root/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2'
    config = data/'search.cfg' if smoke else root/'experiments/Regular.cfg'
    fasta = data/'proteins.fasta' if smoke else root/'raw'/dict(ecoli='Ecoli.fasta', marine='Marine_fw_3rev.fasta', soil='Soil_fw_3rev.fasta')[a.dataset]
    for path in (scans, config, fasta, root/'build/mvh/bin/sipros_mvh', root/'build/mvh_rt/optix_sdk/include/optix.h'):
        if not path.is_file(): p.error(f'Missing input/tool dependency: {path}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root/'output/benchmarks/host_reuse'/f'{a.dataset}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    print('OUTPUT:', output, flush=True)
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(root/'build/mvh_rt/optix_runtime') + ':/usr/local/cuda/lib64'

    def run(cmd, log=None):
        print('RUN:', ' '.join(map(str, cmd)), flush=True)
        if log:
            with log.open('w') as stream:
                subprocess.run(list(map(str, cmd)), env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(list(map(str, cmd)), env=env, check=True)

    manifest = {'options': vars(a), 'baseline_commit': baseline,
                'working_head': subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()}
    (output/'source.diff').write_bytes(subprocess.check_output(git + ['diff', '--no-ext-diff', a.baseline]))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    run(['nvidia-smi'], output/'gpu_environment.log')
    archive = output/'baseline.tar'
    run(git + ['archive', '--format=tar', '-o', str(archive), baseline])
    before = output/'before'
    before.mkdir()
    run(['tar', '-xf', archive, '-C', before])
    after = output/'after'
    shutil.copytree(before, after)
    # Overlay tracked working files, plus new source files in the relevant modules.
    tracked = subprocess.check_output(git + ['ls-files', '-z']).decode().split('\0')
    added = subprocess.check_output(git + ['ls-files', '--others', '--exclude-standard', '-z', '--', 'mvh_cuda', 'MVH_RT', 'shared']).decode().split('\0')
    for name in filter(None, tracked + added):
        src, dst = root/name, after/name
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif dst.is_file():
            dst.unlink()
    # Contract fixtures are ignored by Git; supply the same inputs to both builds.
    for snapshot in (before, after):
        shutil.copytree(data, snapshot/'mvh_cuda/tests/data', dirs_exist_ok=True)
    # Keep manifests of the exact source snapshots, including untracked additions.
    for label, snapshot in [('before', before), ('after', after)]:
        hashes = {str(f.relative_to(snapshot)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted(snapshot.rglob('*')) if f.is_file()}
        (output/f'{label}_source_hashes.json').write_text(json.dumps(hashes, indent=2))
        build = snapshot/'build/mvh_rt/gpu_integration'
        build.parent.mkdir(parents=True)
        (snapshot/'build/mvh').symlink_to(root/'build/mvh', target_is_directory=True)
        (snapshot/'build/mvh_rt/optix_runtime').symlink_to(root/'build/mvh_rt/optix_runtime', target_is_directory=True)
        run(['cmake', '-S', snapshot/'mvh_cuda', '-B', build, '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_CUDA_ARCHITECTURES=120', '-DMVH_CUDA_ENABLE_RT=ON',
             f'-DOPTIX_ROOT={root}/build/mvh_rt/optix_sdk'], output/f'{label}_configure.log')
        run(['cmake', '--build', build, '-j4'], output/f'{label}_build.log')
        run(['ctest', '--test-dir', build, '--output-on-failure', '-R',
             '^(cuda_semantic_contract|gpu_rt_contract|gpu_rt_instanced_contract|gpu_rt_custom_contract|gpu_rt_sphere_layout|sequence_ids_contract|sequence_ids_reuse_contract)$'], output/f'{label}_tests.log')
    records = []
    for repeat in range(a.repeats):
        order = ['before', 'after'] if repeat % 2 == 0 else ['after', 'before']
        backends = a.backends[repeat % len(a.backends):] + a.backends[:repeat % len(a.backends)]
        for label in order:
            destination = output/f'{label}_{repeat + 1}'
            # Use the same measurement script for both source snapshots.
            run([sys.executable, root/'MVH_RT/gpu_bridge/benchmark.py', '--root', output/label,
                 '--output', destination, '--scans', scans, '--config', config, '--fasta', fasta,
                 '--batch', a.batch, '--repeats', 1, '--backends', *backends])
            report = json.loads((destination/'report.json').read_text())
            for item in report['runs']:
                item.update(version=label, comparison_repeat=repeat + 1)
                records.append(item)
            (output/'comparison.json').write_text(json.dumps(records, indent=2))
    lines = ['# Host reuse before/after comparison', '',
             'No application profiling. Means across repeats; raw runs in comparison.json.', '',
             '| Backend | PSM identical | Before search s | After search s | Before wall s | After wall s | Before HWM MiB | After HWM MiB |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    all_equal = True
    for backend in a.backends:
        rows = [r for r in records if r['backend'] == backend]
        equal = len({r['psm_sha256'] for r in rows}) == 1
        all_equal &= equal
        groups = [[r for r in rows if r['version'] == label] for label in ['before', 'after']]
        values = []
        for measure in (lambda r: float(r['summary']['search_seconds']), lambda r: r['wall_seconds'],
                        lambda r: r['host_vm_hwm_observed_bytes'] / 2**20):
            values.extend(f'{statistics.mean(map(measure, group)):.3f}' for group in groups)
        lines.append(f'| {backend} | {equal} | ' + ' | '.join(values) + ' |')
    lines += ['', 'PSM equality is checked within each backend across both versions and all repeats.',
              'Host HWM is observed by sampling; GPU memory in raw reports is device-wide.',
              'Retained output includes source snapshots, binaries, build/test logs and individual PSM files.']
    (output/'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('FINISHED:', output/'REPORT.md', flush=True)
    if not all_equal:
        raise SystemExit('PSM mismatch: inspect comparison.json before accepting performance results')


if __name__ == '__main__':
    main()
