"""Build, check and serially benchmark the existing CPU/CUDA/OptiX pipelines."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# Locate the shared path policy independently of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.output_paths import resolve_output

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', choices=['smoke', 'ecoli', 'marine'], default='smoke')
p.add_argument('--fasta', type=Path, help='Override dataset FASTA')
p.add_argument('--scans', type=Path, help='Override dataset FT2')
p.add_argument('--config', type=Path, help='Override dataset configuration')
p.add_argument('--batch', type=int, default=4000000)
p.add_argument('--repeats', type=int, default=1)
p.add_argument('--backends', nargs='+', choices=['cpu','cuda','rt-triangle','rt-instanced','rt-custom'],
               default=['cpu','cuda','rt-triangle','rt-instanced','rt-custom'])
p.add_argument('--output', type=Path, help='New directory inside container')
p.add_argument('--skip-build', action='store_true')
p.add_argument('--skip-tests', action='store_true')
a = p.parse_args()
if a.batch < 1 or a.repeats < 1 or len(set(a.backends)) != len(a.backends):
    p.error('batch/repeats must be positive and backends must be unique')
root = Path(__file__).resolve().parents[2]
data = root / 'mvh_cuda/tests/data'
smoke = a.dataset == 'smoke'
fasta = a.fasta or (data/'proteins.fasta' if smoke else root/'raw'/('Marine_fw_3rev.fasta' if a.dataset=='marine' else 'Ecoli.fasta'))
scans = a.scans or (data/'sample.ft2' if smoke else root/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2')
config = a.config or (data/'search.cfg' if smoke else root/'experiments/Regular.cfg')
for f in (fasta, scans, config):
    if not f.is_file(): p.error(f'Input missing: {f}')
output = resolve_output(a.output, "benchmarks", "gpu_bridge", a.dataset)
output.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = str(root/'build/mvh_rt/optix_runtime') + ':/usr/local/cuda/lib64'
print('OUTPUT:', output, flush=True)

def run(command, log):
    print('RUN:', ' '.join(map(str,command)), flush=True)
    with (output/log).open('w') as stream:
        subprocess.run(list(map(str,command)), env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

try:
    metadata = {'options': {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                'inputs': list(map(str,(scans,config,fasta)))}
    for command, key in [(['git','rev-parse','HEAD'],'git_head'), (['git','status','--short'],'git_status'),
                         (['nvidia-smi'],'gpu_environment')]:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        metadata[key] = result.stdout + result.stderr
    metadata['optix_validation_source'] = [line.strip() for line in
        (root/'MVH_RT/optix_example/rt_support.cpp').read_text().splitlines() if 'options.validationMode =' in line]
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    diff = subprocess.run(['git','-C',str(root),'diff','--no-ext-diff'], capture_output=True, text=True)
    (output/'source.diff').write_text(diff.stdout if diff.returncode == 0 else 'Git diff unavailable in container: '+diff.stderr)
    import hashlib
    source_hashes = {}
    for base in ['mvh', 'mvh_cuda', 'MVH_RT/gpu_bridge', 'MVH_RT/optix_example', 'shared']:
        for f in (root/base).rglob('*'):
            if f.is_file() and f.suffix in {'.cpp','.cu','.cuh','.h','.py','.sh','.txt'}:
                source_hashes[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
    (output/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2))
    if not a.skip_build:
        run(['cmake','--build',root/'build/mvh','-j4'],'build_cpu.log')
        run(['cmake','--build',root/'build/mvh_rt/gpu_integration','-j4'],'build_gpu.log')
    if not a.skip_tests:
        for name, build in [('cpu','build/mvh'),('gpu','build/mvh_rt/gpu_integration')]:
            run(['ctest','--test-dir',root/build,'--output-on-failure'],f'tests_{name}.log')
    command = [sys.executable,root/'MVH_RT/gpu_bridge/benchmark.py','--root',root,
               '--output',output/'benchmark','--fasta',fasta,'--scans',scans,'--config',config,
               '--batch',str(a.batch),'--repeats',str(a.repeats),'--backends',*a.backends]
    # Inherit stdout: benchmark emits START/DONE so long searches remain visible.
    subprocess.run(list(map(str,command)),env=env,check=True)
    print('FINISHED:',output/'benchmark/REPORT.md',flush=True)
finally:
    if os.geteuid()==0:
        for f in [output,*output.rglob('*')]: os.chown(f,1000,1000)
