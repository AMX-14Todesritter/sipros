"""Alternating full-data CPU/CUDA/RT runs; no correctness observer in timed runs."""
import argparse
import sys
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

# Locate the shared path policy independently of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.output_paths import resolve_output

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, help='New output directory (default: project output tree)')
p.add_argument('--batch', type=int, default=4000000)
p.add_argument('--scans', type=Path)
p.add_argument('--config', type=Path)
p.add_argument('--fasta', type=Path, help='FASTA override; default is raw/Ecoli.fasta')
p.add_argument('--backends', nargs='+', choices=['cpu', 'cuda', 'rt-triangle', 'rt-instanced'],
               default=['cpu', 'cuda', 'rt-triangle'])
p.add_argument('--repeats', type=int, default=3)
a = p.parse_args()
if a.repeats < 1 or a.batch < 1:
    p.error('--repeats and --batch must be positive')
if len(set(a.backends)) != len(a.backends):
    p.error('duplicate backends are not allowed')
a.output = resolve_output(a.output, "benchmarks", "gpu_bridge", "benchmark", project_root=a.root)
a.output.mkdir(parents=True, exist_ok=False)

nvml = ctypes.CDLL('libnvidia-ml.so.1')
if nvml.nvmlInit_v2() != 0:
    raise RuntimeError('NVML initialization failed')
handle = ctypes.c_void_p()
if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
    raise RuntimeError('NVML device lookup failed')


class Memory(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong)]


def gpu_memory():
    m = Memory()
    if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(m)) != 0:
        raise RuntimeError('NVML memory query failed')
    return m.used


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


root = a.root
inputs = [a.scans if a.scans else root / 'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2',
          a.config if a.config else root / 'experiments/Regular.cfg', a.fasta if a.fasta else root / 'raw/Ecoli.fasta']
cpu = root / 'build/mvh/bin/sipros_mvh'
gpu = root / 'build/mvh_rt/gpu_integration/bin/sipros_mvh_cuda'
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = str(root / 'build/mvh_rt/optix_runtime') + ':/usr/local/cuda/lib64'
report = {
    'inputs_sha256': {str(f): sha(f) for f in inputs},
    'binaries_sha256': {str(f): sha(f) for f in (cpu, gpu)},
    'gpu_batch_generated_peptides': a.batch,
    'cpu_threads': 4,
    'verification': False,
    'memory_scope': '50ms sampling; VmHWM/VmRSS from process; NVML device-wide used includes other processes',
    'runs': [],
}
child = None


def save():
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))


def interrupt(signum, frame):
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    report['interrupted'] = True
    save()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, interrupt)
signal.signal(signal.SIGTERM, interrupt)
orders = [a.backends[i % len(a.backends):] + a.backends[:i % len(a.backends)]
          for i in range(a.repeats)]
save()
try:
    for repeat, order in enumerate(orders, 1):
        for backend in order:
            name = f'{backend}_{repeat}'
            output = a.output / name
            command = [str(cpu if backend == 'cpu' else gpu), '-f', str(inputs[0]),
                       '-c', str(inputs[1]), '-fasta', str(inputs[2]), '-o', str(output), '-t', '4']
            if backend != 'cpu':
                command += ['--match-backend', backend, '--peptide-batch-size', str(a.batch)]
            before = gpu_memory()
            peak_gpu, peak_rss, hwm = before, 0, 0
            samples = 0
            started = time.perf_counter()
            print('START', name, flush=True)
            with (a.output / (name + '.log')).open('w') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                (a.output / 'current_process.json').write_text(json.dumps({'name': name, 'pid': child.pid}))
                while child.poll() is None:
                    peak_gpu = max(peak_gpu, gpu_memory())
                    try:
                        status = Path(f'/proc/{child.pid}/status').read_text()
                        for key, value in re.findall(r'^(VmRSS|VmHWM):\s+(\d+)', status, re.M):
                            if key == 'VmRSS':
                                peak_rss = max(peak_rss, int(value) * 1024)
                            else:
                                hwm = max(hwm, int(value) * 1024)
                    except FileNotFoundError:
                        pass
                    samples += 1
                    time.sleep(0.05)
            item = {'name': name, 'backend': backend, 'repeat': repeat, 'command': command,
                    'exit_code': child.returncode, 'wall_seconds': time.perf_counter() - started,
                    'host_rss_sampled_peak_bytes': peak_rss, 'host_vm_hwm_observed_bytes': hwm,
                    'device_used_before_bytes': before, 'device_used_sampled_peak_bytes': peak_gpu,
                    'device_used_peak_minus_before_bytes': peak_gpu - before, 'samples': samples}
            text = (a.output / (name + '.log')).read_text()
            if child.returncode == 0:
                with (output / 'run_summary.tsv').open() as f:
                    item['summary'] = dict(list(csv.reader(f, delimiter='\t'))[1:])
                item['psm_sha256'] = sha(output / 'mvh_psms.tsv')
                item['batches'] = [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in text.splitlines()
                                   if line.startswith('[CUDA scoring]')]
                fields = ['candidates', 'calls', 'kernel_seconds', 'theory_seconds',
                          'gpu_service_seconds', 'allocation_upload_seconds', 'retention_seconds',
                          'download_seconds', 'cached_ions', 'pack_seconds', 'replay_verify_seconds', 'inrange']
                item['totals'] = {k: sum(float(b[k]) for b in item['batches']) for k in fields}
                item['rt_setup_seconds'] = sum(float(x) for x in re.findall(
                    r'\[RT GPU setup\] scans=\d+ seconds=([0-9.eE+-]+)', text))
                item['rt_setup'] = [dict(re.findall(r'(\w+)=([^ ]+)', line))
                                    for line in text.splitlines() if line.startswith('[RT GPU setup]')]
            report['runs'].append(item)
            save()
            print('DONE', name, 'exit=', item['exit_code'], 'wall_seconds=', round(item['wall_seconds'], 3), flush=True)
            if child.returncode != 0:
                raise RuntimeError(f'{name} failed; see log')
    report['averages'] = {}
    for backend in a.backends:
        runs = [r for r in report['runs'] if r['backend'] == backend]
        metrics = {
            'wall_seconds': [r['wall_seconds'] for r in runs],
            'search_seconds': [float(r['summary']['search_seconds']) for r in runs],
            'scoring_kernel_seconds': [r['totals']['kernel_seconds'] for r in runs],
            'rt_setup_seconds': [r['rt_setup_seconds'] for r in runs],
            'host_hwm_mib': [r['host_vm_hwm_observed_bytes'] / 2**20 for r in runs],
            'device_peak_mib': [r['device_used_sampled_peak_bytes'] / 2**20 for r in runs],
            'device_peak_minus_before_mib': [r['device_used_peak_minus_before_bytes'] / 2**20 for r in runs],
        }
        report['averages'][backend] = {k: {'mean': statistics.mean(v), 'stdev': statistics.stdev(v) if len(v)>1 else None}
                                       for k, v in metrics.items()}
    cpu_hashes = {r['psm_sha256'] for r in report['runs'] if r['backend'] == 'cpu'}
    report['cpu_cuda_psms_identical'] = (len(cpu_hashes) == 1 and all(
        r['psm_sha256'] in cpu_hashes for r in report['runs'] if r['backend'] == 'cuda')) if cpu_hashes else None
    report['rt_psms_identical_to_cpu'] = all(
        r['psm_sha256'] in cpu_hashes for r in report['runs'] if r['backend'].startswith('rt-')) if cpu_hashes else None
    report['psm_hashes_by_backend'] = {k: sorted({x['psm_sha256'] for x in report['runs'] if x['backend']==k}) for k in a.backends}
    report['complete'] = True
    save()
    from benchmark_report import write_report
    write_report(a.output)
finally:
    nvml.nvmlShutdown()
    for path in [a.output, *a.output.rglob('*')]:
        if os.geteuid() == 0:
            os.chown(path, 1000, 1000)
