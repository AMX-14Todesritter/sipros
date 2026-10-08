"""Capture a search timeline with NSYS and selected scoring launches with NCU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.output_paths import resolve_output


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['smoke', 'ecoli', 'marine'], default='marine')
    parser.add_argument('--batch', type=positive, default=6000000,
                        help='Generated peptides per batch (default: 6000000)')
    parser.add_argument('--backend', choices=['cuda', 'rt-triangle', 'rt-custom'],
                        default='rt-custom', help='rt-custom is the sphere backend')
    parser.add_argument('--tools', nargs='+', choices=['nsys', 'ncu'], default=['nsys', 'ncu'],
                        help='Selected tools; NSYS always runs before NCU')
    parser.add_argument('--spectrum-cache', choices=['host', 'device'],
                        help='Optimized binary only: keep packed spectra on host or also on GPU')
    parser.add_argument('--range-trace', action='store_true',
                        help='Also export individual NVTX calls, nesting and self time (may be large)')
    parser.add_argument('--ncu-set', choices=['basic', 'detailed', 'full'], default='basic')
    parser.add_argument('--ncu-launch-skip', type=nonnegative, default=0,
                        help='Skip this many matching scoring launches, not all GPU kernels')
    parser.add_argument('--ncu-launch-count', type=positive, default=1)
    parser.add_argument('--ncu-complete-search', action='store_true',
                        help='Continue the search after NCU collection; otherwise terminate early')
    parser.add_argument('--binary', type=Path,
                        default=ROOT/'build/mvh_rt/profile_enabled/bin/sipros_mvh_cuda')
    parser.add_argument('--fasta', type=Path, help='Input override, path inside container')
    parser.add_argument('--scans', type=Path, help='Input override, path inside container')
    parser.add_argument('--config', type=Path, help='Input override, path inside container')
    parser.add_argument('--output', type=Path, help='New directory inside container')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing tools or creating output')
    args = parser.parse_args()
    if len(set(args.tools)) != len(args.tools):
        parser.error('--tools must not contain duplicates')
    return parser, args


def input_paths(args):
    data = ROOT/'mvh_cuda/tests/data'
    smoke = args.dataset == 'smoke'
    return {
        'scans': (args.scans or (data/'sample.ft2' if smoke else
                  (ROOT/'raw/marine/ft/OSU_D10_FASP_Elite_03202014_01.FT2' if args.dataset == 'marine' else ROOT/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2'))).resolve(),
        'config': (args.config or (data/'search.cfg' if smoke else ROOT/'experiments/Regular.cfg')).resolve(),
        'fasta': (args.fasta or (data/'proteins.fasta' if smoke else
                  ROOT/'raw'/('Marine_fw_3rev.fasta' if args.dataset == 'marine' else 'Ecoli.fasta'))).resolve(),
    }


def capture_commands(args, inputs, output):
    search = [str(args.binary.resolve()), '-f', str(inputs['scans']), '-c', str(inputs['config']),
              '-fasta', str(inputs['fasta']), '--match-backend', args.backend,
              '--peptide-batch-size', str(args.batch)]
    if args.spectrum_cache is not None:
        search += ['--spectrum-cache', args.spectrum_cache]
    commands = {}
    if 'nsys' in args.tools:
        commands['nsys'] = ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none',
                            '--cpuctxsw=none', '--output='+str(output/'nsys_capture'),
                            '--gpu-metrics-devices=all',
                            *search, '-o', str(output/'nsys_results')]
        commands['nsys_summary'] = ['nsys', 'stats', '--report',
                                    'nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum',
                                    str(output/'nsys_capture.nsys-rep')]
    if 'nsys' in args.tools:
        reports = 'nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum'
        if args.range_trace:
            reports += ',nvtx_pushpop_trace'
        commands['nsys_csv'] = ['nsys', 'stats', '--report', reports,
                                '--format', 'csv', '--output', str(output/'timings'),
                                str(output/'nsys_capture.sqlite')]
    if 'ncu' in args.tools:
        kernel = 'ScoreSequenceVsSpectrum' if args.backend == 'cuda' else 'optixLaunch'
        commands['ncu'] = ['ncu', '--set', args.ncu_set, '--kernel-name', kernel,
                           '--launch-skip', str(args.ncu_launch_skip),
                           '--launch-count', str(args.ncu_launch_count),
                           '--kill', 'no' if args.ncu_complete_search else 'yes',
                           '--clock-control', 'none', '--export', str(output/'ncu_capture'),
                           *search, '-o', str(output/'ncu_results')]
        commands['ncu_summary'] = ['ncu', '--import', str(output/'ncu_capture.ncu-rep'), '--page', 'details']
    return commands


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Runner:
    def __init__(self, output, env, manifest):
        self.output, self.env, self.manifest = output, env, manifest
        self.child = None

    def save(self):
        (self.output/'manifest.json').write_text(json.dumps(self.manifest, indent=2))

    def interrupt(self, signum, frame):
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGINT)
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait()
        raise KeyboardInterrupt

    def run(self, name, command, echo=True):
        print('RUN:', shlex.join(command), flush=True)
        item = {'command': command, 'started_utc': datetime.now(timezone.utc).isoformat()}
        self.manifest['steps'][name] = item
        self.save()
        with (self.output/(name+'.log')).open('w') as log:
            self.child = subprocess.Popen(command, env=self.env, cwd=ROOT,
                                          start_new_session=True, text=True,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in self.child.stdout:
                log.write(line)
                log.flush()
                if echo:
                    print(line, end='', flush=True)
            item['exit_code'] = self.child.wait()
        self.child = None
        item['finished_utc'] = datetime.now(timezone.utc).isoformat()
        self.save()
        if item['exit_code']:
            raise RuntimeError(f'{name} failed; see {self.output / (name + ".log")}')


def main():
    parser, args = arguments()
    inputs = input_paths(args)
    output = resolve_output(args.output, 'benchmarks', 'profiling',
                            f'{args.dataset}_{args.backend}_batch{args.batch}')
    commands = capture_commands(args, inputs, output)
    if args.dry_run:
        print('OUTPUT:', output)
        for command in commands.values():
            print(shlex.join(command))
        return 0

    for path in [args.binary, *inputs.values()]:
        if not path.is_file():
            parser.error(f'Missing file: {path}')
    if not os.access(args.binary, os.X_OK):
        parser.error(f'Binary is not executable: {args.binary}')
    for tool in args.tools:
        if shutil.which(tool) is None:
            parser.error(f'{tool} is not available inside this container')
    if 'nsys' in args.tools:
        cache = args.binary.resolve().parent.parent/'CMakeCache.txt'
        if not cache.is_file() or 'MVH_ENABLE_PROFILING:BOOL=ON' not in cache.read_text():
            parser.error('NSYS stage timing requires a build configured with -DMVH_ENABLE_PROFILING=ON')
    if output.exists():
        parser.error(f'Output already exists: {output}; choose a new directory')

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(ROOT/'build/mvh_rt/optix_runtime') + ':/usr/local/cuda/lib64'
    output.mkdir(parents=True, exist_ok=False)
    print('OUTPUT:', output, flush=True)
    manifest = {'complete': False, 'options': vars(args).copy(), 'steps': {},
                'ncu_early_termination': not args.ncu_complete_search,
                'timing_note': 'Profiler overhead/replay time is not normal search performance.'}
    manifest['options'] = {k: str(v) if isinstance(v, Path) else v for k, v in manifest['options'].items()}
    runner = Runner(output, env, manifest)
    signal.signal(signal.SIGINT, runner.interrupt)
    signal.signal(signal.SIGTERM, runner.interrupt)
    runner.save()
    try:
        artifacts = [args.binary.resolve(), *inputs.values()]
        build = args.binary.resolve().parent.parent
        artifacts.extend(p for p in [build/'gpu_rt.ptx', build/'gpu_rt_custom.ptx', build/'CMakeCache.txt']
                         if p.is_file())
        manifest['sha256'] = {str(p): sha256(p) for p in artifacts}
        for tool in args.tools:
            runner.run(tool+'_version', [tool, '--version'], echo=False)
        for name, command in commands.items():
            runner.run(name, command, echo=not name.endswith('_summary'))
            if name == 'nsys':
                if not (output/'nsys_capture.nsys-rep').is_file():
                    raise RuntimeError('NSYS did not produce a report')
                # A timeline alone does not prove that the full search finished.
                manifest['nsys_psm_sha256'] = sha256(output/'nsys_results/mvh_psms.tsv')
                if 'search_seconds\t' not in (output/'nsys_results/run_summary.tsv').read_text():
                    raise RuntimeError('NSYS search has no completed run summary')
            elif name == 'nsys_summary':
                if 'mvh/search/database' not in (output/'nsys_summary.log').read_text():
                    raise RuntimeError('NSYS report has no application stage ranges; rebuild with profiling ON')
            elif name == 'nsys_csv':
                timing_csv = output/'timings_nvtx_sum.csv'
                if not timing_csv.is_file() or 'mvh/search/database' not in timing_csv.read_text():
                    raise RuntimeError('NSYS did not export application timing CSV')
                manifest['timing_reports'] = [p.name for p in sorted(output.glob('timings_*.csv'))]
            elif name == 'ncu':
                if not (output/'ncu_capture.ncu-rep').is_file():
                    raise RuntimeError('NCU captured no report; check kernel filter and launch skip')
                ncu_log = (output/'ncu.log').read_text()
                collected = len(re.findall(r'==PROF== Profiling "', ncu_log))
                manifest['ncu_collected_launches'] = collected
                if collected != args.ncu_launch_count:
                    raise RuntimeError(f'NCU collected {collected} launches, expected {args.ncu_launch_count}; '
                                       'check --ncu-launch-skip/count and available scoring launches')
                if args.ncu_complete_search:
                    manifest['ncu_psm_sha256'] = sha256(output/'ncu_results/mvh_psms.tsv')
        manifest['complete'] = True
        runner.save()
        print('FINISHED:', output, flush=True)
        if 'ncu' in args.tools and not args.ncu_complete_search:
            print('NCU intentionally stopped after the selected launches; ncu_results is partial.', flush=True)
        return 0
    except KeyboardInterrupt:
        manifest['interrupted'] = True
        return 130
    except Exception as error:
        manifest['error'] = str(error)
        print('ERROR:', error, file=sys.stderr)
        return 1
    finally:
        runner.save()
        if os.geteuid() == 0:
            owner = ROOT.stat()
            for path in [output, *output.rglob('*')]:
                if not path.is_symlink():
                    os.chown(path, owner.st_uid, owner.st_gid)


if __name__ == '__main__':
    sys.exit(main())
