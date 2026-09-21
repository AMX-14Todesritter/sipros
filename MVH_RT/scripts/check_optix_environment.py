#!/usr/bin/env python3
"""Read-only dependency probe; never installs headers or changes driver configuration."""
import argparse
import ctypes
import json
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--optix-root', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
roots = [args.optix_root] if args.optix_root else []
roots += [Path('/usr/local'), Path('/opt'), Path('/usr/include')]
headers = []
for root in roots:
    if root and root.exists():
        headers += [str(p) for p in root.glob('**/optix.h')]
report = {'nvcc': shutil.which('nvcc'), 'optix_headers': sorted(set(headers)),
          'driver_library_loadable': False, 'driver_library_error': None}
try:
    ctypes.CDLL('libnvoptix.so.1')
    report['driver_library_loadable'] = True
except OSError as error:
    report['driver_library_error'] = str(error)
result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                        capture_output=True, text=True)
report['gpu'] = result.stdout.strip()
report['ready_for_build_and_runtime_attempt'] = bool(report['nvcc'] and headers and report['driver_library_loadable'])
text = json.dumps(report, indent=2) + '\n'
print(text, end='')
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
raise SystemExit(0 if report['ready_for_build_and_runtime_attempt'] else 2)
