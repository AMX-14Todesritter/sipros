#!/usr/bin/env python3
"""Summarize completed standalone timings and cross-check reference snapshots."""
import argparse
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
a = p.parse_args()
runs = json.loads((a.output / 'validation.json').read_text())
reference = json.loads((a.reference / 'validation.json').read_text())
if not runs or not reference or len({r['sha256'] for r in runs + reference}) != 1:
    raise RuntimeError('Baseline/reference hashes differ or records are empty')
stats = {}
for threads in sorted({r['threads'] for r in runs}):
    chosen = [r for r in runs if r['threads'] == threads]
    stats[threads] = {}
    for metric in ('config_and_load_seconds', 'preprocess_seconds', 'prepare_seconds', 'search_seconds'):
        values = [float(r['summary'][metric]) for r in chosen]
        stats[threads][metric] = dict(values=values, median=statistics.median(values), minimum=min(values), maximum=max(values))
result = dict(stats=stats, sha256=runs[0]['sha256'], rows=runs[0]['rows'])
with (a.output / 'benchmark_summary.json').open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
lines = ['| 线程 | 搜索三次值 | 搜索中位数 | 搜索范围 | 前置中位数 |', '|---|---|---|---|---|']
for threads, metrics in stats.items():
    score = metrics['search_seconds']
    values = ', '.join(f'{v:.5f}' for v in score['values'])
    lines.append('| {} | {} | {:.5f} | {:.5f}–{:.5f} | {:.5f} |'.format(
        threads, values, score['median'], score['minimum'], score['maximum'], metrics['prepare_seconds']['median']))
print('\n'.join(lines))
if 1 in stats and 4 in stats:
    print('Search median speedup: {:.3f}x'.format(stats[1]['search_seconds']['median'] / stats[4]['search_seconds']['median']))
print('SHA-256:', result['sha256'])
