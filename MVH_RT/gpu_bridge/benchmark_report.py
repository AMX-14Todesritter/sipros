"""Render saved benchmark results without rerunning the search."""
import json
from pathlib import Path


def write_report(directory):
    directory = Path(directory)
    r = json.loads((directory / 'report.json').read_text())
    lines = ['# CPU / CUDA / GPU-RT benchmark', '',
             'Search includes RT initialization. Scoring includes ion generation, matching and MVH calculation. CPU scoring is not separately timed.', '',
             '| Backend | Runs | Wall s | Search s | Scoring s | RT setup s | Host peak GiB | Device peak GiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for backend, a in r.get('averages', {}).items():
        n = sum(x['backend'] == backend for x in r['runs'])
        vals = []
        for k in ['wall_seconds', 'search_seconds', 'scoring_kernel_seconds', 'rt_setup_seconds']:
            v = a[k]
            vals.append('—' if backend == 'cpu' and k in ('scoring_kernel_seconds', 'rt_setup_seconds') else
                        f"{v['mean']:.3f}" + (f" ± {v['stdev']:.3f}" if v['stdev'] is not None else ''))
        lines.append(f'| {backend} | {n} | ' + ' | '.join(vals) +
                     f" | {a['host_hwm_mib']['mean']/1024:.2f} | {a['device_peak_mib']['mean']/1024:.2f} |")
    lines += ['', 'Times are mean ± sample standard deviation when repeated. Single runs do not measure variability. Device memory is device-wide and includes background processes.', '',
              '| Run | Batches | RT builds | PSM rows | SHA-256 |', '|---|---:|---:|---:|---|']
    for x in r['runs']:
        if x['exit_code'] == 0:
            lines.append(f"| {x['name']} | {len(x['batches'])} | {len(x['rt_setup'])} | {x['summary']['retained_psm_count']} | {x['psm_sha256']} |")
    hashes = {k: {x['psm_sha256'] for x in r['runs'] if x['backend']==k and x['exit_code']==0}
              for k in {x['backend'] for x in r['runs']}}
    lines += ['', '## Output comparisons', '']
    for left, right in [('cpu','cuda'), ('cuda','rt-triangle'), ('cuda','rt-instanced'), ('rt-triangle','rt-instanced'), ('cuda','rt-custom'), ('rt-triangle','rt-custom')]:
        if hashes.get(left) and hashes.get(right):
            lines.append(f'- {left} / {right}: ' + ('identical' if len(hashes[left] | hashes[right]) == 1 else 'different'))
    lines += ['', 'The legacy triangle RT backends have known float-zero/boundary differences; evaluate rt-custom with the score-impact validation script. PSM row differences are not an accuracy percentage. A different output does not automatically fail the performance run.', '',
              'Raw commands, hashes, per-batch timings and memory metrics: report.json. Build/test records and source/configuration snapshot: parent directory when using run_benchmark.sh.', '']
    (directory / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    write_report(p.parse_args().directory)
