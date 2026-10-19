"""Known-work fixture checks counters independently of the matching code."""
import argparse
import csv
import json
from pathlib import Path
import struct
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--normal', required=True)
parser.add_argument('--diagnostics', required=True)
parser.add_argument('--root', required=True)
parser.add_argument('--nvtx')
args = parser.parse_args()
parent = Path(args.root)/'output/tests/matching_diagnostics'
parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=parent) as tmp:
    out = Path(tmp)
    capture = out/'fixture'
    capture.mkdir()
    records = [(0, 0, 2, 100, 102, 0, 0., 1000.),
               (2, 6, 0, 0, -1, 0, 0., 1000.),
               (2, 6, 1, 200, 200, 0, 0., 1000.)]
    (capture/'scans.bin').write_bytes(b''.join(struct.pack('<QQiiiidd', *r) for r in records))
    (capture/'peaks.f64').write_bytes(struct.pack('<ddd', 100.125, 102.125, 200.125))
    (capture/'classes.i32').write_bytes(struct.pack('<iii', 0, 2, 3))
    (capture/'hub.i16').write_bytes(struct.pack('<hhhhhhhh', 0, 1, -1, -1, 1, 2, 0, 1))
    (capture/'candidates.i32x4').write_bytes(b''.join(struct.pack('<iiii', i, 0, i, 2) for i in range(3)))
    (capture/'query_offsets.u64').write_bytes(struct.pack('<QQQQ', 0, 6, 7, 9))
    queries = [100.125, 100.25, 101.25, 99.5, 100.124, 102.125, 100., 200.125, 201.5]
    (capture/'query_mz.f64').write_bytes(struct.pack('<'+'d'*len(queries), *queries))
    (capture/'COMPLETE.json').write_text(json.dumps(dict(format_version=1, endianness='little',
        scan_count=3, peak_count=3, hub_count=8, candidate_count=3,
        query_count=9, fragment_tolerance_da=0.01)))
    runs = [('normal', args.normal), ('diagnostic', args.diagnostics)]
    if args.nvtx:
        runs.append(('nvtx', args.nvtx))
    for name, binary in runs:
        result = subprocess.run([binary, '--capture', str(capture), '--output', str(out/name),
                                 '--repeats', '3', '--warmup', '2', '--chunk-groups', '2'],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+result.stderr
    normal = json.loads((out/'normal/result.json').read_text())
    diagnostic = json.loads((out/'diagnostic/result.json').read_text())
    assert normal['performance_valid'] and not diagnostic['performance_valid']
    assert 'cuda_over_rt' not in diagnostic
    assert not (out/'diagnostic/comparison.tsv').exists()
    assert normal['peak_index_mismatches'] == diagnostic['peak_index_mismatches'] == 0
    def signatures(path):
        return {r['backend']: (r['peak_hits'], r['checksum']) for r in csv.DictReader(path.open(), delimiter='\t')}
    assert signatures(out/'normal/comparison.tsv') == signatures(out/'diagnostic/diagnostic_timings_NOT_PERFORMANCE.tsv')
    counts = {(r['backend'], r['metric']): r['value'] for r in
              csv.DictReader((out/'diagnostic/diagnostics.tsv').open(), delimiter='\t')}
    expected = {'queries': 9, 'peak_hits': 4, 'misses': 5, 'bucket_visits': 6,
                'empty_bucket_visits': 1, 'queries_touching_only_empty_buckets': 1,
                'empty_scan_queries': 1, 'outside_integer_range_queries': 2,
                'peak_checks': 5, 'max_peak_checks_per_query': 1,
                'zero_peak_check_queries': 4, 'checks_hist_0': 4, 'checks_hist_1': 5}
    for key, value in expected.items():
        assert int(counts['cuda', key]) == value, (key, counts['cuda', key], value)
    for key, value in {'queries': 9, 'peak_hits': 4, 'misses': 5,
                       'optix_trace_calls': 8, 'no_handle_queries': 1}.items():
        assert int(counts['rt-custom', key]) == value, (key, counts['rt-custom', key])
    assert counts['rt-custom', 'bvh_node_visits'] == 'unavailable'
    if args.nvtx:
        nvtx = json.loads((out/'nvtx/result.json').read_text())
        assert nvtx['nvtx_enabled'] and not nvtx['diagnostic_build'] and not nvtx['performance_valid']
        assert 'cuda_over_rt' not in nvtx and not (out/'nvtx/diagnostics.tsv').exists()
        assert signatures(out/'normal/comparison.tsv') == signatures(out/'nvtx/profiling_timings_NOT_PERFORMANCE.tsv')
    # Independent summary checksum includes raw hits, class bins and the
    # scorer's class-zero/miss convention. No query-level RT claim is made.
    expected_summaries = [[3, 0, 1, 0, 5, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 1, 1, 0, 0, 0, 0, 0]]
    checksum = 0
    for summary in expected_summaries:
        for value in summary:
            checksum = (checksum * 1099511628211 + value) % (1 << 64)
    for name, binary in runs:
        destination = out/f'{name}_candidate'
        result = subprocess.run([binary, '--capture', str(capture), '--output', str(destination),
                                 '--output-mode', 'candidate', '--repeats', '3', '--warmup', '2',
                                 '--chunk-groups', '2'], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads((destination/'result.json').read_text())
        assert report['peak_index_mismatches'] is None and report['class_mismatches'] is None
        assert report['candidate_summary_mismatches'] == 0
        table = ('comparison.tsv' if name == 'normal' else
                 'diagnostic_timings_NOT_PERFORMANCE.tsv' if name == 'diagnostic' else
                 'profiling_timings_NOT_PERFORMANCE.tsv')
        assert signatures(destination/table) == {b: ('4', str(checksum)) for b in ['cuda', 'rt-custom']}
print('PASS: exact counters, empty scan/bucket/range, class-zero hit, chunk offsets, '
      'repeat-independent counts, normal/diagnostic identity and timing isolation')
