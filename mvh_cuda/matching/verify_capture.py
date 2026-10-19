"""Validate capture structure, sample query bounds and hash every byte."""
import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from capture_io import Capture

parser = argparse.ArgumentParser()
parser.add_argument('directory')
parser.add_argument('--write', action='store_true')
args = parser.parse_args()
with Capture(args.directory) as c:
    m = c.meta
    sizes = {'scans.bin': m['scan_count'] * 48,
             'candidates.i32x4': m['candidate_count'] * 16,
             'query_offsets.u64': (m['candidate_count'] + 1) * 8,
             'query_mz.f64': m['query_count'] * 8,
             'validity.u8': m['candidate_count'],
             'peaks.f64': m['peak_count'] * 8,
             'classes.i32': m['peak_count'] * 4,
             'hub.i16': m['hub_count'] * 2}
    for name, size in sizes.items():
        assert len(c.maps[name]) == size, (name, size)
    assert struct.unpack_from('<Q', c.maps['query_offsets.u64'])[0] == 0
    assert struct.unpack_from('<Q', c.maps['query_offsets.u64'], m['candidate_count'] * 8)[0] == m['query_count']
    # Full monotonicity and candidate-reference validation, bounded memory.
    flags = Counter()
    previous = 0
    for i in range(m['candidate_count']):
        candidate, start, end, flag = c.group(i)
        peptide, precursor, scan, charge = candidate
        assert start == previous and start <= end <= m['query_count'], i
        assert 0 <= peptide < m['peptide_count'] and precursor >= 0, i
        assert 0 <= scan < m['scan_count'] and charge > 0, i
        assert flag in (0, 1, 2), (i, flag)
        if flag == 2:
            assert start == end
        flags[flag] += 1
        previous = end
    assert flags[1] == m['valid_candidates']
    assert flags[0] == m['invalid_candidates']
    assert flags[2] == m['skipped_candidates']
    for i in range(m['scan_count']):
        peak, hub, count, low, high, skip, lower, upper = c.scan(i)
        assert 0 <= peak <= peak + count <= m['peak_count']
        if count:
            assert 0 <= hub <= hub + 2 * (high - low + 1) <= m['hub_count']
    sampled = 0
    for i in range(0, m['candidate_count'], max(1, m['candidate_count'] // 4096)):
        candidate, start, end, _ = c.group(i)
        lower, upper = c.scan(candidate[2])[-2:]
        for j in {start, end - 1} if end > start else ():
            mz = c.query_mz(j)
            assert math.isfinite(mz) and lower <= mz <= upper, (i, j, mz)
            sampled += 1
    hashes = {}
    for name in [*sizes, 'COMPLETE.json', 'scan_sources.tsv']:
        path = c.path / name
        if not path.exists():
            continue
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8388608), b''):
                h.update(block)
        hashes[name] = h.hexdigest()
    result = {'verified': True, 'candidate_count': m['candidate_count'],
              'query_count': m['query_count'], 'sampled_query_bounds': sampled,
              'binary_bytes': sum(sizes.values()), 'sha256': hashes}
    if args.write:
        (c.path / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
