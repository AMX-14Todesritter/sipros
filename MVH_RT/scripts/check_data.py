#!/usr/bin/env python3
"""Replay the saved matching inputs without constructing an integer bucket index."""
import csv
import json
import math
from pathlib import Path
import struct
import sys


def load_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'test_data'
    state = json.loads((data / 'scan_state.json').read_text())
    reference = json.loads((data / 'validation.json').read_text())
    peaks = load_rows(data / 'experimental_peaks.tsv')
    ions = load_rows(data / 'theoretical_ions.tsv')
    expected_matches = load_rows(data / 'reference_matches.tsv')
    assert state['pPeakList_is_null'] and not state['integer_bucket_index_present']
    assert len(peaks) == state['retained_peak_count'] == 101
    assert len(ions) == 13
    masses = [float(row['mz']) for row in peaks]
    classes = [int(row['intensity_class']) for row in peaks]
    assert masses == sorted(masses) and len(set(masses)) == len(masses)
    counts = state['intenClassCounts']
    assert [classes.count(i + 1) for i in range(len(counts) - 1)] == counts[:-1]
    assert sum(counts) == state['totalPeakBins']
    key = [0] * len(counts)
    matched_indices = []
    for ion in ions:
        mz = float(ion['mz'])
        if not state['mzLowerBound'] <= mz <= state['mzUpperBound']:
            continue
        # min() preserves first-encountered ties; a strict tolerance is essential.
        nearest = min(range(len(masses)), key=lambda i: abs(masses[i] - mz))
        if abs(masses[nearest] - mz) < state['fragment_tolerance']:
            key[classes[nearest] - 1] += 1
        else:
            nearest = -1
            key[-1] += 1
        matched_indices.append(nearest)
    assert matched_indices == [int(row['experimental_peak_index']) for row in expected_matches]
    assert key == reference['mvhKey'] == [6, 1, 1, 5]
    table_bytes = (data / 'ln_factorial_table.bin').read_bytes()
    assert len(table_bytes) == (state['totalPeakBins'] + 1) * 8
    table = struct.unpack('<' + 'd' * (len(table_bytes) // 8), table_bytes)
    def ln_choose(n, k):
        return (table[n] - table[n - k]) - table[k]
    score = ln_choose(sum(counts), sum(key)) - sum(ln_choose(n, k) for n, k in zip(counts, key))
    assert math.isclose(score, reference['mvh_score'], rel_tol=0, abs_tol=1e-9)
    print(f'PASS: pre-bucket snapshot, {len(peaks)} peaks, {len(ions)} ions, '
          f'{sum(key[:-1])} matches; replay score={score:.15f}')


if __name__ == '__main__':
    main()
