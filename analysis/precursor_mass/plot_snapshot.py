"""Read Sipros snapshot v1; plot neutral-mass hypotheses without changing search."""
import argparse
import csv
import hashlib
import html
import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('snapshot', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--bin-da', type=float, default=100)
    args = ap.parse_args()
    if not math.isfinite(args.bin_da) or args.bin_da <= 0:
        raise ValueError('bin width must be positive and finite')
    if args.output.exists():
        raise ValueError('output must not exist')
    data = args.snapshot.read_bytes()
    assert data[:8] == b'SMVHSP01'
    assert zlib.crc32(memoryview(data)[:-4]) == struct.unpack_from('<I', data, len(data)-4)[0]
    pos = 8

    def read(fmt):
        nonlocal pos
        values = struct.unpack_from('<'+fmt, data, pos)
        pos += struct.calcsize('<'+fmt)
        assert pos <= len(data)-4
        return values[0] if len(values) == 1 else values

    def skip(n):
        nonlocal pos
        pos += n
        assert pos <= len(data)-4

    def string():
        n = read('Q')
        start = pos
        skip(n)
        return data[start:pos].decode()

    def array(width):
        skip(read('Q')*width)

    assert read('I') == 1
    identity, config, input_file, suffix = [string() for _ in range(4)]
    skip(32)
    scans = read('Q')
    skipped = np.zeros(scans, dtype=bool)
    for i in range(scans):
        skip(36)
        flags = read('III')
        assert all(x in (0, 1) for x in flags)
        skipped[i] = flags[2]
        string(); string()
        array(4); array(8)
        skip(36)
        array(4); array(8); array(4)
        skip(8)
        array(4)
    count = read('Q')
    assert pos + count*20 == len(data)-4
    precursors = np.frombuffer(data, dtype=[('mass', '<f8'), ('charge', '<i4'), ('scan', '<u8')], count=count, offset=pos)
    assert np.all(np.isfinite(precursors['mass'])) and np.all(precursors['mass'] >= 0)
    assert np.all(np.diff(precursors['mass']) >= 0)
    assert np.all(precursors['scan'] < scans) and np.all(precursors['charge'] > 0)
    active = precursors[~skipped[precursors['scan']]]
    masses = active['mass']
    charges = sorted(int(z) for z in np.unique(active['charge']))
    edges = np.arange(math.floor(masses.min()/args.bin_da), math.floor(masses.max()/args.bin_da)+2)*args.bin_da
    hist = np.array([np.histogram(masses[active['charge'] == z], edges)[0] for z in charges])
    assert int(hist.sum()) == len(active)

    def quantiles(a):
        return dict(zip(('min', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'max'), map(float, np.quantile(a, [0, .25, .5, .75, .9, .95, .99, 1]))))

    gaps = np.diff(np.unique(masses))
    low = np.full(scans, np.inf)
    high = np.full(scans, -np.inf)
    np.minimum.at(low, active['scan'], masses)
    np.maximum.at(high, active['scan'], masses)
    spans = (high-low)[np.isfinite(low)]
    summary = dict(snapshot=str(args.snapshot), snapshot_sha256=hashlib.sha256(data).hexdigest(), build_identity=identity,
                   input_file=input_file, scan_count=scans, skipped_scans=int(skipped.sum()), precursor_count=count,
                   active_precursors=len(active), excluded_precursors=count-len(active), bin_da=args.bin_da,
                   neutral_mass_da=quantiles(masses), unique_mass_count=len(np.unique(masses)),
                   adjacent_distinct_mass_gap_da=quantiles(gaps), within_scan_mass_span_da=quantiles(spans),
                   adjacent_distinct_gap_fraction_below_1da=float(np.mean(gaps < 1)),
                   charge_counts={z:int((active['charge'] == z).sum()) for z in charges},
                   fraction_1000_to_3000_da=float(np.mean((masses >= 1000) & (masses < 3000))),
                   counting='All precursor hypotheses linked to non-skipped scans; no mass or scan deduplication in histogram.',
                   interpretation='Mass bins are descriptive workload groups, not shared-theoretical-spectrum equivalence classes.')
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (args.output/'mass_bins.tsv').open('w') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['lower_da_inclusive', 'upper_da_exclusive', 'total']+[f'charge_{z}' for z in charges])
        for i in range(len(edges)-1):
            w.writerow([edges[i], edges[i+1], int(hist[:, i].sum())]+hist[:, i].tolist())
    # Dependency-free SVG; no plotting library installation required.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720">', '<rect width="1200" height="720" fill="#f7f9fc"/>']
    def text(x, y, value, size=14, color='#26364a', anchor='start'):
        svg.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial,sans-serif" font-size="{size}" text-anchor="{anchor}">{html.escape(str(value))}</text>')
    text(80, 45, 'Experimental MS2 precursor neutral-mass distribution', 25)
    text(80, 74, f'Pan_062822_X1iso5 | {len(active):,} hypotheses | {scans-int(skipped.sum()):,} non-skipped scans | {args.bin_da:g} Da bins', 15)
    palette = ['#2675b9', '#31a89a', '#e9a33a', '#9a69b3', '#df6b68', '#7b8da5', '#744e32', '#b0a92f', '#dd79b5']
    for j, z in enumerate(charges):
        x = 80+j*110
        svg.append(f'<rect x="{x}" y="96" width="15" height="15" fill="{palette[j%len(palette)]}"/>')
        text(x+22, 109, f'charge {z}')
    x0, y0, width, height = 90, 145, 1050, 440
    ymax = math.ceil(hist.sum(axis=0).max()/1000)*1000
    for k in range(6):
        y = y0+height-height*k/5
        svg.append(f'<line x1="{x0}" y1="{y}" x2="{x0+width}" y2="{y}" stroke="#dce3ec"/>')
        text(x0-12, y+5, f'{ymax*k/5:,.0f}', anchor='end')
    text(x0, 133, 'Precursor hypotheses per bin', 13)
    bw = width/(len(edges)-1)
    for i in range(len(edges)-1):
        bottom = y0+height
        for j, z in enumerate(charges):
            h = hist[j, i]/ymax*height
            svg.append(f'<rect x="{x0+i*bw+.5:.2f}" y="{bottom-h:.2f}" width="{max(.1,bw-1):.2f}" height="{h:.2f}" fill="{palette[j%len(palette)]}"><title>{edges[i]:g}–{edges[i+1]:g} Da; charge {z}: {hist[j,i]:,}</title></rect>')
            bottom -= h
    tick = max(500, math.ceil((edges[-1]-edges[0])/10/500)*500)
    for m in np.arange(math.ceil(edges[0]/tick)*tick, edges[-1]+1, tick):
        x = x0+(m-edges[0])/(edges[-1]-edges[0])*width
        text(x, 609, f'{m:,.0f}', anchor='middle')
    text(615, 642, 'Precursor neutral mass (Da)', 17, anchor='middle')
    text(80, 678, 'All mass hypotheses retained; skipped scans excluded. Bar width is not a matching tolerance.', 14)
    text(80, 702, 'Similar mass does not imply the same peptide, modifications, or reusable theoretical fragments.', 14)
    svg.append('</svg>')
    (args.output/'precursor_mass.svg').write_text('\n'.join(svg))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
