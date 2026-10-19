import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import shutil
import struct

parser = argparse.ArgumentParser()
parser.add_argument('--binary', required=True)
parser.add_argument('--root', required=True)
parser.add_argument('--replay')
args = parser.parse_args()
root = Path(args.root)
parent = root / 'output/tests/matching_capture'
parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=parent) as directory:
    out = Path(directory)
    common = [args.binary, '-f', str(root/'mvh_cuda/tests/data/sample.ft2'),
              '-c', str(root/'mvh_cuda/tests/data/search.cfg'),
              '-fasta', str(root/'mvh_cuda/tests/data/proteins.fasta'),
              '--peptide-batch-size', '3']
    for name in ['baseline', 'capture']:
        cmd = common + ['-o', str(out/name)]
        if name == 'capture':
            cmd.append('--capture-matching')
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    capture = out/'capture/matching_capture'
    meta = json.loads((capture/'COMPLETE.json').read_text())
    assert meta['cache_charge_stride'] > 0
    assert meta['candidate_count'] > 0 and meta['query_count'] > 0
    result = subprocess.run(['python3', '-B', str(root/'mvh_cuda/matching/verify_capture.py'),
                             str(capture)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    hashes = [hashlib.sha256((out/name/'mvh_psms.tsv').read_bytes()).hexdigest()
              for name in ['baseline', 'capture']]
    assert hashes[0] == hashes[1], 'Capture changed production results'
    if args.replay:
        # Exercise nonzero query bases, empty groups and chunk boundaries.
        assert meta['candidate_count'] == 1
        fixture = out/'replay_input'
        shutil.copytree(capture, fixture)
        n = meta['query_count']
        (fixture/'query_mz.f64').write_bytes((capture/'query_mz.f64').read_bytes() * 2)
        (fixture/'candidates.i32x4').write_bytes((capture/'candidates.i32x4').read_bytes() * 4)
        (fixture/'query_offsets.u64').write_bytes(struct.pack('<QQQQQ', 0, n, n, 2*n, 2*n))
        (fixture/'validity.u8').write_bytes(bytes([1, 1, 1, 1]))
        meta.update(candidate_count=4, query_count=2*n, valid_candidates=4)
        (fixture/'COMPLETE.json').write_text(json.dumps(meta))
        for replicas in [1, 3]:
            replay_dir = out/f'replay{replicas}'
            replay = subprocess.run([args.replay, '--capture', str(fixture), '--output',
                                     str(replay_dir), '--chunk-groups', str(replicas),
                                     '--replicas', str(replicas), '--warmup', '1',
                                     '--repeats', '3'], capture_output=True, text=True)
            assert replay.returncode == 0, replay.stdout + replay.stderr
            result = json.loads((replay_dir/'result.json').read_text())
            assert result['complete'] and result['query_count'] == 2*n*replicas
            assert result['scan_count'] == meta['scan_count']*replicas
            assert result['peak_index_mismatches'] == 0
print('PASS: cached-theory capture, structure, group boundaries, production PSM identity')
