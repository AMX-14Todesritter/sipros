"""Bounded-memory access to matching capture v1; no NumPy dependency."""
import json
import mmap
import struct
from pathlib import Path


class Capture:
    SCAN = struct.Struct('<QQiiiidd')
    CANDIDATE = struct.Struct('<iiii')

    def __init__(self, directory):
        self.path = Path(directory)
        self.meta = json.loads((self.path / 'COMPLETE.json').read_text())
        if self.meta['format_version'] != 1 or self.meta['endianness'] != 'little':
            raise ValueError('Unsupported capture format')
        self.files = []
        self.maps = {}
        try:
            for name in ['scans.bin', 'candidates.i32x4', 'query_offsets.u64',
                         'query_mz.f64', 'validity.u8', 'peaks.f64', 'classes.i32', 'hub.i16']:
                f = (self.path / name).open('rb')
                self.files.append(f)
                self.maps[name] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if f.seek(0, 2) else b''
        except BaseException:
            self.close()
            raise

    def close(self):
        for value in self.maps.values():
            if isinstance(value, mmap.mmap):
                value.close()
        for f in self.files:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def scan(self, index):
        if not 0 <= index < self.meta['scan_count']:
            raise IndexError(index)
        return self.SCAN.unpack_from(self.maps['scans.bin'], index * self.SCAN.size)

    def group(self, index):
        if not 0 <= index < self.meta['candidate_count']:
            raise IndexError(index)
        candidate = self.CANDIDATE.unpack_from(self.maps['candidates.i32x4'], index * 16)
        start, end = struct.unpack_from('<QQ', self.maps['query_offsets.u64'], index * 8)
        return candidate, start, end, self.maps['validity.u8'][index]

    def query_mz(self, index):
        if not 0 <= index < self.meta['query_count']:
            raise IndexError(index)
        return struct.unpack_from('<d', self.maps['query_mz.f64'], index * 8)[0]

    def iter_group_queries(self, index):
        candidate, start, end, _ = self.group(index)
        for i in range(start, end):
            yield candidate[2], self.query_mz(i)
