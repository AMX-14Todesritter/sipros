"""Verify upstream identity except the documented class-priority peak matcher."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

module = Path(__file__).resolve().parents[1]
root = module.parent
sys.path.insert(0, str(root/'mvh/tests'))
from source_methods import methods

manifest = json.loads((module / 'original/manifest.json').read_text())
expected = set(manifest['files'])
actual = {str(p.relative_to(module/'original')) for folder in ('src','include')
          for p in (module/'original'/folder).rglob('*') if p.is_file()}
assert actual == expected, 'Unexpected or missing original files'
for name, sha in manifest['files'].items():
    data = (module/'original'/name).read_bytes()
    original = subprocess.check_output(['git', '-c', 'safe.directory='+str(root), '-C', str(root),
                                        'show', manifest['source_commit']+':'+name])
    assert hashlib.sha256(original).hexdigest() == sha, name
    if name == 'src/ms2scan.cpp':
        # Only the class-priority matcher intentionally differs from upstream.
        current_text, original_text = data.decode(), original.decode()
        current_matcher = methods(current_text, 'PeakList')['findNear']
        original_matcher = methods(original_text, 'PeakList')['findNear']
        canonical = methods((root/'mvh/original/src/ms2scan.cpp').read_text(), 'PeakList')['findNear']
        assert current_matcher == canonical, 'CPU/CUDA reference matchers diverged'
        restored = current_text.replace(current_matcher, original_matcher, 1)
        assert restored.encode() == original, 'Unexpected original change: '+name
    else:
        assert data == original, 'Changed original: '+name
print(f'PASS: {len(expected)} files checked against {manifest["source_commit"]}; only PeakList::findNear may differ')
