"""Verify every transplanted source/header against both manifest and Git baseline."""
import hashlib
import json
import subprocess
from pathlib import Path

module = Path(__file__).resolve().parents[1]
root = module.parent
manifest = json.loads((module / 'original/manifest.json').read_text())
expected = set(manifest['files'])
actual = {str(p.relative_to(module/'original')) for folder in ('src','include')
          for p in (module/'original'/folder).rglob('*') if p.is_file()}
assert actual == expected, 'Unexpected or missing original files'
for name, sha in manifest['files'].items():
    data = (module/'original'/name).read_bytes()
    original = subprocess.check_output(['git', '-c', 'safe.directory='+str(root), '-C', str(root),
                                        'show', manifest['source_commit']+':'+name])
    assert data == original, 'Changed original: '+name
    assert hashlib.sha256(data).hexdigest() == sha, name
print(f'PASS: {len(expected)} files byte-identical to {manifest["source_commit"]}')
