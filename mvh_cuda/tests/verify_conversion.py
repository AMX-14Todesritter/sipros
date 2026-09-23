from pathlib import Path
import json,hashlib
from source_methods import methods
m=Path(__file__).resolve().parents[1];cpu=m.parent/'mvh'
manifest=json.loads((m/'CPU_BASELINE.json').read_text())
for name,sha in manifest['files'].items():
 assert hashlib.sha256((cpu/name).read_bytes()).hexdigest()==sha, 'CPU baseline changed: '+name
changes={'src/database_search.cpp':{'processPeptideArrayMvh','assignPeptides2Scans','searchDatabaseMvh'},'src/spectrum_input.cpp':{'preProcessAllMs2Mvh'},'src/mvh_scan_vector.cpp':{'preMvh'}}
for filename,allowed in changes.items():
 original=methods((cpu/filename).read_text(),'MvhScanVector');converted=methods((m/filename).read_text(),'MvhScanVector')
 assert original.keys()==converted.keys()
 for name in original:
  if name not in allowed:assert original[name]==converted[name], 'Serial method changed: '+name
 assert '#pragma omp' not in (m/filename).read_text()
print('PASS: CPU baseline unchanged; unmodified methods identical; GPU assignment/search orchestration explicitly allowed; all four OpenMP loops replaced')
