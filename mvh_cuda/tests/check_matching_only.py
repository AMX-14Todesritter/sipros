"""Known classes, class-zero hits, no-handle scans, empty groups and chunk bases."""
import argparse,csv,json,struct,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--strict',required=True);p.add_argument('--raw',required=True);p.add_argument('--root',required=True);a=p.parse_args()
parent=Path(a.root)/'output/tests/matching_only';parent.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(dir=parent) as tmp:
 out=Path(tmp);cap=out/'capture';cap.mkdir()
 scans=[(0,0,4,100,102,0,0.,1000.),(4,6,0,0,0,0,0.,1000.),(4,8,1,608,608,0,0.,1000.)]
 (cap/'scans.bin').write_bytes(b''.join(struct.pack('<QQiiiidd',*s) for s in scans))
 (cap/'peaks.f64').write_bytes(struct.pack('<5d',100.,100.0078125,101.,102.,608.84259))
 (cap/'classes.i32').write_bytes(struct.pack('<5i',1,0,2,3,3))
 (cap/'hub.i16').write_bytes(struct.pack('<10h',0,2,2,3,3,4,-1,-1,0,1))
 groups=[(0,0,0,2),(0,0,1,2),(0,0,0,2),(0,0,2,3),(0,0,0,2)]
 (cap/'candidates.i32x4').write_bytes(b''.join(struct.pack('<4i',*c) for c in groups))
 queries=[100.,100.0078125,150.,608.83258996688005,101.004,102.02,99.97]
 (cap/'query_mz.f64').write_bytes(struct.pack('<7d',*queries))
 (cap/'query_offsets.u64').write_bytes(struct.pack('<6Q',0,2,3,3,4,7))
 (cap/'validity.u8').write_bytes(bytes([1]*5))
 (cap/'COMPLETE.json').write_text(json.dumps({'format_version':1,'endianness':'little','scan_count':3,'peak_count':5,'hub_count':10,'candidate_count':5,'query_count':7,'fragment_tolerance_da':.01}))
 expected=1469598103934665603
 for v in [1,0,0,0,2,0,0]:expected=((expected^v)*1099511628211)&((1<<64)-1)
 for name,binary,chunk in [('strict_chunks',a.strict,1),('strict_full',a.strict,0),('raw_chunks',a.raw,1)]:
  cmd=[binary,'--capture',str(cap),'--output',str(out/name),'--chunk-groups',str(chunk),'--repeats','2','--warmup','1','--threads','2']
  r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
  info=json.loads((out/name/'result.json').read_text());assert info['complete'] and info['cuda_oracle_mismatches']==0 and info['queries']==7
  rows=list(csv.DictReader((out/name/'comparison.tsv').open(),delimiter='\t'))
  assert int(rows[0]['class_checksum'])==expected and int(rows[0]['positive_class_queries'])==2
  if name.startswith('strict'):assert info['results_equivalent'] and info['rt_postcheck'] and int(rows[1]['class_checksum'])==expected
  else:assert not info['rt_postcheck'] and info['results_equivalent']==(info['class_mismatches']==0)
  times=list(csv.DictReader((out/name/'samples.tsv').open(),delimiter='\t'));assert len(times)==info['chunk_count']*4
 print('MATCHING-ONLY CONTRACT PASSED')
