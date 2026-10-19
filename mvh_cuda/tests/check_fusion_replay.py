"""Validate threaded CPU-only and GPU fusion modes, chunk offsets and bad candidates."""
import argparse, ctypes, csv, json, os, shutil, struct, subprocess, tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--search',required=True);p.add_argument('--replay',required=True);p.add_argument('--root',required=True);a=p.parse_args()
root=Path(a.root);parent=root/'output/tests/fusion_replay';parent.mkdir(parents=True,exist_ok=True)
def run(cmd,env=None):
 r=subprocess.run(list(map(str,cmd)),capture_output=True,text=True,env=env)
 assert r.returncode==0,r.stdout+r.stderr
with tempfile.TemporaryDirectory(dir=parent) as tmp:
 out=Path(tmp);search=out/'search'
 run([a.search,'-f',root/'mvh_cuda/tests/data/sample.ft2','-c',root/'mvh_cuda/tests/data/search.cfg','-fasta',root/'mvh_cuda/tests/data/proteins.fasta','--peptide-batch-size','3','--capture-matching','-o',search])
 cap=search/'matching_capture';meta=json.loads((cap/'COMPLETE.json').read_text());assert meta['candidate_count']==1
 class Scan(ctypes.Structure):
  _fields_=[('peakOffset',ctypes.c_uint64),('hubOffset',ctypes.c_uint64),('candidateOffset',ctypes.c_uint64)]+[(k,ctypes.c_int) for k in ['peaks','lowest','highest','candidates','skip','topCount']]+[('counts',ctypes.c_int*9),('totalBins',ctypes.c_int),('lower',ctypes.c_double),('upper',ctypes.c_double)]
 scans=(cap/'fusion_scans.bin').read_bytes();assert len(scans)==ctypes.sizeof(Scan)
 skip=Scan.from_buffer_copy(scans);skip.skip=1;(cap/'fusion_scans.bin').write_bytes(scans+bytes(skip))
 c=struct.unpack('<4i',(cap/'candidates.i32x4').read_bytes());invalid=(*c[:3],0);skipped=(c[0],c[1],1,c[3])
 n=meta['query_count'];assert n>0
 # Nonempty, invalid, nonempty and skipped groups require nonzero query bases.
 candidates=[c,invalid,c,skipped]
 (cap/'candidates.i32x4').write_bytes(b''.join(struct.pack('<4i',*x) for x in candidates))
 (cap/'query_mz.f64').write_bytes((cap/'query_mz.f64').read_bytes()*2)
 (cap/'query_offsets.u64').write_bytes(struct.pack('<5Q',0,n,n,2*n,2*n))
 (cap/'validity.u8').write_bytes(bytes([1,0,1,2]))
 meta.update(scan_count=2,candidate_count=4,query_count=2*n);(cap/'COMPLETE.json').write_text(json.dumps(meta))
 checks=[]
 for name,threads,backends in [('cpu1',1,'cpu'),('cpu4',4,'cpu'),('all',4,'cpu,cuda,rt'),('all_no_postcheck',4,'cpu,cuda,rt')]:
  env=os.environ.copy()
  if backends=='cpu':env['CUDA_VISIBLE_DEVICES']=''
  run([a.replay,'--capture',cap,'--output',out/name,'--threads',threads,'--backends',backends,'--repeats','2','--warmup','1','--chunk-groups','1','--rt-postcheck','off' if name=='all_no_postcheck' else 'on'],env)
  result=json.loads((out/name/'result.json').read_text());assert result['rt_postcheck']==(name!='all_no_postcheck') and result['results_equivalent'];assert result['complete'] and result['threads_actual']==threads and result['groups']==4 and result['result_mismatches']==0
  rows=list(csv.DictReader((out/name/'comparison.tsv').open(),delimiter='\t'));checks.append([r['summary_checksum'] for r in rows]);assert len(set(checks[-1]))==1
 assert len(set(sum(checks,[])))==1
 # Explicit prefix selection must agree on queries/groups and run all modes.
 run([a.replay,'--capture',cap,'--output',out/'prefix','--threads','1','--backends','cpu','--chunk-groups','1','--max-chunks','2','--repeats','1','--warmup','0'])
 result=json.loads((out/'prefix/result.json').read_text());assert result['groups']==2 and result['queries']==n
print('FUSION REPLAY CONTRACT PASSED')
