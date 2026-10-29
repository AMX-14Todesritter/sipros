"""Validate same-batch GPU scoring-metadata reuse against a frozen baseline."""
import csv,hashlib,json,os,re,shutil,statistics,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';D=E/'device_metadata_reuse_v1_20261029';MP=D/'manifest.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(m):
 p=MP.with_suffix('.tmp');p.write_text(json.dumps(m,indent=2));p.replace(MP)
def call(cmd,log,env=None):
 with log.open('w') as f:subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 lock=(E/'.benchmark.lock').open('a');import fcntl;fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 m=json.loads(MP.read_text()) if MP.exists() else {'status':'preparing','batch':2000000,'native_architecture':120,'theoretical_ion_cache':'off','warmups':1,'measured_repeats':5,'runs':[]}
 m.setdefault('resume_notes',[]).append('Resume after user opened the development container; preserve completed measurements and use identical saved binaries.');save(m)
 s=D/'optimized_source';s.mkdir(exist_ok=True)
 for name in ('mvh_cuda','MVH_RT'):
  if not (s/name).exists():shutil.copytree(R/name,s/name,ignore=shutil.ignore_patterns('__pycache__'))
 for name in ('MSToolkit','shared'):
  p=s/name
  if not p.is_symlink():p.symlink_to(os.path.relpath(R/name,s),target_is_directory=True)
 protected=['mvh_cuda/cuda/scoring.cuh','mvh_cuda/cuda/theoretical.cuh','MVH_RT/gpu_bridge/device.cu','MVH_RT/gpu_bridge/custom_device.cu','MVH_RT/gpu_bridge/custom_geometry.cu','mvh_cuda/include/peptide_batch.h','mvh_cuda/cuda/peptide_generation.cu']
 for f in protected:assert sha(D/'baseline_source'/f)==sha(s/f),f
 m['unchanged_hashes']={f:sha(s/f) for f in protected};save(m)
 sys.path.insert(0,str(R/'evaluation/scripts'));from prepare_batch_binary import restore_multi
 old=subprocess.check_output(['git','show','82522a4:mvh_cuda/src/spectrum_input.cpp'],cwd=R,text=True)
 for variant in ('baseline','optimized'):
  if (D/(variant+'_contracts.log')).exists() and '100% tests passed' in (D/(variant+'_contracts.log')).read_text():continue
  source=D/(variant+'_source');p=source/'mvh_cuda/src/spectrum_input.cpp';p.write_text(restore_multi(p.read_text(),old))
  b=D/(variant+'_build');m['status']='building_'+variant;save(m)
  call(['cmake','-S',str(source/'mvh_cuda'),'-B',str(b),'-G','Unix Makefiles','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=120','-DMVH_CUDA_ENABLE_RT=ON','-DMVH_ENABLE_EVALUATION_TIMING=ON','-DMVH_ENABLE_FLOW_COUNTERS=OFF','-DMVH_ENABLE_PROFILING=OFF','-DOPTIX_ROOT='+str(R/'build/mvh_rt/optix_sdk')],D/(variant+'_configure.log'))
  call(['cmake','--build',str(b),'--target','sipros_mvh_cuda','mvh_cuda_contract','mvh_gpu_rt_contract','sequence_ids_contract','sequence_ids_reuse_contract','peptide_batch_contract','-j','4'],D/(variant+'_build.log'))
 env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(R/'build/mvh_rt/optix_runtime')+':/usr/local/cuda/lib64',PYTHONDONTWRITEBYTECODE='1')
 for k in ('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR','SIPROS_DIAGNOSTIC_HOT_REPLAY','SIPROS_CONTINUOUS_REPLAY','SIPROS_PEAK_AUDIT_DIR'):env.pop(k,None)
 env['OPTIX_CACHE_PATH']=str(D/'optix_cache');env['CUDA_CACHE_PATH']=str(D/'cuda_cache');Path(env['OPTIX_CACHE_PATH']).mkdir(exist_ok=True);Path(env['CUDA_CACHE_PATH']).mkdir(exist_ok=True)
 for variant in ('baseline','optimized'):
  if (D/(variant+'_contracts.log')).exists() and '100% tests passed' in (D/(variant+'_contracts.log')).read_text():continue
  call(['ctest','--test-dir',str(D/(variant+'_build')),'-R','^(cuda_semantic_contract|gpu_rt_custom_contract|gpu_rt_instanced_contract|sequence_ids_contract|sequence_ids_reuse_contract|peptide_batch_contract)$','--output-on-failure'],D/(variant+'_contracts.log'),env)
 # Same raw scan records for both modes. Primary input keeps only the first Z pair.
 headers=[];blocks=[];block=[]
 with (R/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2').open() as f:
  for line in f:
   if line.startswith('S\t'):
    if block:blocks.append(''.join(block))
    if len(blocks)==256:break
    block=[line]
   elif block:block.append(line)
   else:headers.append(line)
 assert len(blocks)==256
 multi=''.join(headers)+''.join(blocks);(D/'multi.FT2').write_text(multi)
 (D/'primary.FT2').write_text(''.join('\t'.join(line.split()[:3])+'\n' if line.startswith('Z\t') else line for line in multi.splitlines(keepends=True)))
 n=0
 with (R/'raw/Marine_fw_3rev.fasta').open() as f,(D/'marine4096.fasta').open('w') as out:
  for line in f:
   if line.startswith('>'):
    n+=1
    if n>4096:break
   out.write(line)
 shutil.copy2(R/'experiments/Regular.cfg',D/'Regular.cfg')
 m['inputs']={p.name:sha(p) for p in [D/'multi.FT2',D/'primary.FT2',D/'marine4096.fasta',D/'Regular.cfg']};m['status']='running';save(m)
 sys.path.insert(0,str(R/'evaluation/scripts'));from run import GpuMemory,log_metrics
 gpu=GpuMemory()
 for mode in ('primary','multi'):
  for repeat in range(6):
   for backend in (('cuda','rt-custom') if repeat%2==0 else ('rt-custom','cuda')):
    for variant in (('baseline','optimized') if repeat%2==0 else ('optimized','baseline')):
     if any(x['mode']==mode and x['backend']==backend and x['variant']==variant and x['repeat']==repeat for x in m['runs']):continue
     rid=f'{mode}_{backend}_{variant}_r{repeat}';out=D/'runs'/rid;out.parent.mkdir(exist_ok=True);log=D/(rid+'.log')
     cmd=[str(D/(variant+'_build/bin/sipros_mvh_cuda')),'-f',str(D/(mode+'.FT2')),'-fasta',str(D/'marine4096.fasta'),'-c',str(D/'Regular.cfg'),'-o',str(out),'-t','1','--match-backend',backend,'--peptide-batch-size','2000000','--peptide-generation','cuda','--result-restoration','final','--spectrum-cache','host']
     peak=gpu.read();host=0
     with log.open('w') as f:
      p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
      while p.poll() is None:
       peak=max(peak,gpu.read())
       try:
        for value in re.findall(r'^VmHWM:\s+(\d+)',Path(f'/proc/{p.pid}/status').read_text(),re.M):host=max(host,int(value)*1024)
       except FileNotFoundError:pass
       try:p.wait(timeout=.1)
       except subprocess.TimeoutExpired:pass
     assert p.returncode==0,rid
     text=log.read_text();x=log_metrics(text);bs=x['batches'];t=x['totals'];assert bs and all(int(b['cached_ions'])==0 and int(b['cache_charge_stride'])==0 for b in bs)
     row={'mode':mode,'backend':backend,'variant':variant,'repeat':repeat,'warmup':repeat==0,'scoring_batches':len(bs),'candidates':int(t['candidates']),'fused_seconds':t['fused_device_seconds'],'pack_seconds':t['pack_seconds'],'upload_seconds':t['allocation_upload_seconds'],'complete_seconds':sum(t[k] for k in ('pack_seconds','gpu_service_seconds','replay_verify_seconds')),'search_seconds':float(re.findall(r'^Search seconds: (.*)$',text,re.M)[-1]),'device_peak_bytes':peak,'host_peak_bytes':host,'psm_sha256':sha(out/'mvh_psms.tsv')}
     if variant=='optimized':
      metadata=[dict(re.findall(r'(\w+)=(\d+)',l)) for l in text.splitlines() if l.startswith('[CUDA peptide metadata]')];assert len(metadata)==len(bs)
      assert all(int(v['uploaded_bytes'])==4*int(v['peptides']) and int(v['reused_offset_bytes'])==8*int(v['peptides']) for v in metadata)
      row['metadata_upload_bytes']=sum(int(v['uploaded_bytes']) for v in metadata)
     m['runs'].append(row);save(m)
 m['status']='completed';save(m)
 lines=['# Same-batch GPU metadata reuse v1','','Frozen baseline versus optimized source on native sm120; fixed compute75 OptiX shaders, theoretical-ion cache off, serial processing. Real X1 first 256 scans and Marine first 4096 FASTA records. Primary and multi-precursor inputs, CUDA and RT-custom. One excluded warmup plus five measured processes per cell, alternating variant/backend order. Device-wide memory sampled at 100ms; host VmHWM observed. No large benchmark.','','| Mode | Backend | Variant | Pack ms | Upload/allocate ms | Complete scoring ms | Search s | GPU peak GiB |','|---|---|---|---:|---:|---:|---:|---:|']
 for mode in ('primary','multi'):
  for backend in ('cuda','rt-custom'):
   cells=[x for x in m['runs'] if x['mode']==mode and x['backend']==backend];assert len({x['psm_sha256'] for x in cells})==1,'Version changed retained PSMs';assert len({x['candidates'] for x in cells})==1
   for v in ('baseline','optimized'):
    rs=[x for x in cells if x['variant']==v and not x['warmup']]
    med=lambda k:statistics.median(x[k] for x in rs)
    lines.append(f'| {mode} | {backend} | {v} | {med("pack_seconds")*1000:.3f} | {med("upload_seconds")*1000:.3f} | {med("complete_seconds")*1000:.3f} | {med("search_seconds"):.3f} | {med("device_peak_bytes")/2**30:.3f} |')
 lines+=['','All retained PSM hashes and candidate totals match baseline/optimized within each mode/backend; six semantic contracts pass in each build. The device metadata ABI and matching/theory/MVH implementations are unchanged. Host upload per peptide drops from sizeof(PeptideInput)=16 bytes to sizeof(int)=4 bytes; preprocessed uint64 offsets are reused, then immediately released after one assembly kernel. Stable sequence IDs and result provenance stay on the host. This does not eliminate the generation download/CPU Peptide objects or all host packing. Retaining offsets temporarily adds up to 8 bytes/peptide before metadata assembly, plus 4 bytes/peptide for transient IDs; both are freed before scoring. Measure larger workloads separately before claiming throughput or peak-memory gains.','']
 (D/'report.md').write_text('\n'.join(lines));m['psm_equivalence_verified']=True;save(m)
 with (D/'measurements.csv').open('w') as f:
  cols=list(dict.fromkeys(k for row in m['runs'] for k in row));w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(m['runs'])
if __name__=='__main__':
 try:main()
 except BaseException as error:
  if MP.exists():
   m=json.loads(MP.read_text());m.update(status='failed',error=repr(error));save(m)
  raise
