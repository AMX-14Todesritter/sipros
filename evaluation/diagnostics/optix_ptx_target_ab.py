"""Compare compute75/compute120 shader PTX in one isolated sm120 native binary."""
import csv,fcntl,hashlib,json,os,re,shutil,statistics,subprocess,sys,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';D=E/'optix_ptx_target_ab_20261028';B=E/'build/gpu_multi_optix_ptx_ab_20261028';MP=D/'manifest.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(m):
 p=MP.with_suffix('.tmp');p.write_text(json.dumps(m,indent=2));p.replace(MP)
def call(cmd,log,env=None):
 with log.open('w') as f:subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 D.mkdir(exist_ok=True);lock=(E/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 m={'status':'building','native_architecture':120,'shader_targets':[75,120],'precursor':'multi','cache':'off','runs':[],'contracts':[]};save(m)
 if not (B/'build_manifest.json').exists():call([sys.executable,'-B',str(R/'evaluation/scripts/prepare_batch_binary.py'),'--precursor-mode','multi','--cuda-architectures','120','--build-name',B.name],D/'build.log')
 call(['cmake','--build',str(B),'--target','mvh_gpu_rt_contract','-j','4'],D/'contract_build.log')
 (D/'ptx75').mkdir(exist_ok=True);(D/'ptx120').mkdir(exist_ok=True)
 for name in ('gpu_rt.ptx','gpu_rt_custom.ptx'):shutil.copy2(B/name,D/'ptx75'/name)
 m['compile_commands']=[]
 for src,name in [('device.cu','gpu_rt.ptx'),('custom_device.cu','gpu_rt_custom.ptx')]:
  cmd=['/usr/local/cuda/bin/nvcc','--ptx','--std=c++17','--gpu-architecture=compute_120','--generate-line-info','--fmad=false','-I'+str(R/'build/mvh_rt/optix_sdk/include'),'-I'+str(B/'source/mvh_cuda/cuda'),'-I'+str(B/'source/MVH_RT/gpu_bridge'),str(B/'source/MVH_RT/gpu_bridge'/src),'-o',str(D/'ptx120'/name)]
  m['compile_commands'].append(cmd);save(m);call(cmd,D/(name+'_compile120.log'))
 # Fixed real-data multi-precursor control: first 256 X1 scans, first 4096 Marine records.
 headers=[];blocks=[];block=[]
 with (R/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2').open() as f:
  for l in f:
   if l.startswith('S\t'):
    if block:blocks.append(''.join(block))
    if len(blocks)==256:break
    block=[l]
   elif block:block.append(l)
   else:headers.append(l)
 assert len(blocks)==256;(D/'scans.FT2').write_text(''.join(headers)+''.join(blocks))
 count=0
 with (R/'raw/Marine_fw_3rev.fasta').open() as f,(D/'marine4096.fasta').open('w') as out:
  for l in f:
   if l.startswith('>'):
    count+=1
    if count>4096:break
   out.write(l)
 shutil.copy2(R/'experiments/Regular.cfg',D/'Regular.cfg')
 env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(R/'build/mvh_rt/optix_runtime')+':/usr/local/cuda/lib64',PYTHONDONTWRITEBYTECODE='1')
 for k in ('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR','SIPROS_DIAGNOSTIC_HOT_REPLAY','SIPROS_CONTINUOUS_REPLAY','SIPROS_PEAK_AUDIT_DIR'):env.pop(k,None)
 def install(target):
  for name in ('gpu_rt.ptx','gpu_rt_custom.ptx'):shutil.copy2(D/f'ptx{target}'/name,B/name)
  env['OPTIX_CACHE_PATH']=str(D/f'cache_optix{target}');env['CUDA_CACHE_PATH']=str(D/f'cache_cuda{target}')
  Path(env['OPTIX_CACHE_PATH']).mkdir(exist_ok=True);Path(env['CUDA_CACHE_PATH']).mkdir(exist_ok=True)
 for target in (75,120):
  install(target)
  for backend,arg in [('rt-custom','custom'),('rt-instanced','instanced')]:
   log=D/f'contract_{target}_{backend}.log';call([str(B/'mvh_gpu_rt_contract'),arg],log,env);m['contracts'].append({'target':target,'backend':backend,'status':'passed'});save(m)
 m.update(status='running',native_binary_sha256=sha(B/'bin/sipros_mvh_cuda'),ptx_sha256={str(t):{n:sha(D/f'ptx{t}'/n) for n in ('gpu_rt.ptx','gpu_rt_custom.ptx')} for t in (75,120)});save(m)
 for rep in range(6):
  for target in ((75,120) if rep%2==0 else (120,75)):
   install(target)
   for backend in (('rt-instanced','rt-custom') if rep%2==0 else ('rt-custom','rt-instanced')):
    rid=f'ptx{target}_{backend}_r{rep}';out=D/'runs'/rid;out.parent.mkdir(exist_ok=True)
    cmd=[str(B/'bin/sipros_mvh_cuda'),'-f',str(D/'scans.FT2'),'-fasta',str(D/'marine4096.fasta'),'-c',str(D/'Regular.cfg'),'-o',str(out),'-t','1','--match-backend',backend,'--peptide-batch-size','2000000','--peptide-generation','cuda','--result-restoration','final','--spectrum-cache','host']
    call(cmd,D/(rid+'.log'),env);text=(D/(rid+'.log')).read_text();bs=[dict(re.findall(r'(\w+)=([^\s]+)',l)) for l in text.splitlines() if l.startswith('[CUDA scoring]')]
    assert bs and all(b['cached_ions']=='0' and b['cache_charge_stride']=='0' for b in bs)
    row={'target':target,'backend':backend,'repeat':rep,'warmup':rep==0,'command':cmd,'candidates':sum(int(b['candidates']) for b in bs),'fused_seconds':sum(float(b['fused_device_seconds']) for b in bs),'complete_scoring_seconds':sum(float(b[k]) for b in bs for k in ('pack_seconds','gpu_service_seconds','replay_verify_seconds')),'psm_sha256':sha(out/'mvh_psms.tsv')}
    assert row['candidates']>0;m['runs'].append(row);save(m)
 install(75);m['status']='completed';save(m)
 lines=['# OptiX PTX target A/B','','One native sm120 binary; only its isolated runtime shader PTX is exchanged. Multi-precursor, cache off, 256 real X1 scans + 4096 Marine records, 2M generated-peptide batch. One excluded warmup and five measured processes per target/backend, alternating order. No large benchmark or profiler.','','| Backend | PTX target | Fused median ms | Fused min–max ms | Complete scoring median ms |','|---|---|---:|---|---:|']
 for backend in ('rt-instanced','rt-custom'):
  hashes=set()
  for t in (75,120):
   rs=[r for r in m['runs'] if r['backend']==backend and r['target']==t and not r['warmup']];assert len(rs)==5
   hashes.update(r['psm_sha256'] for r in rs);v=[r['fused_seconds']*1000 for r in rs];cs=[r['complete_scoring_seconds']*1000 for r in rs]
   lines.append(f'| {backend} | compute_{t} | {statistics.median(v):.3f} | {min(v):.3f}–{max(v):.3f} | {statistics.median(cs):.3f} |')
  assert len(hashes)==1,'PTX target changed PSM results'
 assert len({r['candidates'] for r in m['runs']})==1
 lines+=['','Both targets passed both RT semantic contracts. PSM hashes are identical between targets and repeats within each backend. Candidate totals match all cells. These ordinary small-workload event intervals include first-use effects within each process and do not establish full-database performance. The production PTX targets and baseline build were not modified.','']
 (D/'report.md').write_text('\n'.join(lines));m['psm_equivalence_verified']=True;save(m)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if MP.exists():
   m=json.loads(MP.read_text());m.update(status='failed',error=repr(e));save(m)
  raise
