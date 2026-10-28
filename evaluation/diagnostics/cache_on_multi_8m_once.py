"""One cache-enabled CUDA/RT-custom Marine/X1 multi-precursor 8M comparison."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,re
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';D=E/'cache_on_multi_8m_once_20261028';B=D/'build';MP=D/'manifest.json'
def save(m):
 p=MP.with_suffix('.tmp');p.write_text(json.dumps(m,indent=2));p.replace(MP)
def call(cmd,log,env=None):
 with log.open('w') as f:subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 D.mkdir(exist_ok=False);m={'status':'building','native_architecture':120,'optix_ptx_target':120,'precursor':'multi','batch':8000000,'repeats':1,'warmups':0,'cache_policy':'enabled; remove fixed 32M-key ceiling only; preserve both free-memory guards and direct fallback','production_default_unchanged':True};save(m)
 ref=E/'build/gpu_multi_optix_ptx_ab_20261028';S=D/'source';shutil.copytree(ref/'source',S,symlinks=True)
 engine=S/'mvh_cuda/cuda/engine.cu';s=engine.read_text();assert s.count('constexpr bool useIonCache=false;')==1;s=s.replace('constexpr bool useIonCache=false;','constexpr bool useIonCache=true;');assert s.count('keys < 32000000 && ')==1;s=s.replace('keys < 32000000 && ','')
 a='        if (ionCount <= freeBytes / (2 * sizeof(double))) {';assert a in s;s=s.replace(a,'        std::cout << "[ION CACHE capacity] keys=" << keys << " ions=" << ionCount << " free_bytes=" << freeBytes << " eligible=" << (ionCount <= freeBytes / (2 * sizeof(double))) << std::endl;\n'+a);engine.write_text(s)
 p=S/'mvh_cuda/CMakeLists.txt';p.write_text(p.read_text().replace('--gpu-architecture=compute_75','--gpu-architecture=compute_120'))
 protected=['mvh_cuda/cuda/scoring.cuh','mvh_cuda/cuda/theoretical.cuh','MVH_RT/gpu_bridge/device.cu','MVH_RT/gpu_bridge/custom_device.cu','MVH_RT/gpu_bridge/custom_geometry.cu']
 for f in protected:assert (S/f).read_bytes()==(ref/'source'/f).read_bytes()
 m['protected_hashes']={f:hashlib.sha256((S/f).read_bytes()).hexdigest() for f in protected};save(m)
 call(['cmake','-S',str(S/'mvh_cuda'),'-B',str(B),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=120','-DMVH_CUDA_ENABLE_RT=ON','-DMVH_ENABLE_EVALUATION_TIMING=ON','-DMVH_ENABLE_FLOW_COUNTERS=OFF','-DMVH_ENABLE_PROFILING=OFF','-DOPTIX_ROOT='+str(R/'build/mvh_rt/optix_sdk')],D/'configure.log')
 call(['cmake','--build',str(B),'--target','sipros_mvh_cuda','-j','4'],D/'build.log')
 m['status']='running';m['binary_sha256']=hashlib.sha256((B/'bin/sipros_mvh_cuda').read_bytes()).hexdigest();save(m)
 cmd=[sys.executable,'-B',str(R/'evaluation/scripts/run.py'),'--id',D.name,'--category','pilot','--scans',str(R/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2'),'--fasta',str(R/'raw/Marine_fw_3rev.fasta'),'--config',str(R/'experiments/Regular.cfg'),'--gpu-binary',str(B/'bin/sipros_mvh_cuda'),'--backends','cuda','rt-custom','--batch','8000000','--threads','1','--repeats','1','--warmups','0','--min-device-free-gib','1'];m['command']=cmd;save(m)
 call(cmd,D/'runner.log');n=json.loads((E/'results/pilot'/D.name/'manifest.json').read_text());rows=[]
 for x in n['runs']:
  bs=x['batches'];t=x['totals'];cached=[b for b in bs if int(b['cached_ions'])>0]
  log=(E/'results/pilot'/D.name/(x['name']+'.log')).read_text();decisions=[dict(re.findall(r'(\w+)=(\d+)',l)) for l in log.splitlines() if l.startswith('[ION CACHE capacity]')]
  rows.append({'backend':x['backend'],'scoring_batches':len(bs),'cached_batches':len(cached),'direct_batches':len(bs)-len(cached),'cached_candidate_fraction':sum(int(b['candidates']) for b in cached)/int(t['candidates']),'cached_ions_total':sum(int(b['cached_ions']) for b in cached),'cache_preparation_seconds':t['theory_seconds'],'fused_seconds':t['fused_device_seconds'],'complete_scoring_seconds':sum(t[k] for k in ('pack_seconds','gpu_service_seconds','replay_verify_seconds')),'search_seconds':float(x['summary']['search_seconds']),'device_peak_GiB':x['device_peak_bytes']/2**30,'candidates':int(t['candidates']),'psm_sha256':x['psm_sha256'],'capacity_decisions':decisions})
 m.update(status='completed',results=rows);save(m)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  if MP.exists():
   m=json.loads(MP.read_text());m.update(status='failed',error=repr(error));save(m)
  raise
