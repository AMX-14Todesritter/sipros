"""Serial reproducible fused-scoring runs. Execute inside project container."""
import argparse, csv, ctypes, datetime, fcntl, hashlib, itertools, json, os
from pathlib import Path
import re, signal, statistics, subprocess, time
from inventory import ROOT, OUT, sha
BACKENDS=['cpu','cuda','rt-instanced','rt-custom']

def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def fields(line):return dict(re.findall(r'(\w+)=([^\s]+)',line))

def log_metrics(text):
    batches=[fields(l) for l in text.splitlines() if l.startswith('[CUDA scoring]')]
    assignments=[fields(l) for l in text.splitlines() if l.startswith('[CUDA assignment]')]
    setup=[fields(l) for l in text.splitlines() if l.startswith('[RT GPU setup]')]
    totals={}
    for k in sorted({k for b in batches for k in b}):
        try:totals[k]=sum(float(b[k]) for b in batches if k in b)
        except ValueError:pass
    return {'batches':batches,'assignment_batches':assignments,'rt_setup':setup,'totals':totals,
            'generated_peptides':sum(int(b['generated']) for b in assignments),
            'rt_setup_seconds':sum(float(b.get('seconds',0)) for b in setup)}

class GpuMemory:
    class Memory(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    def __init__(self):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1');self.handle=ctypes.c_void_p()
        if self.lib.nvmlInit_v2() or self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle)):raise RuntimeError('NVML initialization failed')
    def read(self):
        m=self.Memory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(m)):raise RuntimeError('NVML memory query failed')
        return m.used

def compare(left,right,destination):
    def groups(path):
        with path.open() as f:
            rows=csv.DictReader(f,delimiter='\t')
            # Run inputs must have identical hashes; scan_id is meaningful within that input.
            yield from ((k,list(v)) for k,v in itertools.groupby(rows,key=lambda r:r['scan_id']))
    stats={'spectra_with_results':0,'top1_peptide_agree':0,'top1_peptide_charge_agree':0,'top1_score_exact':0,
           'topk_set_intersection':0,'topk_set_union':0,'common_candidate_scores_differ':0,'candidate_score_pairs_compared':0,
           'common_candidate_proteins_differ':0,'max_absolute_score_difference':0.0,'spectrum_groups_differ':0}
    stats.update(reference_only_spectra=0,compared_only_spectra=0)
    def aligned():
        # Array position is only an ordering aid for identical-input files;
        # original scan IDs below remain the compared spectrum identities.
        la,lb=iter(groups(left)),iter(groups(right));a=next(la,None);b=next(lb,None)
        while a is not None or b is not None:
            ia=int(a[1][0]['scan_index']) if a else float('inf')
            ib=int(b[1][0]['scan_index']) if b else float('inf')
            if ia==ib:
                yield a,b;a=next(la,None);b=next(lb,None)
            elif ia<ib:
                yield a,None;a=next(la,None)
            else:
                yield None,b;b=next(lb,None)
    with destination.open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['scan_id','issue','reference','compared'])
        for a,b in aligned():
            if a is None or b is None:
                present=a or b;key='reference_only_spectra' if a else 'compared_only_spectra'
                stats[key]+=1;stats['spectrum_groups_differ']+=1
                stats['topk_set_union']+=len(present[1])
                writer.writerow([present[0],'spectrum_result_coverage',json.dumps(a),json.dumps(b)])
                continue
            if a[0]!=b[0]:raise RuntimeError('Original scan identities differ at aligned input record')
            sid,ar=a;_,br=b;stats['spectra_with_results']+=1
            identity=lambda r:(r['peptide'],r['precursor_charge'])
            stats['top1_peptide_agree']+=ar[0]['peptide']==br[0]['peptide']
            stats['top1_peptide_charge_agree']+=identity(ar[0])==identity(br[0])
            stats['top1_score_exact']+=ar[0]['mvh_score']==br[0]['mvh_score']
            # List can contain multiple precursor masses/charges. Keep these in candidate identities.
            key=lambda r:(r['peptide'],r['original_peptide'],r['precursor_charge'],r['precursor_mass'])
            am={key(r):r for r in ar};bm={key(r):r for r in br}
            if len(am)!=len(ar) or len(bm)!=len(br):raise RuntimeError('Ambiguous candidate identity; comparison cannot deduplicate silently')
            common=am.keys()&bm.keys();union=am.keys()|bm.keys()
            stats['topk_set_intersection']+=len(common);stats['topk_set_union']+=len(union)
            different=False
            for k in sorted(common):
                x,y=am[k],bm[k];d=abs(float(x['mvh_score'])-float(y['mvh_score']))
                stats['candidate_score_pairs_compared']+=1
                stats['max_absolute_score_difference']=max(stats['max_absolute_score_difference'],d)
                if x['mvh_score']!=y['mvh_score']:
                    stats['common_candidate_scores_differ']+=1;different=True
                    writer.writerow([sid,'score',json.dumps(x),json.dumps(y)])
                if x['protein_names']!=y['protein_names']:stats['common_candidate_proteins_differ']+=1;different=True
            if am.keys()!=bm.keys():
                different=True;writer.writerow([sid,'candidate_set',json.dumps(sorted(am.keys()-bm.keys())),json.dumps(sorted(bm.keys()-am.keys()))])
            if identity(ar[0])!=identity(br[0]):
                different=True;writer.writerow([sid,'top1',json.dumps(ar[0]),json.dumps(br[0])])
            stats['spectrum_groups_differ']+=different
    stats['topk_set_jaccard']=stats['topk_set_intersection']/stats['topk_set_union'] if stats['topk_set_union'] else None
    stats['scope']='unfiltered retained MVH top lists; no FDR or biological-accuracy claim'
    return stats

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id',required=True);p.add_argument('--category',default='pilot',choices=['pilot','accuracy','batch_size','real_world','database_scaling','scan_scaling','end_to_end'])
    p.add_argument('--scans',type=Path,required=True);p.add_argument('--fasta',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'experiments/Regular.cfg')
    p.add_argument('--backends',nargs='+',choices=BACKENDS,default=BACKENDS[1:])
    p.add_argument('--batch',type=int,default=2000000);p.add_argument('--repeats',type=int,default=5);p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--peak-audit',action='store_true');p.add_argument('--max-host-rss-gib',type=float,default=0);p.add_argument('--continue-on-backend-failure',action='store_true');p.add_argument('--threads',type=int,default=4);p.add_argument('--verify',action='store_true');p.add_argument('--score-impact',action='store_true')
    p.add_argument('--gpu-binary',type=Path,default=OUT/'build/gpu/bin/sipros_mvh_cuda');p.add_argument('--cpu-binary',type=Path,default=OUT/'build/cpu/bin/sipros_mvh')
    a=p.parse_args()
    if (OUT/'manifests/LARGE_BENCHMARKS_PAUSED.json').exists() and a.category in ('real_world','batch_size','database_scaling','scan_scaling','end_to_end'):p.error('Large-scale performance benchmarks paused by user')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',a.id):p.error('invalid run id')
    if a.batch<1 or a.repeats<1 or a.warmups<0 or len(set(a.backends))!=len(a.backends):p.error('invalid batching/repetitions/backends')
    if 'cpu' in a.backends and (a.category in ('database_scaling','scan_scaling') or 'soil' in a.fasta.name.lower()):p.error('CPU excluded from Soil and scaling experiments')
    for path in (a.scans,a.fasta,a.config):
        if not path.is_file():p.error('missing input: '+str(path))
    for backend in a.backends:
        if not (a.cpu_binary if backend=='cpu' else a.gpu_binary).is_file():p.error('missing binary for '+backend)
    (OUT/'manifests').mkdir(parents=True,exist_ok=True)
    with (OUT/'.benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dest=OUT/'results'/a.category/a.id;dest.mkdir(parents=True,exist_ok=False)
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(OUT/'tmp'),SIPROS_OUTPUT_ROOT=str(OUT))
        env['LD_LIBRARY_PATH']=str(ROOT/'build/mvh_rt/optix_runtime')+':/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
        for key,folder in [('OPTIX_CACHE_PATH','optix'),('CUDA_CACHE_PATH','cuda')]:
            directory=OUT/'cache'/folder;directory.mkdir(parents=True,exist_ok=True);env[key]=str(directory)
        gpu=GpuMemory()
        def capture(cmd):return subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True).stdout
        source_before=capture(['git','diff','HEAD','--','mvh','mvh_cuda','MVH_RT','shared'])
        (dest/'source.diff').write_text(source_before)
        script_dir=dest/'suite_source';script_dir.mkdir()
        for script in (ROOT/'evaluation/scripts').glob('*.py'):(script_dir/script.name).write_bytes(script.read_bytes())
        manifest={'status':'running','id':a.id,'category':a.category,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
          'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':capture(['git','rev-parse','HEAD']).strip(),
          'source_diff_sha256':hashlib.sha256(source_before.encode()).hexdigest(),
          'suite_source_sha256':{p.name:sha(p) for p in script_dir.iterdir()},
          'source_sha256':{str(p.relative_to(ROOT)):sha(p) for base in ('mvh','mvh_cuda','MVH_RT','shared') for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.cu','.cuh')},
          'build_cache':{str(b.parents[1]/'CMakeCache.txt'):(b.parents[1]/'CMakeCache.txt').read_text() for b in {a.cpu_binary if x=='cpu' else a.gpu_binary for x in a.backends}},
          'inputs':{str(x.resolve()):sha(x) for x in (a.scans,a.fasta,a.config)},
          'binaries':{str(b):sha(b) for b in {a.cpu_binary if x=='cpu' else a.gpu_binary for x in a.backends}},
          'timing_scope':'fused_device_seconds excludes RT Params upload; kernel_seconds preserves outer event interval; theory cache preparation, search and process wall separate',
          'diagnostic':a.verify or a.score_impact or a.peak_audit or any(re.search(r'MVH_ENABLE_(?:FLOW_COUNTERS|PROFILING):[^=]*=ON', (b.parents[1]/'CMakeCache.txt').read_text()) for b in {a.cpu_binary if x=='cpu' else a.gpu_binary for x in a.backends}),'memory_scope':'100ms sampled device-wide NVML used, includes desktop; host VmRSS/VmHWM observed while child alive',
          'gpu_before':capture(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv']),
          'runs':[]}
        if a.peak_audit:manifest['isolated_audit_build']=json.loads((OUT/'manifests/build_peak_audit.json').read_text())
        mp=dest/'manifest.json';save(mp,manifest)
        active=None;failed_backends={}
        def stop(signum,frame):
            if active is not None and active.poll() is None:active.terminate()
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            for iteration in range(a.warmups+a.repeats):
                warm=iteration<a.warmups;rep=iteration-a.warmups+1
                order=a.backends[iteration%len(a.backends):]+a.backends[:iteration%len(a.backends)]
                for backend in order:
                    if backend in failed_backends:continue
                    name=f'{backend}_{"warmup"+str(iteration+1) if warm else "r"+str(rep)}'
                    directory=dest/name;binary=a.cpu_binary if backend=='cpu' else a.gpu_binary
                    cmd=[str(binary),'-f',str(a.scans.resolve()),'-fasta',str(a.fasta.resolve()),'-c',str(a.config.resolve()),'-o',str(directory),'-t',str(a.threads)]
                    if backend!='cpu':
                        cmd+=['--match-backend',backend,'--peptide-batch-size',str(a.batch),'--peptide-generation','cuda','--result-restoration','final','--spectrum-cache','host']
                        if a.verify:cmd+=['--verify-cuda']
                        if a.score_impact and backend!='cuda':cmd+=['--score-impact']
                    item={'name':name,'backend':backend,'warmup':warm,'repeat':rep if not warm else None,'command':cmd,'status':'running'}
                    manifest['runs'].append(item);save(mp,manifest);print('START '+name,flush=True)
                    before=gpu.read();peak=before;rss=0;hwm=0;start=time.perf_counter()
                    if a.peak_audit:env['SIPROS_PEAK_AUDIT_DIR']=str(directory/'peak_audit')
                    with (dest/(name+'.log')).open('w') as log:
                        active=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                        while active.poll() is None:
                            peak=max(peak,gpu.read())
                            try:
                                status=Path(f'/proc/{active.pid}/status').read_text()
                                for k,v in re.findall(r'^(VmRSS|VmHWM):\s+(\d+)',status,re.M):
                                    if k=='VmRSS':rss=max(rss,int(v)*1024)
                                    else:hwm=max(hwm,int(v)*1024)
                            except FileNotFoundError:pass
                            if a.max_host_rss_gib and rss>a.max_host_rss_gib*2**30:
                                item['resource_guard']='host RSS exceeded configured '+str(a.max_host_rss_gib)+' GiB'
                                active.terminate()
                            try:active.wait(timeout=.1)
                            except subprocess.TimeoutExpired:pass
                    item.update(exit_code=active.returncode,wall_seconds=time.perf_counter()-start,device_before_bytes=before,device_peak_bytes=peak,device_peak_delta_bytes=peak-before,host_sampled_rss_bytes=rss,host_observed_hwm_bytes=hwm)
                    if active.returncode:
                        item['status']='failed';failed_backends[backend]={'run':name,'exit_code':active.returncode,'log':str(dest/(name+'.log'))}
                        manifest['backend_failures']=failed_backends;save(mp,manifest)
                        if a.continue_on_backend_failure:
                            print('FAILED '+name+'; continuing independent backends',flush=True);continue
                        raise RuntimeError(name+' failed; see preserved log')
                    with (directory/'run_summary.tsv').open() as f:item['summary']=dict(list(csv.reader(f,delimiter='\t'))[1:])
                    item.update(log_metrics((dest/(name+'.log')).read_text()));item['psm_sha256']=sha(directory/'mvh_psms.tsv');item['status']='completed'
                    # Identical immutable exports share disk blocks via hard links; every run retains its own path.
                    prior=next((r for r in manifest['runs'][:-1] if r.get('status')=='completed' and r.get('psm_sha256')==item['psm_sha256']),None)
                    if prior:
                        current=directory/'mvh_psms.tsv';link=directory/'mvh_psms.shared.tmp'
                        os.link(dest/prior['name']/'mvh_psms.tsv',link);link.replace(current)
                        item['psm_shared_with']=prior['name']
                    save(mp,manifest);print('DONE '+name+' '+str(round(item['wall_seconds'],3))+'s',flush=True)
            measured=[r for r in manifest['runs'] if not r['warmup'] and r['status']=='completed']
            manifest['statistics']={}
            for backend in a.backends:
                if backend in failed_backends:continue
                runs=[r for r in measured if r['backend']==backend]
                if len(runs)!=a.repeats:raise RuntimeError('Incomplete measured repetitions for '+backend)
                metrics={'wall_seconds':[r['wall_seconds'] for r in runs],'search_seconds':[float(r['summary']['search_seconds']) for r in runs]}
                for key in ('kernel_seconds','fused_device_seconds','theory_seconds','gpu_service_seconds'):
                    if backend!='cpu' and all(key in r['totals'] for r in runs):metrics[key]=[r['totals'][key] for r in runs]
                manifest['statistics'][backend]={k:{'median':statistics.median(v),'min':min(v),'max':max(v),'stdev':statistics.stdev(v) if len(v)>1 else None,'n':len(v)} for k,v in metrics.items()}
                hashes={r['psm_sha256'] for r in runs};manifest['statistics'][backend]['repeat_psm_identical']=len(hashes)==1
                if len(hashes)!=1:raise RuntimeError('PSMs changed across repeats for '+backend)
            usable=[b for b in a.backends if b in manifest['statistics']]
            reference=next((r for r in measured if r['backend']==('cuda' if 'cuda' in usable else usable[0])),None) if usable else None
            manifest['comparison_reference']=reference['backend'] if reference else None
            manifest['comparisons']={}
            for backend in usable:
                if backend==reference['backend']:continue
                other=next(r for r in measured if r['backend']==backend)
                manifest['comparisons'][backend]=compare(dest/reference['name']/'mvh_psms.tsv',dest/other['name']/'mvh_psms.tsv',dest/(backend+'_differences.tsv'))
            if capture(['git','diff','HEAD','--','mvh','mvh_cuda','MVH_RT','shared'])!=source_before:raise RuntimeError('Production source changed during experiment; timings invalidated')
            if any(sha(Path(b))!=h for b,h in manifest['binaries'].items()):raise RuntimeError('Binary changed during experiment')
            manifest['status']='failed' if failed_backends else 'completed';manifest['measurements_validated']=True;save(mp,manifest)
            columns=['backend','repeat','wall_seconds','search_seconds','fused_event_seconds','fused_device_seconds','theory_seconds','gpu_service_seconds','candidates','generated_peptides','gpu_peak_bytes','host_hwm_bytes','psm_sha256']
            with (dest/'metrics.csv').open('w') as f:
                w=csv.writer(f);w.writerow(columns)
                for r in measured:
                    if r['backend'] not in manifest['statistics']:continue
                    w.writerow([r['backend'],r['repeat'],r['wall_seconds'],r['summary']['search_seconds'],r['totals'].get('kernel_seconds'),r['totals'].get('fused_device_seconds'),r['totals'].get('theory_seconds'),r['totals'].get('gpu_service_seconds'),r['totals'].get('candidates'),r['generated_peptides'],r['device_peak_bytes'],r['host_observed_hwm_bytes'],r['psm_sha256']])
        except BaseException as e:
            if active is not None and active.poll() is None:
                active.terminate()
                try:active.wait(timeout=10)
                except subprocess.TimeoutExpired:active.kill();active.wait()
            manifest['status']='invalidated' if 'changed during' in str(e) else 'failed';manifest['error']=repr(e);save(mp,manifest);raise
        finally:
            manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(mp,manifest)
        print(str(dest),flush=True)
        if failed_backends:raise SystemExit(1)
if __name__=='__main__':main()
