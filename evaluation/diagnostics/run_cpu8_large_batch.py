"""User requested CPU 8 threads / 12M assigned batch, shrink only after a failed run."""
import os,sys,json,subprocess,shutil,hashlib,re,csv,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');D=E/'diagnostics'/('cpu8_large_batch_'+tag);D.mkdir(exist_ok=False);S=D/'source';B=D/'build';shutil.copytree(E/'diagnostics/cpu_primary_baseline/source',S,symlinks=True)
env=os.environ.copy();env['OMP_DYNAMIC']='FALSE'
for k in ('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR','SIPROS_DIAGNOSTIC_HOT_REPLAY','SIPROS_CONTINUOUS_REPLAY'):env.pop(k,None)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();header=S/'mvh/include/mvh_scan_vector.h';original=header.read_text();assert '#define PEPTIDE_ARRAY_SIZE 2000000' in original
protected=['mvh/original/src/MVH.cpp','mvh/original/src/ms2scan.cpp','mvh/original/src/peptide.cpp','mvh/src/spectrum_input.cpp','mvh/src/database_search.cpp']
assert all(sha(S/f)==sha(E/'diagnostics/cpu_primary_baseline/source'/f) for f in protected)
m={'status':'running','output':str(D),'threads':8,'precursor_mode':'primary-only','requested_cpu_batch':12000000,'cpu_batch_unit':'assigned peptide entries','fallback_policy':[12000000,8000000,4000000,2000000],'parent_cpu_reference':'cpu_primary_baseline','gpu_reference':'full_marine_x2_primary_20261026T192358Z','protected_source_sha256':{f:sha(S/f) for f in protected},'attempts':[]};mp=D/'manifest.json';mp.write_text(json.dumps(m,indent=2));print(json.dumps({'output':str(D)},indent=2),flush=True)
for batch in m['fallback_policy']:
 header.write_text(original.replace('#define PEPTIDE_ARRAY_SIZE 2000000',f'#define PEPTIDE_ARRAY_SIZE {batch}'))
 A=D/str(batch);A.mkdir();shutil.copyfile(header,A/'mvh_scan_vector.h');item={'batch':batch,'status':'building','attempt_dir':str(A)};m['attempts'].append(item);mp.write_text(json.dumps(m,indent=2))
 commands=[]
 if not (B/'build.ninja').exists():commands.append(['cmake','-S',str(S/'mvh'),'-B',str(B),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release'])
 commands.append(['cmake','--build',str(B),'--target','sipros_mvh','-j','4']);item['build_commands']=commands
 for i,c in enumerate(commands):
  with (A/f'build_{i}.log').open('w') as log:p=subprocess.run(c,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:item.update(status='build_failed',exit_code=p.returncode);m['status']='failed';mp.write_text(json.dumps(m,indent=2));raise RuntimeError('Build failed independently of runtime batch')
 (A/'bin').mkdir();shutil.copyfile(B/'CMakeCache.txt',A/'CMakeCache.txt');binary=A/'bin/sipros_mvh';shutil.copy2(B/'bin/sipros_mvh',binary);item['binary_sha256']=sha(binary)
 rid=f'cpu8_{tag}_batch{batch}';cmd=[sys.executable,'-B',str(R/'evaluation/scripts/run.py'),'--id',rid,'--category','accuracy','--backends','cpu','--threads','8','--batch',str(batch),'--repeats','1','--warmups','0','--scans',str(E/'inputs/ecoli_x2/Pan_062822_X2iso5.FT2'),'--fasta',str(R/'raw/Marine_fw_3rev.fasta'),'--cpu-binary',str(binary)]
 item.update(id=rid,status='running',command=cmd);mp.write_text(json.dumps(m,indent=2));print(f'RUN CPU threads=8 assigned_batch={batch} id={rid}',flush=True)
 with (A/'driver.log').open('w') as log:p=subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
 item['exit_code']=p.returncode;rp=E/'results/accuracy'/rid/'manifest.json';n=json.loads(rp.read_text()) if rp.exists() else {};runs=n.get('runs',[])
 if not runs:
  item.update(status='preflight_failed',reason='No CPU child was launched; not a runtime batch failure');m['status']='preflight_failed';mp.write_text(json.dumps(m,indent=2));raise RuntimeError('Runner preflight failed; repair driver rather than shrinking batch')
 if p.returncode or runs[0]['status']!='completed':item['status']='failed';mp.write_text(json.dumps(m,indent=2));print(f'FAILED batch={batch}; preserving artifacts and trying next smaller batch',flush=True);continue
 r=runs[0];assert int(r['summary']['omp_max_threads'])==8 and int(r['summary']['peptide_batch_size'])==batch
 text=(rp.parent/'cpu_r1.log').read_text();batches=[dict(re.findall(r'(\w+)=([^\s]+)',l)) for l in text.splitlines() if l.startswith('[CPU batch]')];generated=int(re.search(r'\[CPU generated\] peptides=(\d+)',text)[1]);previous=json.loads((E/'diagnostics/cpu_primary_baseline/manifest.json').read_text())
 item.update(status='completed',search_seconds=float(r['summary']['search_seconds']),generated_peptides=generated,associations=sum(int(x['associations']) for x in batches),assigned_peptides=sum(int(x['assigned_peptides']) for x in batches),scoring_batches=len(batches),batch_service_seconds=sum(float(x['batch_service_seconds']) for x in batches),scoring_loop_seconds=sum(float(x['scoring_seconds']) for x in batches),preprocess_seconds=sum(float(x['preprocess_seconds']) for x in batches),cleanup_seconds=sum(float(x['cleanup_seconds']) for x in batches),psm_sha256=r['psm_sha256'],psm_matches_cpu4=r['psm_sha256']==previous['cpu_psm_sha256'],peak_host_rss_bytes=r.get('host_observed_hwm_bytes'),cpu4_over_this_search_ratio=previous['cpu_search_seconds']/float(r['summary']['search_seconds']))
 with (A/'batches.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(batches[0]));w.writeheader();w.writerows(batches)
 if not item['psm_matches_cpu4']:
  sys.path.insert(0,str(R/'evaluation/scripts'));from run import compare
  item['comparison_to_cpu4']=compare(E/'results/accuracy/full_marine_x2_primary_cpu/cpu_r1/mvh_psms.tsv',rp.parent/'cpu_r1/mvh_psms.tsv',A/'cpu4_differences.tsv')
 m.update(status='completed',successful_batch=batch,result=item);mp.write_text(json.dumps(m,indent=2));break
else:m['status']='failed';mp.write_text(json.dumps(m,indent=2))
lines=['# CPU 8 线程／大 batch 验证','','Marine 全库＋X2 全部谱图，primary-only，8 OpenMP 线程。先尝试 12M 个已分配肽段 batch，仅在运行失败时按 8M→4M→2M 缩小。GPU 的 12M 为生成肽段数，单位不同。匹配、MVH、前体 reader 与已验证 CPU 阶段观察器保持原样，仅改 CPU batch 常量。',f"状态：{m['status']}。",'','| 尝试 batch | 状态 |','|---|---|']
for x in m['attempts']:lines.append(f"| {x['batch']:,} | {x['status']} |")
if m['status']=='completed':
 x=m['result'];previous=json.loads((E/'diagnostics/cpu_primary_baseline/manifest.json').read_text());gpu=json.loads((E/'diagnostics'/m['gpu_reference']/'primary_analysis.json').read_text());lines+=['','| 配置 | 完整 MVH search s |','|---|---:|',f"| CPU 4 线程 / 2M assigned batch | {previous['cpu_search_seconds']:.3f} |",f"| CPU 8 线程 / {x['batch']/1000000:g}M assigned batch | {x['search_seconds']:.3f} |"]
 for g in gpu['summary']:lines.append(f"| {g['backend']} / 12M generated batch | {g['search_seconds']:.3f} |")
 lines+=['',f"4线程旧配置／8线程新配置：{x['cpu4_over_this_search_ratio']:.3f}×。同时改变线程数和 batch，不能归因为纯线程扩展收益。",f"CPU 8线程生成 {x['generated_peptides']:,} 条肽段，{x['associations']:,} 个候选关联，{x['scoring_batches']} 个评分批次。与 CPU4 最终 PSM SHA256 相同：{x['psm_matches_cpu4']}。",f"CPU8 原生批处理累计 {x['batch_service_seconds']:.3f} s，其中评分循环 {x['scoring_loop_seconds']:.3f} s，肽段预处理 {x['preprocess_seconds']:.3f} s，对象释放 {x['cleanup_seconds']:.3f} s。其边界与 GPU kernel 或完整评分不完全一致，主要比较完整 search。",f"观测主机 VmHWM：{(x['peak_host_rss_bytes'] or 0)/2**30:.3f} GiB（采样，可能漏掉瞬时峰值）。"]
lines+=['','单次、无预热、未固定 CPU affinity/频率；不声称线程扩展最优或统计显著。失败日志与每次构建的二进制/头文件都保留。其他大型实验仍暂停。',''];(D/'report.md').write_text('\n'.join(lines));print(json.dumps(m,indent=2))
