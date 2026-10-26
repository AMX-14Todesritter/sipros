"""User-authorized one-off full Marine/X2 validation; other queues stay paused."""
import os,sys,json,subprocess,hashlib,datetime,statistics,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');D=E/'diagnostics'/('full_marine_x2_primary_'+tag);D.mkdir(exist_ok=False)
scans=E/'inputs/ecoli_x2/Pan_062822_X2iso5.FT2';fasta=R/'raw/Marine_fw_3rev.fasta';binary=E/'diagnostics/build/bin/sipros_mvh_cuda';rid='full_marine_x2_primary_'+tag
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cmd=[sys.executable,'-B',str(R/'evaluation/scripts/run.py'),'--id',rid,'--category','accuracy','--scans',str(scans),'--fasta',str(fasta),'--backends','rt-custom','cuda','--batch','12000000','--repeats','1','--warmups','0','--gpu-binary',str(binary),'--continue-on-backend-failure']
env=os.environ.copy();env.update(SIPROS_DIAGNOSTIC_NO_ION_CACHE='1');env.pop('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR',None);env.pop('SIPROS_DIAGNOSTIC_HOT_REPLAY',None);env.pop('SIPROS_CONTINUOUS_REPLAY',None)
m={'status':'running','id':rid,'scope':'single explicitly authorized full Marine + Pan X2 validation; not a resumed suite','request':'再进行一次试验，不使用多前体，进行一次profiling','batch':12000000,'batch_unit':'generated peptide entries','precursor_mode':'primary-only','cache':'disabled','backends':['rt-custom','cuda'],'repeats':1,'warmups':0,'fasta_sha256':sha(fasta),'scans_sha256':sha(scans),'binary_sha256':sha(binary),'command':cmd,'output':str(D),'results_manifest':str(E/'results/accuracy'/rid/'manifest.json'),'large_queue_resumed':False}
mp=D/'manifest.json';mp.write_text(json.dumps(m,indent=2));print(json.dumps({'output':str(D),'id':rid,'command':cmd},indent=2),flush=True)
with (D/'driver.log').open('w') as log:p=subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
m['exit_code']=p.returncode
result_path=E/'results/accuracy'/rid/'manifest.json';n=json.loads(result_path.read_text()) if result_path.exists() else {};rows=[]
for r in n.get('runs',[]):
 t=r.get('totals',{});row={'backend':r['backend'],'status':r['status'],'exit_code':r.get('exit_code'),'completed_batches':len(r.get('batches',[])),'generated_peptides':r.get('generated_peptides'),'candidates':int(t['candidates']) if 'candidates' in t else None,'fused_ms':t.get('fused_device_seconds',0)*1000,'gpu_service_ms':t.get('gpu_service_seconds',0)*1000,'rt_setup_ms':r.get('rt_setup_seconds',0)*1000,'search_ms':float(r['summary']['search_seconds'])*1000 if 'summary' in r else None,'psm_sha256':r.get('psm_sha256')}
 if r['status']=='completed':assert all(int(b['cached_ions'])==0 and int(b['cache_charge_stride'])==0 for b in r['batches'])
 rows.append(row)
m.update(status='completed' if p.returncode==0 and len(rows)==2 and all(r['status']=='completed' for r in rows) else 'failed',summary=rows,comparisons=n.get('comparisons',{}));mp.write_text(json.dumps(m,indent=2))
with (D/'summary.csv').open('w') as f:
 if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# 全 Marine + Pan X2：单次 primary-only、无缓存验证','','用户指定 batch=12,000,000（生成肽段条目），CUDA 与 RT-custom 各运行一次，不做重放或预热。核心匹配算法未修改，其他大规模实验队列继续暂停。',f"状态：{m['status']}；退出码：{p.returncode}。",'','| 后端 | 状态 | 批次数 | 候选关联 | 融合评分 ms | 完整 GPU 服务 ms | RT setup ms | MVH search ms |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['backend']} | {r['status']} | {r['completed_batches']} | {r['candidates']} | {r['fused_ms']:.3f} | {r['gpu_service_ms']:.3f} | {r['rt_setup_ms']:.3f} | {r['search_ms']} |")
if m['status']=='completed':
 a=next(r for r in rows if r['backend']=='cuda');b=next(r for r in rows if r['backend']=='rt-custom');ratios={k:a[k]/b[k] for k in ('fused_ms','gpu_service_ms','search_ms')};m['cuda_over_rt_custom']=ratios;m['psm_identical']=a['psm_sha256']==b['psm_sha256'];mp.write_text(json.dumps(m,indent=2));lines+=['',f"CUDA/RT-custom：融合评分 {ratios['fused_ms']:.3f}×；GPU 服务 {ratios['gpu_service_ms']:.3f}×；MVH search {ratios['search_ms']:.3f}×。",f"最终 PSM SHA256 相同：{m['psm_identical']}。",'比较详情：`'+json.dumps(m['comparisons'],ensure_ascii=False)+'`']
lines+=['','无预热、单次普通进程，融合评分包含首次启动影响，不等同于此前热重放中位数。服务包含 RT setup，不应再次叠加；最终 PSM 一致不能替代完整峰级审计。本轮没有逐查询观察器；不得将已有小子集审计外推为全量严格等价。','失败时保留部分日志与计数，不能把已完成批次累计当作全量性能结果；不会擅自降低指定 batch 或重试。所有配置、原始日志和结果保留在 output。','']
(D/'report.md').write_text('\n'.join(lines));print(json.dumps(m,indent=2),flush=True)
