"""Quick CUDA vs RT-custom verification, ion cache disabled in every run."""
import os,sys,subprocess,json,re,statistics,csv,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');D=E/'diagnostics'/('no_cache_verify_'+tag);D.mkdir(exist_ok=False)
env=os.environ.copy();env.update(SIPROS_DIAGNOSTIC_MULTI_PRECURSOR='1',SIPROS_DIAGNOSTIC_NO_ION_CACHE='1');env.pop('SIPROS_CONTINUOUS_REPLAY',None)
scans=E/'inputs/ecoli_x1_first256/spectra.FT2';fasta=E/'diagnostics/marine_control/marine_2048families.fasta'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m={'status':'running','output':str(D),'scans_sha256':sha(scans),'fasta_sha256':sha(fasta),'cache':'disabled','precursor_mode':'multi','runs':[]};mp=D/'manifest.json';rows=[]
for round in range(3):
 for mode in ('hot','service'):
  if mode=='hot':env['SIPROS_DIAGNOSTIC_HOT_REPLAY']='1';binary=E/'diagnostics/hot_build/bin/sipros_mvh_cuda'
  else:env.pop('SIPROS_DIAGNOSTIC_HOT_REPLAY',None);binary=E/'diagnostics/build/bin/sipros_mvh_cuda'
  backends=['cuda','rt-custom'] if round%2==0 else ['rt-custom','cuda']
  rid=f'no_cache_{tag}_{mode}_{round}'
  cmd=[sys.executable,'-B',str(R/'evaluation/scripts/run.py'),'--id',rid,'--category','accuracy','--scans',str(scans),'--fasta',str(fasta),'--backends',*backends,'--batch','2000000','--repeats','1','--warmups','0','--gpu-binary',str(binary)]
  item={'id':rid,'mode':mode,'round':round,'command':cmd,'binary_sha256':sha(binary),'status':'running'};m['runs'].append(item);mp.write_text(json.dumps(m,indent=2));print(rid,flush=True)
  with (D/(rid+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT)
  item.update(status='completed' if p.returncode==0 else 'failed',exit_code=p.returncode);mp.write_text(json.dumps(m,indent=2))
  if p.returncode:raise RuntimeError(rid)
  manifest=json.loads((E/'results/accuracy'/rid/'manifest.json').read_text())
  assert manifest['comparisons']['rt-custom']['spectrum_groups_differ']==0
  for run in manifest['runs']:
   assert all(int(b['cached_ions'])==0 and int(b['cache_charge_stride'])==0 for b in run['batches'])
   row={'mode':mode,'round':round,'backend':run['backend'],'candidates':int(run['totals']['candidates']),'generated_peptides':run['generated_peptides'],'psm_sha256':run['psm_sha256'],'cache_disabled_verified':True,'service_ms':run['totals']['gpu_service_seconds']*1000,'search_ms':float(run['summary']['search_seconds'])*1000,'rt_setup_ms':run['rt_setup_seconds']*1000}
   if mode=='hot':
    totals={i:0. for i in range(7)};counts={i:0 for i in range(7)}
    for line in (E/'results/accuracy'/rid/(run['name']+'.log')).read_text().splitlines():
     if not line.startswith('[FUSED diagnostic replay]'):continue
     v=dict(re.findall(r'(\w+)=([^\s]+)',line));assert int(v['cached_ions'])==0
     if v['warmup']=='0':i=int(v['trial']);totals[i]+=float(v['fused_device_seconds'])*1000;counts[i]+=1
    assert all(n==len(run['batches']) for n in counts.values())
    row['hot_fused_ms']=statistics.median(totals.values());row['hot_trials_ms']=list(totals.values())
   rows.append(row)
assert len({r['psm_sha256'] for r in rows})==1
summary=[]
for b in ('cuda','rt-custom'):
 hot=[r['hot_fused_ms'] for r in rows if r['mode']=='hot' and r['backend']==b];service=[r['service_ms'] for r in rows if r['mode']=='service' and r['backend']==b];search=[r['search_ms'] for r in rows if r['mode']=='service' and r['backend']==b]
 summary.append({'backend':b,'hot_fused_median_ms':statistics.median(hot),'hot_round_min_ms':min(hot),'hot_round_max_ms':max(hot),'service_median_ms':statistics.median(service),'service_min_ms':min(service),'service_max_ms':max(service),'search_median_ms':statistics.median(search)})
ratios={k:summary[0][k]/summary[1][k] for k in ('hot_fused_median_ms','service_median_ms','search_median_ms')}
m.update(status='completed',all_psm_identical=True,summary=summary,cuda_over_rt_custom=ratios);mp.write_text(json.dumps(m,indent=2));(D/'raw_metrics.json').write_text(json.dumps(rows,indent=2))
with (D/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
lines=['# 关闭理论离子缓存：CUDA 与 RT-custom 快速复核','','固定 X1 前 256 张谱图、Marine 4,096 条序列、multi 前体、2M batch 上限。三个独立进程轮次，轮换后端执行顺序。所有批次 cached_ions=0、cache_charge_stride=0，确认无缓存。','','| 后端 | 热融合评分 ms | 各轮中位数范围 ms | 普通完整 GPU 评分服务 ms | MVH search ms |','|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['backend']} | {r['hot_fused_median_ms']:.3f} | {r['hot_round_min_ms']:.3f}–{r['hot_round_max_ms']:.3f} | {r['service_median_ms']:.3f} | {r['search_median_ms']:.3f} |")
lines += ['',f"CUDA/RT-custom：热融合评分 {ratios['hot_fused_median_ms']:.3f}×；完整 GPU 评分服务 {ratios['service_median_ms']:.3f}×；MVH search {ratios['search_median_ms']:.3f}×。",'','热评分每批同上下文预热 3 次、测量 7 次，先将两批同序号测量相加，再取中位数，最后对三轮中位数再取中位数；保留每轮原始值。普通评分服务为另一组不重放进程，包含上传、RT 准备、评分、保留、下载等，包含首次运行影响；不能把热 kernel 比例当成整个流程比例。','全部 12 次后端运行的最终 PSM SHA256 一致；重放与普通运行一致。未重新插入峰级观察器：上一轮相同输入上的逐候选查询指纹与峰级审计已经完成；本轮验证性能可重复性与输出。','GPU 时钟未锁定，仅为小负载快速验证，不能推广为全 Marine 或更大规模结论。大型实验持续暂停，核心匹配算法未修改。','']
(D/'report.md').write_text('\n'.join(lines));print(json.dumps(m,indent=2))
