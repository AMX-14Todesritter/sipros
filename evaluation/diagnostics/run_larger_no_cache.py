"""Single larger multi-precursor verification with reduced warmup/repetitions."""
import os,sys,json,subprocess,re,statistics,hashlib,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';D=E/'diagnostics/larger_no_cache';mp=D/'manifest.json';m=json.loads(mp.read_text());m['runs']=[];rows=[];comparisons={}
env=os.environ.copy();env.update(SIPROS_DIAGNOSTIC_MULTI_PRECURSOR='1',SIPROS_DIAGNOSTIC_NO_ION_CACHE='1');env.pop('SIPROS_CONTINUOUS_REPLAY',None)
for mode in ('hot','service'):
 if mode=='hot':env['SIPROS_DIAGNOSTIC_HOT_REPLAY']='1';binary=E/'diagnostics/larger_no_cache_build/bin/sipros_mvh_cuda';backends=['cuda','rt-custom']
 else:env.pop('SIPROS_DIAGNOSTIC_HOT_REPLAY',None);binary=E/'diagnostics/build/bin/sipros_mvh_cuda';backends=['rt-custom','cuda']
 rid='larger_no_cache_'+mode
 cmd=[sys.executable,'-B',str(R/'evaluation/scripts/run.py'),'--id',rid,'--category','accuracy','--scans',m['scans'],'--fasta',m['fasta'],'--backends',*backends,'--batch',str(m['batch']),'--repeats','1','--warmups','0','--gpu-binary',str(binary)]
 item={'id':rid,'mode':mode,'status':'running','command':cmd,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()};m['runs'].append(item);m['status']='running';mp.write_text(json.dumps(m,indent=2));print(rid,flush=True)
 with (D/(mode+'.log')).open('w') as log:p=subprocess.run(cmd,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
 item.update(exit_code=p.returncode,status='completed' if p.returncode==0 else 'failed');mp.write_text(json.dumps(m,indent=2))
 if p.returncode:raise RuntimeError(rid)
 n=json.loads((E/'results/accuracy'/rid/'manifest.json').read_text());comparisons[mode]=n['comparisons']['rt-custom']
 for run in n['runs']:
  assert run['status']=='completed';assert all(int(b['cached_ions'])==0 and int(b['cache_charge_stride'])==0 for b in run['batches'])
  t=run['totals'];row={'mode':mode,'backend':run['backend'],'batches':len(run['batches']),'generated_peptides':run['generated_peptides'],'candidates':int(t['candidates']),'reported_inrange':int(t['inrange']),'psm_sha256':run['psm_sha256'],'gpu_service_ms':t['gpu_service_seconds']*1000,'search_ms':float(run['summary']['search_seconds'])*1000,'rt_setup_ms':run['rt_setup_seconds']*1000}
  if mode=='hot':
   totals={i:0.0 for i in range(3)};counts={i:0 for i in range(3)}
   for line in (E/'results/accuracy'/rid/(run['name']+'.log')).read_text().splitlines():
    if not line.startswith('[FUSED diagnostic replay]'):continue
    v=dict(re.findall(r'(\w+)=([^\s]+)',line));assert int(v['cached_ions'])==0
    if v['warmup']=='0':i=int(v['trial']);totals[i]+=float(v['fused_device_seconds'])*1000;counts[i]+=1
   assert all(c==len(run['batches']) for c in counts.values())
   row.update(hot_fused_ms=statistics.median(totals.values()),hot_trials_ms=list(totals.values()),hot_min_ms=min(totals.values()),hot_max_ms=max(totals.values()))
  rows.append(row)
summary=[]
for b in ('cuda','rt-custom'):
 h=next(r for r in rows if r['mode']=='hot' and r['backend']==b);s=next(r for r in rows if r['mode']=='service' and r['backend']==b)
 summary.append({'backend':b,'hot_fused_ms':h['hot_fused_ms'],'hot_min_ms':h['hot_min_ms'],'hot_max_ms':h['hot_max_ms'],'gpu_service_ms':s['gpu_service_ms'],'search_ms':s['search_ms'],'rt_setup_ms':s['rt_setup_ms'],'hot_psm_matches_service':h['psm_sha256']==s['psm_sha256']})
ratios={k:summary[0][k]/summary[1][k] for k in ('hot_fused_ms','gpu_service_ms','search_ms')}
m.update(status='completed',measurements=rows,summary=summary,cuda_over_rt_custom=ratios,comparisons=comparisons,candidate_counts_identical=len({r['candidates'] for r in rows})==1,all_psm_identical=len({r['psm_sha256'] for r in rows})==1,large_queue_resumed=False);mp.write_text(json.dumps(m,indent=2))
with (D/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
lines=['# 较大固定负载：多前体、无缓存 CUDA 与 RT-custom','','X1 前 4,096 张谱图；Marine 前 8,192 个 target/decoy 家族（16,384 条序列）；multi 前体；理论离子缓存关闭；batch 上限 2M。只有两个后端。',f"实际生成 {rows[0]['generated_peptides']:,} 条肽段，{rows[0]['candidates']:,} 个候选关联，共 {rows[0]['batches']} 个评分批次。所有批次 cached_ions=0 / cache_charge_stride=0。",'','| 后端 | 热融合评分 ms | 三次总时间范围 ms | 普通完整 GPU 评分服务 ms | MVH search ms | RT setup ms |','|---|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['backend']} | {r['hot_fused_ms']:.3f} | {r['hot_min_ms']:.3f}–{r['hot_max_ms']:.3f} | {r['gpu_service_ms']:.3f} | {r['search_ms']:.3f} | {r['rt_setup_ms']:.3f} |")
lines+=['',f"CUDA/RT-custom：热融合 kernel {ratios['hot_fused_ms']:.3f}×；GPU 评分服务 {ratios['gpu_service_ms']:.3f}×；MVH search {ratios['search_ms']:.3f}×。",'','每批同上下文预热 1 次、测量 3 次；同一 trial 的各批热评分时间相加后取中位数。另一组普通进程不重放评分，用于完整 GPU 服务和搜索计时；其运行顺序与热评分组相反。服务包括上传、RT 准备、评分、保留、下载、释放等；search 包含肽段生成、host packing 等。RT setup 来自日志的单独测量，不应再加到已包含它的服务时间上。','',f"四次运行候选计数一致：{m['candidate_counts_identical']}；全部 PSM SHA256 一致：{m['all_psm_identical']}。"]
for mode,c in comparisons.items():lines.append(f"- {mode}：CUDA 与 RT-custom 有 {c['spectrum_groups_differ']} 个谱图结果组不同，公共候选评分差异 {c['common_candidate_scores_differ']} 个。")
lines+=['','本轮未插入逐查询/峰级观察器；最终 PSM 一致不证明所有峰级匹配逐项等价。上轮小子集审计不能替代本轮更大子集审计。仅一个进程/条件、三次热测量，GPU 时钟未锁定，旨在快速验证，不作统计显著性或全数据库外推。','核心匹配算法未改动；诊断驱动的离子公式、匹配、几何、MVH 文件哈希保持不变，见 larger_no_cache_build.json。其余大型 batch、DB-1、Soil DB-2 与 scan 扩展队列仍暂停。所有配置、日志、结果及构建产物都在 output。','']
(D/'report.md').write_text('\n'.join(lines));print(json.dumps({'output':str(D),'summary':summary,'ratios':ratios,'comparisons':comparisons,'all_psm_identical':m['all_psm_identical']},indent=2))
