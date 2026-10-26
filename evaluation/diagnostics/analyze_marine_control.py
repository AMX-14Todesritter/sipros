"""Report bounded Marine control without claiming query-level equivalence."""
import json,re,statistics,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'output/evaluation';D=E/'diagnostics/marine_control'
rows=[];checks=[];manifests={}
for cache in ('auto','disabled'):
 m=json.loads((E/'results/accuracy'/('marine_control_multi_'+cache)/'manifest.json').read_text());manifests[cache]=m
 for r in m['runs']:
  assert r['status']=='completed'
  text=(E/'results/accuracy'/('marine_control_multi_'+cache)/(r['name']+'.log')).read_text();trials={i:0. for i in range(7)};counts={i:0 for i in range(7)}
  for line in text.splitlines():
   if not line.startswith('[FUSED diagnostic replay]'):continue
   v=dict(re.findall(r'(\w+)=([^\s]+)',line))
   if v['warmup']=='0':i=int(v['trial']);trials[i]+=float(v['fused_device_seconds'])*1000;counts[i]+=1
  assert all(c==len(r['batches']) for c in counts.values())
  vals=list(trials.values());t=r['totals'];median=statistics.median(vals)
  rows.append({'cache':cache,'backend':r['backend'],'batches':len(r['batches']),'generated_peptides':r['generated_peptides'],'candidates':int(t['candidates']),'logical_queries':int(t['inrange']),'cached_ions':int(t['cached_ions']),'hot_fused_median_ms':median,'hot_fused_min_ms':min(vals),'hot_fused_max_ms':max(vals),'theory_once_ms':t['theory_seconds']*1000,'theory_once_plus_hot_fused_ms':t['theory_seconds']*1000+median,'psm_sha256':r['psm_sha256']})
 for backend,comparison in m['comparisons'].items():checks.append({'cache':cache,'backend':backend,**comparison})
for r in rows:
 ref=next(x for x in rows if x['cache']==r['cache'] and x['backend']=='cuda');r['cuda_over_backend_hot_ratio']=ref['hot_fused_median_ms']/r['hot_fused_median_ms']
within={b:next(r for r in rows if r['cache']=='auto' and r['backend']==b)['psm_sha256']==next(r for r in rows if r['cache']=='disabled' and r['backend']==b)['psm_sha256'] for b in ('cuda','rt-instanced','rt-custom')}
equal_counts=len(set((r['generated_peptides'],r['candidates'],r['logical_queries']) for r in rows))==1
with (D/'timings.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(D/'checks.json').write_text(json.dumps({'aggregate_workload_counts_equal':equal_counts,'cache_toggle_psm_identical_within_backend':within,'comparisons_to_cuda':checks,'query_sequence_fingerprints_collected':False},indent=2))
lines=['# Marine 小规模融合评分控制实验','','固定 Pan X1 前 256 张谱图，Marine 前 2,048 个 target/decoy 家族（4,096 条序列），multi 前体，batch 上限 2,000,000。使用相同诊断二进制，仅切换缓存环境变量。每批同上下文预热 3 次、测量 7 次；表中热评分为各批同序号测量之和的中位数。','','| 缓存 | 后端 | 热评分 ms | 相对 CUDA 加速比 | 单次 theory ms | theory + 热评分 ms |','|---|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['cache']} | {r['backend']} | {r['hot_fused_median_ms']:.4f} | {r['cuda_over_backend_hot_ratio']:.3f}× | {r['theory_once_ms']:.4f} | {r['theory_once_plus_hot_fused_ms']:.4f} |")
lines+=['','## 已验证',f"生成肽段、候选关联和逻辑查询总数在全部六组中一致：{equal_counts}。",f"每组生成 {rows[0]['generated_peptides']:,} 条肽段，{rows[0]['candidates']:,} 个候选关联，{rows[0]['logical_queries']:,} 个逻辑查询。",f'同后端切换缓存后最终 PSM 文件 SHA256 是否相同：{within}。','']
for c in checks:lines.append(f"- {c['cache']}/{c['backend']}：与 CUDA 有 {c['spectrum_groups_differ']} 个谱图结果组不同，公共候选评分不同 {c['common_candidate_scores_differ']} 个。")
lines+=['','## 解释与局限','RT-instanced 的日志 inrange 为 3,110,705，比 CUDA/RT-custom 的 3,110,658 多 47。该计数来自结果恢复/合并，不能直接解释为物理 launch 查询数不同，也不能据此宣称三后端工作量完全等价；需逐候选、逐查询审计确认。','开启缓存时，离子生成计入独立 theory 阶段；关闭缓存时，离子生成进入融合 kernel，因此两个条件的热评分计时范围不同。theory + 热评分列仅是分阶段观察值的和，theory 未做同样的热重复，不能当作稳定端到端加速比。','本实验核对了输入 SHA、批次计数、缓存状态和最终结果，未采集逐候选/逐查询指纹；计数一致不能证明全部请求逐项一致，最终 PSM 一致也不能证明峰级匹配完全等价。','固定输入热重放是诊断实验；未锁 GPU 时钟，每条件仅一个进程，顺序执行，不能推广为全 Marine 性能。核心匹配算法未修改。本次未新增 kernel profiler 捕获，已有 Ecoli NCU 结论仍需 Marine 独立 profile 验证。','大型 batch、数据库扩展和 scan 扩展仍暂停。后续应先完成逐查询审计和 Marine profiling，再讨论算法修正或恢复大规模性能实验。','']
(D/'report.md').write_text('\n'.join(lines))
print(json.dumps({'rows':rows,'checks':checks,'cache_psm':within,'equal_counts':equal_counts},indent=2))
