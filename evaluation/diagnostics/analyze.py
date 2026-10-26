"""Evidence-based diagnostic report; no benchmark execution or algorithm changes."""
import csv,io,json,math,re,statistics,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'output/evaluation';OUT=E/'diagnostics';R=OUT/'reports'
def local(path):return Path(path.replace('/workspace/sipros',str(ROOT)))
def read(path):return json.loads(path.read_text())
def csvout(name,rows):
 with (R/name).open('w') as f:
  if rows:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def ncu(path,meta):
 lines=path.read_text().splitlines();first=next(i for i,l in enumerate(lines) if l.startswith('"ID",'));rows=list(csv.DictReader(io.StringIO('\n'.join(lines[first:]))));units=rows[0];r=next(x for x in rows if x.get('ID','').isdigit());row=dict(meta)
 fields={'kernel_time':'gpu__time_duration.sum','registers_requested':'launch__registers_per_thread','registers_allocated':'launch__registers_per_thread_allocated','entry_stack_bytes':'launch__stack_size','achieved_occupancy_pct':'sm__warps_active.avg.pct_of_peak_sustained_active','sm_warp_instructions':'smsp__inst_executed.sum','dram_reads':'dram__bytes_op_read.sum','dram_writes':'dram__bytes_op_write.sum','l2_sectors':'lts__t_sectors.sum','fp64_pipe_elapsed_pct':'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed','local_load_requests':'l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum','local_store_requests':'l1tex__t_requests_pipe_lsu_mem_local_op_st.sum'}
 for label,key in fields.items():row[label]=float(r[key]) if r.get(key) else None;row[label+'_unit']=units.get(key);row[label+'_counter']=key
 row['raw_file']=str(path);return row
cold=[];equiv=[];profiles=[];timeline=[];hot=[];checks=[]
controls=read(R/'controls_manifest.json')
for c in controls['cells']:
 m=read(local(c['run_manifest']));den=m['statistics']['cuda']['fused_device_seconds']['median'];candidate_sets=[]
 for b,s in m['statistics'].items():
  runs=[r for r in m['runs'] if r['backend']==b and not r['warmup']];r=runs[0];v=s['fused_device_seconds'];candidate_sets.append(tuple(x['candidates'] for x in r['batches']))
  cold.append({'precursor_mode':c['precursor_mode'],'cache_policy':c['cache_policy'],'backend':b,'repetitions':v['n'],'event_median_ms':v['median']*1000,'event_min_ms':v['min']*1000,'event_max_ms':v['max']*1000,'event_stdev_ms':v['stdev']*1000,'cuda_over_backend_event_ratio':den/v['median'],'candidates':int(r['totals']['candidates']),'logical_postmerge_queries':int(r['totals']['inrange']),'generated_peptides':r['generated_peptides'],'cache_layout':json.dumps([(x['cached_ions'],x['cache_charge_stride']) for x in r['batches']]),'repeat_psm_identical':s['repeat_psm_identical'],'source_manifest':c['run_manifest']})
  if b!='cuda':equiv.append({'precursor_mode':c['precursor_mode'],'cache_policy':c['cache_policy'],'backend':b,**m['comparisons'][b]})
 checks.append({'precursor_mode':c['precursor_mode'],'cache_policy':c['cache_policy'],'candidate_count_per_batch_equal':len(set(candidate_sets))==1})
# Cache control must not change results within an otherwise identical method.
for mode in ('primary','multi'):
 a=read(E/'results/accuracy'/('diagnostic_256_'+mode+'_auto')/'manifest.json');b=read(E/'results/accuracy'/('diagnostic_256_'+mode+'_disabled')/'manifest.json')
 for backend in ('cuda','rt-instanced','rt-custom'):
  pick=lambda m:next(r for r in m['runs'] if r['backend']==backend and not r['warmup'])
  checks.append({'precursor_mode':mode,'backend':backend,'cache_on_off_psms_identical':pick(a)['psm_sha256']==pick(b)['psm_sha256']})
for p in read(R/'control_profiles.json')['profiles']:
 if p['status']=='completed':profiles.append(ncu(local(p['metrics_file']),{k:p[k] for k in ('precursor_mode','cache_policy','backend')}))
for p in read(R/'control_timelines.json')['runs']:
 if p['status']!='completed':continue
 folder=local(p['directory'])
 with (folder/'timings_cuda_gpu_kern_sum.csv').open() as f:kernels=[r for r in csv.DictReader(f) if 'ScoreSequenceVsSpectrum' in r['Name'] or r['Name']=='optixLaunch']
 kernel_ms=sum(int(r['Total Time (ns)']) for r in kernels)/1e6;log=(folder/'step_0.log').read_text();event_ms=sum(float(v) for v in re.findall(r' fused_device_seconds=([0-9.eE+-]+)',log))*1000
 timeline.append({k:p[k] for k in ('precursor_mode','cache_policy','backend')}|{'kernel_total_ms':kernel_ms,'event_interval_total_ms':event_ms,'interval_minus_kernel_ms':event_ms-kernel_ms,'scoring_launches':sum(int(r['Instances']) for r in kernels),'source':str(folder)})
full=[];m=read(E/'results/accuracy/all_peak_ecoli_x1/manifest.json')
for r in m['runs']:
 f=E/'results/accuracy/all_peak_ecoli_x1'/r['name']/'peak_audit/peak_audit_batches.tsv'
 with f.open() as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
 sums={k:sum(int(x[k]) for x in rows) for k in rows[0] if k!='batch'};full.append({'backend':r['backend'],**sums,'identity_mismatch_rate':sums['peak_identity_mismatches']/sums['queries']})
hot_meta=read(R/'hot_controls.json') if (R/'hot_controls.json').is_file() else None
if hot_meta and hot_meta['status']=='completed':
 for cell in hot_meta['cells']:
  m=read(local(cell['run_manifest']));ref=read(E/'results/accuracy'/('diagnostic_256_'+cell['precursor_mode']+'_'+cell['cache_policy'])/'manifest.json')
  for r in m['runs']:
   if r['status']!='completed':continue
   text=(local(cell['run_manifest']).parent/(r['name']+'.log')).read_text();totals={i:0.0 for i in range(7)};counts={i:0 for i in range(7)}
   for line in text.splitlines():
    if not line.startswith('[FUSED diagnostic replay]'):continue
    values=dict(re.findall(r'(\w+)=([^\s]+)',line))
    if values['warmup']=='0':i=int(values['trial']);totals[i]+=float(values['fused_device_seconds']);counts[i]+=1
   assert all(v==len(r['batches']) for v in counts.values()),counts
   values=[totals[i]*1000 for i in range(7)];match=next(x for x in ref['runs'] if x['backend']==r['backend'] and not x['warmup'])['psm_sha256']==r['psm_sha256']
   hot.append({'precursor_mode':cell['precursor_mode'],'cache_policy':cell['cache_policy'],'backend':r['backend'],'warmups_per_batch':3,'repetitions':7,'median_ms':statistics.median(values),'min_ms':min(values),'max_ms':max(values),'stdev_ms':statistics.stdev(values),'normal_psm_identical':match,'source_manifest':cell['run_manifest']})
 for row in hot:
  ref=next(x for x in hot if x['precursor_mode']==row['precursor_mode'] and x['cache_policy']==row['cache_policy'] and x['backend']=='cuda');row['cuda_over_backend_ratio']=ref['median_ms']/row['median_ms']
for name,data in [('controls_process_intervals.csv',cold),('controls_psm_consistency.csv',equiv),('kernel_profiles.csv',profiles),('event_vs_gpu_timeline.csv',timeline),('full_x1_matching.csv',full),('same_context_hot_controls.csv',hot)]:csvout(name,data)
old=read(R/'historical_summary.json');hw=read(E/'manifests/inventory.json')['environment'];version={'current_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'current_gpu':hw['gpu']['output'],'nvcc':hw['nvcc']['output'],'compiler':hw['compiler']['output'],'nsys':hw['nsys']['output'],'ncu':hw['ncu']['output'],'historical_exact_1_26_manifest':'not located; deleted untracked output cannot be reconstructed from chat alone','historical_resolved_profile':'8569c24 + conflict fixes; Marine FASTA/Pan X1, batch8M, 18,219,397 first-batch associations, 522,874,365 physical queries; first-batch NCU ratio 1.128x','historical_precursor_rule_change':'8569c24, previous 82522a4/b67962a use extra hypotheses; b67962a independently changes class-priority to nearest-mass matching','historical_gpu':'RTX 5070 Ti Laptop GPU, CC12.0, 46 SMs verified in preserved raw counters','historical_driver_nvcc_versions':'not established for the exact 1.26 run'}
(R/'version_comparison.json').write_text(json.dumps(version,indent=2));(R/'verification_checks.json').write_text(json.dumps(checks,indent=2))
lines=['# CUDA / RT 融合评分诊断报告','','## 状态与结论','',
'大型性能实验保持暂停；未恢复 batch、DB-1、Soil DB-2 或 scan 扩展。已完成测量、配置和日志均保留。前一条停止请求后发出的 SIGTERM 已使正在运行的 Marine 子进程退出；当前不完整轮次标为中断，不当作算法失败或完整性能结果。',
'', '**已证实：前体规则改变了工作负载，理论离子缓存改变了融合 kernel 内实际执行的计算；当前 CUDA-event 区间还包含明显的非 kernel 时间。不能直接把历史约 1.26× 与当前比值作同条件比较。**',
'', '## 1. 前体假设：用户线索的代码证据','',
f"同一份 Pan X1 的 {old['scan_count']:,} 张谱：历史多前体规则构造 {old['old_precursor_count']:,} 个条目；当前首个仪器电荷规则构造 {old['new_precursor_count']:,} 个，前体条目相差 {old['old_precursor_count']/old['new_precursor_count']:.3f} 倍。移除了 {old['removed_count']:,} 个额外假设，{old['changed_scan_count']:,} 张谱的候选前体列表发生变化。",
'`8569c24` 的源码 diff 明确删除额外 charge/mz 对的展开。旧规则不只是对同一 m/z 尝试不同电荷，也包含隔离窗口内其他 m/z 假设；并有与主前体过近时排除的逻辑。因此不能用“把 primary charge 复制几份”代替原规则。',
'这一点证实两代负载不等价，但**不能单凭条目数证明 1.26× 的唯一原因**。候选关联、每候选离子数、同 charge 缓存复用、MVH 资格和 GPU 调度也会变化。',
'', '## 2. 同输入后端比较：不满足严格匹配等价','',
'完整 Ecoli FASTA＋X1 的独立融合路径诊断：三后端均处理 8,115,758 个候选关联、237,923,118 次实际碎片查询。输入 SHA、配置、分批候选数量、理论生成代码与缓存决策已核对；不依赖相同 PSM 行数推断匹配正确性。尚未保存全部查询序列的逐项/逐 bit 证明，计数一致本身不是完整输入序列证明。',
'', '| 后端 | 查询数 | 峰身份不一致 | class 不一致 |','|---|---:|---:|---:|']
for r in full:lines.append(f"| {r['backend']} | {r['queries']:,} | {r['peak_identity_mismatches']:,} | {r['class_mismatches']:,} |")
lines += ['', 'sphere 的两次差异已映射到原始 X1 scan 26325、38114，实际峰均在严格 0.01 Da 容差之外约 **3.312×10⁻⁸ Da**。最终保留 PSM 相同，不代表匹配等价。Instanced 的全量身份差异是 78,805；保存的 256 个例子均是漏掉有效峰，但例子按每批上限截取，不能据此推断全部差异分布。',
'', '## 3. 有界 2×2 控制：只用 256 张 X1 谱','',
'使用完整小型 Ecoli FASTA、相同 256 张谱和 batch 上限 2M。当前算法仅在隔离副本的三个 FT2 reader 方法中选择 primary/历史 multi；缓存开关仅控制原有 `useIonCache`。匹配、几何、理论离子公式和 MVH 代码 SHA 保持不变，见 control_build.json。每格一次进程预热＋五次新进程测量，后端顺序轮换。下表是 **进程内程序记录的融合事件区间**，不是纯 kernel 时间。',
'', '| 前体模式 | 缓存 | 候选关联 | CUDA ms | instanced ms | sphere ms | CUDA/sphere |','|---|---|---:|---:|---:|---:|---:|']
for mode,cache in [('primary','auto'),('multi','auto'),('primary','disabled'),('multi','disabled')]:
 rows=[x for x in cold if x['precursor_mode']==mode and x['cache_policy']==cache];d={x['backend']:x for x in rows};lines.append(f"| {mode} | {cache} | {d['cuda']['candidates']:,} | {d['cuda']['event_median_ms']:.3f} | {d['rt-instanced']['event_median_ms']:.3f} | {d['rt-custom']['event_median_ms']:.3f} | {d['rt-custom']['cuda_over_backend_event_ratio']:.3f}× |")
lines += ['', '多假设＋关闭缓存出现约 **1.28×**，与用户回忆的数值接近；但这只是小型 Ecoli 控制、当前 nearest-mass 几何，并非历史全 Marine 的精确复现。自动缓存时，仅恢复 multi 并未恢复 RT 优势，说明这里存在**前体展开与缓存的交互**。每格候选数在三后端间一致，缓存 on/off 的同后端最终 PSM 哈希一致；instanced 的跨后端 PSM 仍不一致。',
'', '## 4. 计时边界与预热核验','',
'当前自动缓存条件包含 `keys = actual_batch_peptides × min(maxCharge+1,9)` 且 `keys < 32,000,000`，还有显存预算检查。2M/当前 charge 可启用；历史 Marine 的 8M/10M 满批通常超过该键数阈值而关闭。缓存开启时，离子生成在 theory_seconds 中，kernel 主要读缓存；关闭时，生成在融合 kernel 中。**相同函数名不保证相同计算范围。**',
'原调度器的一次预热是独立子进程，后续每次测量又新建进程，不能保证同 CUDA 上下文中的评分 kernel 热启动。NSYS 对同一次运行验证了事件区间与 GPU 时间戳的差别：',
'', '| 条件 | 后端 | 同次事件区间 ms | GPU kernel 总计 ms |','|---|---|---:|---:|']
for r in timeline:lines.append(f"| {r['precursor_mode']}/{r['cache_policy']} | {r['backend']} | {r['event_interval_total_ms']:.3f} | {r['kernel_total_ms']:.3f} |")
lines += ['', '这些差额表明区间包含提交/调度/可能的加载等待；尚未把差额唯一归因于 JIT 或某个驱动函数。NCU replay 耗时也不能替代无 profiler 的 baseline。', '', '## 5. 同上下文热 kernel 控制','']
if hot:
 lines += ['增加独立 launch-only 驱动：每批相同只读评分输入在同一 CUDA 上下文预热 3 次、测量 7 次，覆盖 Result 后只执行一次原有 Top retention。最终 PSM 与非 replay 控制逐后端比对，防止重放改变科学结果。这是固定输入的热缓存诊断，不是完整应用吞吐量。','','| 前体/缓存 | CUDA ms | instanced ms | sphere ms | CUDA/sphere | 输出不变 |','|---|---:|---:|---:|---:|---|']
 for mode,cache in [('primary','auto'),('multi','auto'),('primary','disabled'),('multi','disabled')]:
  d={x['backend']:x for x in hot if x['precursor_mode']==mode and x['cache_policy']==cache};lines.append(f"| {mode}/{cache} | {d['cuda']['median_ms']:.3f} | {d['rt-instanced']['median_ms']:.3f} | {d['rt-custom']['median_ms']:.3f} | {d['rt-custom']['cuda_over_backend_ratio']:.3f}× | {all(x['normal_psm_identical'] for x in d.values())} |")
else:lines.append('同上下文热重复诊断仍在进行；未填入尚不存在的数据。')
lines += ['', '## 6. 硬件、软件、版本与瓶颈证据','',
'已保存的 2026-10-08 raw counters 和当前环境均指向 **RTX 5070 Ti Laptop GPU、CC12.0、46 SM**。当前是 CUDA 12.8.93、GCC 11.4、NSYS 2026.3.1、NCU 2026.2.1；driver 库/内核报告存在 610.57.01/610.88 不同层级字段，应保留原值，不混作版本升级证据。原 1.26× 对应的确切 driver、NVCC、binary SHA、manifest 尚未定位，不能声称软件栈完全相同。',
'历史源码至少有两类独立变更：`b67962a` 改 class-priority 为 nearest-mass 并改球体坐标；`8569c24` 改多前体为 primary-only。若 1.26× 来自更早的 `82522a4`，匹配语义也不同；若来自两提交之间，仍需确切版本证明。',
'从 Git 恢复的另一份可验证旧记录是 8569c24＋冲突修复、Marine＋X1、8M batch、首批 18,219,397 个关联。其 NCU 首批 CUDA 198.527 ms、sphere 175.973 ms，约 1.128×；它是**另一份记录，不冒充 1.26× 那次**。原未跟踪输出已按用户此前清理要求删除，部分小型指标可由 Git 恢复，聊天摘要不能替代原始日志。',
'本次已采集 12 个 NCU 小型首批 profile 和 12 个 NSYS 小型完整运行。逐项 counter 名称、原单位、命令和源文件见 kernel_profiles.csv / control_profiles.json。缓存路径下，OptiX 仍有较多 SM 指令与较高寄存器需求；直接生成路径下，CUDA 的 FP64 和指令负担显著增加。证据支持相对成本随 kernel 工作组成变化，不支持“RT Core 突然失效”。不能仅凭 occupancy 或指令数认定瓶颈，更不能声称已观测 RT Core 利用率。',
'', '## 7. 尚未解决与下一步建议','',
'1. 找到 1.26× 那次确切 commit、输入 SHA、生成肽段 batch 单位、缓存状态、前体规则和计时边界；缺失部分继续标记未证实。',
'2. 如需更强的逐请求一致性证明，保存 canonical candidate/charge/query 序列及指纹或逐项对照。当前计数与共享源代码是证据，不能替代完整逐 bit 审计。',
'3. 将 cold process、同上下文热 kernel、GPU timestamp、整个评分服务和完整 MVH 搜索分列；不可混用加速比。',
'4. 下一轮仅在小规模 Marine FASTA 子集＋相同 X1 谱上交叉验证前体/缓存因素；不要直接重启全 Marine、Soil 或规模扫描。',
'5. 在另行授权前不修改核心匹配算法。先讨论严格容差、等距和 float 返回结果的正确性要求，再决定设计修正。',
'6. 大型 benchmarking 保持暂停。当前尚未满足“全部后端匹配等价”的条件；诊断中再现某个加速数值不等于比较方法已全面验证。',
'', '## 证据文件','', '- controls_process_intervals.csv：五次新进程的区间计时、min/max/stdev、候选与缓存布局。','- same_context_hot_controls.csv：同上下文 3 warmups＋7 repetitions，及输出一致性。','- event_vs_gpu_timeline.csv：同次事件区间与真实 kernel 时间。','- kernel_profiles.csv：硬件指标和原 counter/单位。','- full_x1_matching.csv：全量峰级一致性统计。','- historical_summary.json、historical_20261008_metrics.json、historical_chat_excerpts.json：历史证据及其证据级别。','- version_comparison.json、control_build.json、verification_checks.json：版本/构建和验证检查。']
evidence=['### 多假设控制的 NCU 证据（首批，非 baseline 计时）','', '| 缓存/后端 | SM warp 指令 | Local load 请求 | FP64 管线占用 % | DRAM 读 |','|---|---:|---:|---:|---:|']
for row in profiles:
 if row['precursor_mode']=='multi' and row['backend'] in ('cuda','rt-custom'):
  evidence.append(f"| {row['cache_policy']}/{row['backend']} | {row['sm_warp_instructions']:,.0f} | {row['local_load_requests']:,.0f} | {row['fp64_pipe_elapsed_pct']:.2f} | {row['dram_reads']:.3f} {row['dram_reads_unit']} |")
evidence += ['', '关闭缓存后，CUDA 的 warp 指令从约 1,098 万增到 1.473 亿，sphere 从约 4,547 万增到 9,019 万；CUDA 的 local load 和 FP64 压力增幅更大。这支持直接生成路径的相对执行代价不同，不能把全部差异单独归因于 FP64、寄存器溢出或 RT traversal。配置存储、编译器 lowering 与局部数组访问还需 SASS 控制验证。', '时钟未锁定，系统仍有桌面 GPU 负载；同型号 GPU 不保证功耗、温度和动态频率相同。前 256 张谱的结果不能推广为全 Marine/Soil 的加速结论。', '']
position=lines.index('## 7. 尚未解决与下一步建议');lines[position:position]=evidence
(R/'diagnostic_report.md').write_text('\n'.join(lines)+'\n')
print('Diagnostic report generated; hot controls:',len(hot),'profile rows:',len(profiles),'timeline rows:',len(timeline))
