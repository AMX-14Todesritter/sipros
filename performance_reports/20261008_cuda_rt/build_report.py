from pathlib import Path
import csv,json,re,html,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
csv.field_size_limit(10000000)
Q=522785199; A=18219397
work=json.loads((ROOT/'output/benchmarks/profiling/20261008_characterization_final/search_work.json').read_text())
data={}
for b,short in [('cuda','cuda'),('rt-custom','rt')]:
 rows=list(csv.DictReader(open('/tmp/sipros_'+short+'_metrics.csv')))
 units,raw=rows[-2:]
 inst=list(csv.DictReader(open('/tmp/sipros_'+short+'_instances.csv')))[-1]
 def val(k):
  try:return float(raw[k])
  except (KeyError,ValueError):return None
 def category(k):return {name.strip():int(n) for name,n in re.findall(r'([^;(]+): (\d+)',inst[k])}
 hw={'warp_instructions':val('smsp__inst_executed.sum'),'l2_bytes':32*val('lts__t_sectors.sum'),
     'dram_read_bytes':32*val('dram__sectors_op_read.sum'),'dram_write_bytes':32*val('dram__sectors_op_write.sum'),
     'kernel_ns':1e6*val('gpu__time_duration.sum')}
 hw['dram_bytes']=hw['dram_read_bytes']+hw['dram_write_bytes']
 samples={k:float(v or 0) for k,v in raw.items() if k.startswith('smsp__pcsamp_warps_issue_stalled_') and not k.endswith('_not_issued')}
 sampletotal=sum(samples.values())
 stalls={}
 for n in ['long_scoreboard','short_scoreboard','wait','not_selected','no_instruction','branch_resolving']:
  sk='smsp__pcsamp_warps_issue_stalled_'+('no_instructions' if n=='no_instruction' else n)
  stalls[n]={'warps_per_issue_active_cycle':val('smsp__average_warps_issue_stalled_'+n+'_per_issue_active.ratio'), 'pc_sample_percent':100*samples.get(sk,0)/sampletotal}
 base=ROOT/'output/benchmarks/profiling'/('20261008_batch8000000_'+b)
 nvtx={r['Range'].removeprefix(':'):float(r['Total Time (ns)'])/1e9 for r in csv.DictReader((base/'timings_nvtx_sum.csv').open())}
 manifest=json.loads((ROOT/'output/benchmarks/profiling'/('20261008_metrics_'+b)/'manifest.json').read_text())
 src=[ROOT/'mvh_cuda/cuda/scoring.cuh',ROOT/'mvh_cuda/cuda/engine.cu',ROOT/'MVH_RT/gpu_bridge/custom_device.cu']
 data[b]={'hardware':hw,'normalized_by_reported_queries':{k:v/Q for k,v in hw.items()},'normalized_by_associations':{k:v/A for k,v in hw.items()},
  'stalls':stalls,'raw_metrics':raw,'metric_units':units,'sass_categories_partial':category('sass__inst_executed_per_opcode_category'),
  'sass_opcodes_partial':category('sass__inst_executed_per_opcode'),'nvtx_seconds':nvtx,'ncu_manifest':manifest,'diagnostic':work[b],
  'diagnostic_binary_sha256':hashlib.sha256((ROOT/'build/mvh_rt/search_work_diagnostic/bin/sipros_mvh_cuda').read_bytes()).hexdigest()}
 shutil.copyfile('/tmp/sipros_'+short+'_metrics.csv',OUT/(short+'_hardware.csv'))
 shutil.copyfile('/tmp/sipros_'+short+'_instances.csv',OUT/(short+'_sass_instances.csv'))
report={'scope':'profiling_0.0 HEAD 8569c24 with conflict fixes; split-mz built-in spheres, NOT guaranteed double-precision equivalence',
 'reported_query_denominator':Q,'candidate_associations':A,'batch_index_zero_based':0,'generated_peptides':8000000,
 'timing_instrumentation':False,'diagnostic_instrumentation':True,'data':data}
(OUT/'metrics.json').write_text(json.dumps(report,indent=2))
c=data['cuda'];r=data['rt-custom']
def fmt(x):return 'N/A' if x is None else f'{x:,.3f}'
def table(headers,rows):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def para(s):return '<p>'+s+'</p>'
parts=['<h1>CUDA / OptiX RT 性能表征</h1>',para('2026-10-08 · RTX 5070 Ti Laptop · batch 8,000,000 · 仅表征，无算法优化。')]
parts+=['<h2>结论与证据范围</h2>',para('RT 的评分 kernel 更快与更低的读流量、更高的发射率和更多活跃线程一致；不能只归因于指令数、BVH 剪枝或 RT Core。额外约 24 亿条硬件 warp 指令无法用当前不完整的 OptiX SASS 归因完整分解。端到端差额主要由首次 RT 构建成本及共同主机阶段的运行差异解释。')]
parts+=['<h2>1. 可比性审计</h2>',para('两份 NCU manifest 的二进制、配置、FASTA、FT2 和 PTX SHA256 完全相同。输入为 Marine_fw_3rev.fasta + Pan_062822_X1iso5.FT2；并非 Marine RAW 转换文件。两边均为第一完整批（0-based batch 0），生成 800 万肽段、18,219,397 关联；不是尾批。首次结构准备发生在 RT 搜索 kernel 之前。两个后端使用同一 ion-generation/scoreCandidate 路径和相同同步边界。'),
 para('NCU --set full，launch-skip=0，launch-count=1，clock-control=none；默认 cache-control=all。CUDA 48 passes，RT 47 passes。没有应用级稳定态 warm-up，因此此采样是受缓存清理控制的首批测量，不应当作严格稳态 benchmark。SM 时钟约 2.499 / 2.444 GHz，DRAM 都约 10.991 GHz；频率并未锁定。NSYS 的 59 批平均和 NCU 的首批不能混为同一口径。'),
 para('当前实现并非保证保精度：RT 拆分 double m/z 后使用 float 内置 sphere，无 double 复核/fallback；容差边界、极近邻、等距命中可能不同。原完整搜索 PSM 哈希不同，首批命中计数相差 6。')]
rows=[]
for label,key,scale,unit in [('SM warp instructions','warp_instructions',1e9,'B'),('L2 traffic','l2_bytes',1e9,'GB'),('DRAM read','dram_read_bytes',1e9,'GB'),('DRAM write','dram_write_bytes',1e9,'GB'),('DRAM read+write','dram_bytes',1e9,'GB'),('NCU first kernel','kernel_ns',1e6,'ms')]:
 rows.append([label,fmt(c['hardware'][key]/scale)+' '+unit,fmt(r['hardware'][key]/scale)+' '+unit])
rows.append(['NSYS average scoring kernel','210.3266 ms','189.5108 ms'])
parts+=['<h2>2. 更新硬件对比</h2>',table(['Metric','CUDA','RT'],rows)]
rows=[]
keys=[('L1/TEX hit rate (%)','l1tex__t_sector_hit_rate.pct'),('L1/TEX throughput (% peak)','l1tex__throughput.avg.pct_of_peak_sustained_elapsed'),('L2 hit rate (%)','lts__t_sector_hit_rate.pct'),('L2 throughput (% peak)','lts__throughput.avg.pct_of_peak_sustained_elapsed'),('DRAM active (% elapsed)','dram__cycles_active.avg.pct_of_peak_sustained_elapsed'),('Eligible warps/scheduler','smsp__warps_eligible.avg.per_cycle_active'),('Achieved active warps/SM','sm__warps_active.avg.per_cycle_active'),('Achieved occupancy (%)','sm__warps_active.avg.pct_of_peak_sustained_active'),('Registers/thread','launch__registers_per_thread'),('Stack size (bytes, NCU launch metric)','launch__stack_size'),('Active threads/executed warp instruction','smsp__thread_inst_executed_per_inst_executed.ratio'),('Predicated-on threads/executed warp instruction','smsp__thread_inst_executed_pred_on_per_inst_executed.ratio')]
for label,key in keys:rows.append([label,fmt(float(c['raw_metrics'][key])),fmt(float(r['raw_metrics'][key]))])
rows.append(['Issued warps/scheduler active cycle',fmt(float(c['raw_metrics']['smsp__inst_issued.sum'])/float(c['raw_metrics']['smsp__cycles_active.sum'])),fmt(float(r['raw_metrics']['smsp__inst_issued.sum'])/float(r['raw_metrics']['smsp__cycles_active.sum']))])
rows.append(['Branch consistency','87.778%','N/A: reported 0 is not interpretable'])
parts+=[table(['Metric','CUDA','RT'],rows),para('更低 occupancy 不等于更慢：RT 每 scheduler 的可发射 warp 和实际发射率更高。寄存器压力存在，但未做单变量实验，不能量化其代价。CUDA 理论 occupancy 66.67%；RT 对应理论值未由本次 OptiX 报告可靠提供。')]
rows=[]
for n in c['stalls']:
 rows.append([n,fmt(c['stalls'][n]['warps_per_issue_active_cycle']),fmt(r['stalls'][n]['warps_per_issue_active_cycle']),fmt(c['stalls'][n]['pc_sample_percent'])+'%',fmt(r['stalls'][n]['pc_sample_percent'])+'%'])
parts+=['<h2>3. Warp stalls：两种归一化必须同时看</h2>',table(['Stall','CUDA: stalled warps / issue-active cycle','RT: same','CUDA PC sample share','RT PC sample share'],rows),
 para('per_issue_active.ratio 是 NCU 的 stalled-warps/发射活跃周期口径，不是 ms，也不是核函数总时间百分比。PC 采样占比按各后端所有 issue-state 样本求和归一化，包含 selected。两者分母不同。RT Long Scoreboard 的样本占比更高（45.81% vs 40.12%），但相对发射活跃周期的量更低（5.99 vs 13.30）；不能用前者单独否定、或后者单独证明总内存等待时间变短。'),
 para('Short Scoreboard 的两种口径都明显降低；该状态还包含 MIO、常量访问、特殊函数等依赖，不能直接当作 DRAM stall。Wait 和 branch resolving 的采样占比也不能简单换算成节省的毫秒。结合发射率 0.179→0.250、eligible 0.204→0.305、warp latency 36.795→14.112 cycles，调度/依赖行为改善有支持，但因果分摊未测。')]
parts+=['<h2>4. 指令 mix 与额外指令归因</h2>']
cats=sorted(set(c['sass_categories_partial'])|set(r['sass_categories_partial']))
parts.append(table(['可归因 SASS category（warp instructions）','CUDA','RT，部分覆盖'],[[k,f"{c['sass_categories_partial'].get(k,0):,}",f"{r['sass_categories_partial'].get(k,0):,}"] for k in cats]))
parts.append(para('CUDA SASS 分类覆盖约 160.079 亿条，几乎等于硬件总数；RT 分类只有 92.792 亿条，而硬件总数为 184.205 亿条，覆盖约 50.37%。因此不能用上述局部分类做完整差分，不能把剩余约 91.4 亿条都断言为 OptiX bookkeeping。'))
ops=['DADD','DFMA','DMUL','DSETP','FSETP','FFMA','HFMA2','MUFU','F2F','FRND','LEA','LD','LDG','LDL','STL','BRA','BSSY','BSYNC','RET']
parts.append(table(['可见 opcode','CUDA','RT，部分覆盖'],[[k,f"{c['sass_opcodes_partial'].get(k,0):,}",f"{r['sass_opcodes_partial'].get(k,0):,}"] for k in ops]))
parts.append(para('double/FP32/half 指令和比较都列出原始 opcode，避免把 category Floating Point 当成纯 FP32。当前源码确定 RT 额外执行 floor/subtract、float 转换、ray/payload 设置与 primitive-ID 映射；但无法把这些代码归因成那 24.12 亿条总差额。CUDA 硬件 branch 指令 29.234 亿，RT 24.073 亿；这反驳了“RT 总 branch 更多”的简单假设。'))
parts+=['<h2>5. 内存访问分解</h2>']
rows=[]
for label,key in [('L2 TEX-origin read','lts__t_sectors_srcunit_tex_op_read.sum'),('L2 TEX-origin write','lts__t_sectors_srcunit_tex_op_write.sum'),('L1 global load','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'),('L1 global store','l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'),('L1 local load','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum'),('L1 local store','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum')]:
 rows.append([label,fmt(float(c['raw_metrics'][key])*32/1e9),fmt(float(r['raw_metrics'][key])*32/1e9)])
parts.append(table(['sector traffic（GB）','CUDA','RT'],rows))
parts.append(para('不同层级的 sector 流量不能相加；L2 TEX-origin 分解仅覆盖该来源，不是所有 L2 客户端。最明显的变化在读取：DRAM read 8.607→4.449 GB；write 46.367→47.155 GB，写入并未减少。不能把全部流量下降都归给实验峰读取，因为局部数组、理论离子计算、结果和 OptiX 内部流量均包含在 kernel 内。'))
parts.append(para('CUDA 可见 local load/store 指令为 8.523/2.712 亿；RT 的标准 SASS local 计数返回 0，但 opcode 列表仍有 LDL/STL 且硬件 local sectors 非零，故 0 不代表不存在 local memory。CUDA NCU 可见 register-spill 归因为 0；动态索引局部数组等仍可能使用 local memory。RT 的 0 spill 也不能证明其完整内部无溢出。不能据此判断整数/小数表示增加了多少 spill。'))
parts+=['<h2>6. 实际搜索工作：独立诊断，不使用其耗时</h2>']
wc=work['cuda']['counts'];wr=work['rt-custom']['counts'];actual=wc['queries']
rows=[['日志 inrange（Top 合并后）',f'{Q:,}',f'{Q:,}'],['实际 lookup queries',f'{actual:,}',f"{wr['queries']:,}"],['CUDA matcher calls',f"{wc['cuda_calls']:,}",'N/A'],['Buckets visited',f"{wc['buckets']:,}",'N/A'],['Hub scalar accesses（2×visits）',f"{2*wc['buckets']:,}",'N/A'],['Peak entries / distance / tolerance checks',f"{wc['peak_checks']:,}",'N/A'],['optixTrace calls','N/A',f"{wr['traces']:,}"],['closest-hit callbacks','N/A',f"{wr['closest_hits']:,}"],['miss callbacks','N/A',f"{wr['misses']:,}"],['Scored fragment matches',f"{wc['scored_hits']:,}",f"{wr['scored_hits']:,}"],['Unscored searches',f"{wc['unscored']:,}",f"{wr['unscored']:,}"]]
parts.append(table(['Diagnostic count','CUDA','RT'],rows))
parts.append(para(f'实际查询比日志多 {actual-Q:,}。scoreCandidate 先推测性计算全部候选，scorePeptidesMVH 的 Top 合并阶段会把重复序列的 Result 清空并从 ScanCounts.predicted 排除。诊断 reported_queries 两边均精确恢复 522,785,199，invalid_queries=0；差额不是无效序列。两边工作相同，但原“queries”是合并后的统计，不能称为准确的 optixTrace 总数。'))
parts.append(para(f'以实际查询为分母：CUDA peak checks/query={wc["peak_checks"]/actual:.6f}，bucket visits/query={wc["buckets"]/actual:.6f}；RT traces/query={wr["traces"]/actual:.6f}。最接近的工作记录是查询开始前的 per-candidate 本地累加、完成后分片原子汇总；RT closest/miss 通过单次 trace 返回的 payload 分类，ANYHIT 已禁用，callback 各一次。未增加 custom intersection、双精度复核或命中后 rejection；“miss”与 class-0 未计分含义不同。'))
parts.append(para('诊断二进制独立目录 search_work_diagnostic、sm_120、FLOW_COUNTERS=ON。计时二进制不重建；5 项语义/计数契约测试通过。诊断从同一输入首批运行，在计数与结果下载完成后外部 SIGTERM，输出只用于计数，不是完整搜索、也不是性能基准。'))
parts+=['<h2>7. 归一化（按用户给定的日志查询分母）</h2>']
rows=[]
for label,k in [('SM warp instructions/query','warp_instructions'),('L2 bytes/query','l2_bytes'),('DRAM bytes/query','dram_bytes'),('kernel ns/query','kernel_ns')]:rows.append([label,fmt(c['normalized_by_reported_queries'][k]),fmt(r['normalized_by_reported_queries'][k])])
rows += [['CUDA peak comparisons / reported query',fmt(wc['peak_checks']/Q),'N/A'],['RT traces / reported query','N/A',fmt(wr['traces']/Q)]]
parts.append(table(['Normalized metric','CUDA','RT'],rows))
parts.append(para('每查询 ns 是批次总时间/查询总数，属于并行吞吐归一化，不是单条查询延迟。traces/reported-query 略大于 1 正是统计口径差额；改用实际查询则为 1。'))
parts.append(table(['Per candidate association','CUDA','RT'],[[k,fmt(c['normalized_by_associations'][k]),fmt(r['normalized_by_associations'][k])] for k in ['warp_instructions','l2_bytes','dram_bytes','kernel_ns']]))
parts+=['<h2>8. 端到端阶段分解（NSYS 59 批累计，秒）</h2>']
phases=[('Config/load','mvh/run/config_and_load'),('Scan preprocess','mvh/run/preprocess_scans'),('Generate peptides','mvh/search/generate_peptides'),('Continuous input packing','mvh/pack/continuous_inputs'),('Candidate assignment','mvh/batch/assign_scans'),('Peptide preprocessing','mvh/batch/preprocess_peptides'),('Scoring host packing','mvh/pack/all'),('GPU scoring service, inclusive','mvh/gpu/service'),('Host result conversion/merge','mvh/host/restore_results'),('Delete CPU peptide objects','mvh/batch/delete_peptides'),('PSM export','mvh/run/export_psms'),('End-to-end runner','mvh/run')]
parts.append(table(['Stage','CUDA','RT','RT−CUDA'],[[label,fmt(c['nvtx_seconds'][k]),fmt(r['nvtx_seconds'][k]),fmt(r['nvtx_seconds'][k]-c['nvtx_seconds'][k])] for label,k in phases]))
parts.append(para('主阶段分别覆盖同级或评分子阶段；存在初始化/清理等小残差，GPU service 包含下表子范围。下面子项不能再加到上面，也不能把 synchronization 与 GPU kernel 相加。'))
sub=[('Allocate/upload','mvh/gpu/allocate_upload'),('RT prepare/reuse','mvh/rt/prepare_or_reuse'),('Geometry creation/wait','mvh/rt/sphere/geometry_and_wait'),('Scene build','mvh/rt/SceneResources::build'),('BVH/GAS build/wait','mvh/rt/accel/build_and_wait'),('Theory cache','mvh/gpu/theory_cache'),('Search submit/wait','mvh/gpu/match_and_score'),('Result retain','mvh/gpu/retain_top'),('Result compaction','mvh/gpu/compact_results'),('Result download','mvh/gpu/download_results'),('Device synchronizations, overlapping','mvh/gpu/synced'),('Scoring completion wait, overlapping','mvh/gpu/scoring_completion_wait')]
parts.append(table(['Nested/inclusive stage','CUDA','RT'],[[label,fmt(c['nvtx_seconds'].get(k)),fmt(r['nvtx_seconds'].get(k))] for label,k in sub]))
parts.append(para('没有额外添加 NVTX：现有范围已覆盖要求的 preprocess、data preparation、geometry、build、search、copy、conversion、postprocess 和 synchronization；计时范围语义见 mvh_cuda/PROFILING.md。'))
parts+=['<h2>9. BVH 复用和成本摊销</h2>',para('首次准备一次 scene，含 46,065 个非空 scan 的 GAS 构建，不是每批重建。GPU optixAccelBuild 46,065 次合计 4.909 秒；host build/wait 5.260 秒。后续 58 批仅做 requireSameLayout，合计约 0.0186 秒；没有每批 BVH update。'),
 para('RT prepare/reuse 累计 5.562 秒 / 59 个 scoring launches = 94.27 ms/批。核心 GPU 评分累计 CUDA 12.409 秒 vs RT 11.181 秒，仅节省 1.228 秒，即平均 20.82 ms/批。首次构建摊销仍大于当前评分收益。以固定每批收益粗算约需 268 个等价完整批才能覆盖 5.562 秒，但 batch 总工作并不完全相同，不能当预测模型。'),
 para('整段搜索 RT 多约 6.61 秒：生成阶段 +2.26 秒，评分/恢复阶段 +3.88 秒，删除阶段 +0.38 秒，其他共同阶段与残差约 +0.09 秒。评分 GPU service 内 RT prepare +5.56 秒，allocate/upload −1.216 秒，评分 submit/wait −1.208 秒，其他小差额。全 runner 多约 6.96 秒；这是单次测量，不能把共同阶段差异全部视为 RT 算法成本。')]
parts+=['<h2>10. 按证据强度回答问题</h2>',
 para('<b>直接测量：</b>RT 流量减少主要在读端；发射率、eligible warps 和每条 warp 指令的活跃线程增加；occupancy 降低；首批和全批平均评分均更快。CUDA 查询平均仅检查约 0.27 个峰，RT 每个实际范围内查询一次 trace。BVH 构建一次并复用，首次成本超过这份搜索的 kernel 累计节省。'),
 para('<b>强支持的解释：</b>不同指令 mix 和 lane/调度效率意味着“指令更多”不必“时间更长”。Long/Short Scoreboard 相对 issue-active 的计数、读取流量与更高发射率共同支持依赖/访存行为改善。RT 取消了应用层显式桶和峰循环，工作移交给内置 traversal/intersection；但不代表其内部读取为零。'),
 para('<b>尚未证实：</b>20.82 ms 优势中各因素贡献多少；RT Core 本身的纯贡献；BVH 剪枝强度；额外 24.12 亿条指令的完整类别和内部地址/控制 bookkeeping 占比；寄存器压力和拆分表示单独增加的 spilling；严格稳定态或重复运行的置信区间。不能把 memory-stall 指标解释成已定位每一毫秒。'),
 para('CUDA branch consistency 87.78% 对应约 12.22% 非一致分支口径，不能当作 12.22% 时间损失。Branch resolving per-issue 0.839 vs 0.886，远小于 scoreboard 指标；当前证据不足以将分支作为主瓶颈。活跃线程 7.82→20.49 支持 lane 效率变化，但 RT 的标准 branch 一致率不可用。'),
 para('ncu --query-metrics 的当前设备目录没有提供可直接使用的 BVH node visits、built-in sphere tests 或 RT Core utilization 指标；命中 rtcore 的描述多数是 non-RTCORE，不能误认为硬件 RT 指标。当前 exact pruning 不可测，不能由 14% L2 降幅倒推节点数。')]
parts+=['<h2>11. 复现资料与限制</h2>',para('metrics.json 保存 NCU 原始值及单位、来源 manifest、归一化结果、诊断命令和计数。*_hardware.csv / *_sass_instances.csv 保留原始导出。build_report.py 可重新生成报告（需要原采样导出位于 /tmp）；run_search_work.py 为独立诊断运行器，输出目录需改成新路径以避免覆盖。诊断改动全部由 MVH_ENABLE_FLOW_COUNTERS 宏隔离，未改计时二进制、原匹配行为、几何表示或 block size。'),
 para('参考 NVIDIA 官方说明：<a href="https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html">Profiling Guide</a>、<a href="https://docs.nvidia.com/nsight-compute/ReleaseNotes/">OptiX profiling restrictions</a>。指标和结论以本机测量为准。')]
body=''.join(parts)
(OUT/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>CUDA RT 性能表征</title><style>body{max-width:1120px;margin:36px auto;padding:0 24px;font:16px/1.65 system-ui;color:#17212b}h1,h2{line-height:1.3}h2{margin-top:36px}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}td,th{border:1px solid #d7dfe7;padding:9px;text-align:left}th{background:#eef3f8}tr:nth-child(even){background:#fafbfd}p{margin:15px 0}a{color:#1261aa}</style>'+body+'</html>')
print(OUT/'report.html')
