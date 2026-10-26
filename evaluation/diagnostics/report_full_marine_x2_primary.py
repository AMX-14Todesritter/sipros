"""Primary-only full-workload timings and separate first-launch NCU evidence."""
import json,csv,sys,statistics,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';rid=sys.argv[1];D=E/'diagnostics'/rid;P=E/'results/accuracy'/rid
n=json.loads((P/'manifest.json').read_text());m=json.loads((D/'manifest.json').read_text());profiles=json.loads((D/'profiles/manifest.json').read_text());metrics=json.loads((D/'profiles/selected_metrics.json').read_text())
runs={r['backend']:r for r in n['runs']};rows=[]
for b in ('cuda','rt-custom'):
 r=runs[b];t=r['totals'];rows.append({'backend':b,'generated_peptides':r['generated_peptides'],'candidates':int(t['candidates']),'reported_queries':int(t['inrange']),'reported_matched':int(t['matched']),'batches':len(r['batches']),'fused_seconds':t['fused_device_seconds'],'gpu_service_seconds':t['gpu_service_seconds'],'pack_seconds':t['pack_seconds'],'restore_seconds':t['replay_verify_seconds'],'complete_scoring_seconds':sum(t[k] for k in ('pack_seconds','gpu_service_seconds','replay_verify_seconds')),'search_seconds':float(r['summary']['search_seconds']),'rt_setup_seconds':r['rt_setup_seconds']})
assert rows[0]['candidates']==rows[1]['candidates'];assert all(int(b['cached_ions'])==0 and int(b['cache_charge_stride'])==0 for r in n['runs'] for b in r['batches'])
with (D/'primary_timings.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (P/'rt-custom_differences.tsv').open() as f:differences=list(csv.DictReader(f,delimiter='\t'))
(D/'result_discrepancies.json').write_text(json.dumps(differences,indent=2,ensure_ascii=False));scanids=sorted({r['scan_id'] for r in differences},key=int)
old=json.loads((E/'results/accuracy/full_marine_x2_20261026T183852Z/manifest.json').read_text());oldruns={r['backend']:r for r in old['runs']}
for b in ('cuda','rt-custom'):
 r=runs[b];p=D/'profiles'/b/'capture.log';text=p.read_text();assign=[l for l in text.splitlines() if l.startswith('[CUDA assignment]')]
 # NCU may kill before buffered host stdout is flushed. Source command and exact input/binary SHA remain provenance.
 expected=r['batches'][0]['candidates']
 (D/'profiles'/b/'provenance.json').write_text(json.dumps({'baseline_first_batch_candidates':int(expected),'baseline_first_batch_reported_queries':int(r['batches'][0]['inrange']),'captured_assignment_log':assign,'binary_sha256':m['binary_sha256'],'first_scoring_launch_only':True},indent=2))
a,c=rows;ratios={k:a[k]/c[k] for k in ('fused_seconds','gpu_service_seconds','complete_scoring_seconds','search_seconds')}
status={'baseline_completed':True,'profiling_status':profiles['status'],'primary_only':True,'cache_disabled':True,'candidate_counts_equal':True,'full_query_fingerprints_collected':False,'psm_identical':m['psm_identical'],'differing_scan_ids':scanids,'summary':rows,'ratios':ratios,'other_large_suites_paused':True};(D/'primary_analysis.json').write_text(json.dumps(status,indent=2))
lines=['# Marine 全库 + X2：primary-only 运行与 kernel profiling','','## 条件','','Marine 完整 FASTA、X2 全部 46,862 张谱图，batch 12,000,000（生成肽段条目），无理论离子缓存，不展开多前体，仅原始 S 行主前体 m/z + 第一条 Z 电荷。两个后端各一遍普通进程，无预热；本轮先 RT-custom 后 CUDA。其他实验队列继续暂停。',f"每后端生成 {a['generated_peptides']:,} 条肽段，{a['candidates']:,} 个候选关联，{a['batches']} 个评分批次。日志 reported inrange 均为 {a['reported_queries']:,}，这不是逐查询物理指纹审计。",'','## 无 profiler 性能结果','','| 范围 | CUDA s | RT-custom s | CUDA/RT |','|---|---:|---:|---:|']
for label,k in [('融合评分 kernel','fused_seconds'),('设备评分服务（不含 host 打包和恢复）','gpu_service_seconds'),('完整评分阶段','complete_scoring_seconds'),('完整 MVH search','search_seconds')]:lines.append(f'| {label} | {a[k]:.3f} | {c[k]:.3f} | {ratios[k]:.3f}× |')
lines+=['','完整评分阶段 = 同次运行的 host packing + GPU service + host restoration，相邻 start/packed/executed/restored 时间戳差的总和；包含 RT 结构准备、设备上传、缓存检查、评分、Top 保留、压缩、下载、临时资源释放与 host 结果恢复。不含肽段生成、候选分配、原始谱图预处理或导出。不要将设备评分服务的比例简称为完整评分加速。',f"RT 一次性 setup 为 {c['rt_setup_seconds']:.3f} s；本轮融合评分节省 {a['fused_seconds']-c['fused_seconds']:.3f} s，但完整评分只节省 {a['complete_scoring_seconds']-c['complete_scoring_seconds']:.3f} s。",'','## 独立 Nsight Compute profiling','','两个后端分别采集本次输入与相同二进制的第一个真实评分 kernel，batch 12M，CUDA 过滤 ScoreSequenceVsSpectrum、RT 过滤 optixLaunch。NCU detailed kernel replay，完成首个捕获后按 --kill yes 故意结束诊断子进程；它不是完整搜索，也不是基准运行。普通基准在 profiler 启动前已经完整结束。原始 .ncu-rep、raw CSV、日志、命令和单位均保留。',f"Profiling 状态：{profiles['status']}。",'','| 指标（原始单位） | CUDA | RT-custom |','|---|---:|---:|']
lookup={r['backend']:r for r in metrics}
labels={'kernel_time':'NCU replay kernel duration（仅 profile）','registers_requested':'请求寄存器/线程','entry_stack_bytes':'入口栈大小','achieved_occupancy_pct':'实际 occupancy','sm_warp_instructions':'SM warp 指令','dram_reads':'DRAM 读取','dram_writes':'DRAM 写入','fp64_pipe_elapsed_pct':'FP64 管线 elapsed 占比','local_load_requests':'local load 请求','local_store_requests':'local store 请求'}
if len(lookup)==2:
 for key,label in labels.items():
  def value(b):
   r=lookup[b];v=r.get(key);return 'unavailable' if v is None else f"{v:,.4g} {r.get(key+'_unit','')}"
  lines.append(f'| {label} | {value("cuda")} | {value("rt-custom")} |')
lines+=['','计数器描述的是首批内核，不代表所有 39 批。请求寄存器不等于实际分配数量，入口栈大小也不等于 spill 字节；local 请求不能全部认定为寄存器溢出。occupancy、FP64 百分比或 SM 指令数量单独不足以认定唯一瓶颈。OptiX traversal 的专用硬件工作不由 SM 指令数完整覆盖，本轮未直接测量 RT Core 利用率。NCU replay duration 不替代无 profiler 的 event 计时或端到端时间。','', '## 与上一轮 multi 的边界','','| 模式 | 候选关联 | 日志 reported inrange（CUDA） |','|---|---:|---:|']
for mode,r in [('multi',oldruns['cuda']),('primary-only',runs['cuda'])]:lines.append(f"| {mode} | {int(r['totals']['candidates']):,} | {int(r['totals']['inrange']):,} |")
lines+=['','两轮输入和 batch 相同，但发生在不同时段，执行顺序相反，没有锁 GPU 时钟。关闭多前体同时改变候选数量、电荷分布与离子生成工作组成，因此不能把速度变化简化为候选数比例，也不能说只有 multi 才能出现 RT kernel 收益。本次 primary-only 也观测到 RT kernel 更快，但完整评分和搜索收益很小，且单次数据不能声称统计显著。','', '## 正确性限制',f"45,258 个有结果谱图 Top-1 肽段、电荷和分数一致；最终保留列表中 {n['comparisons']['rt-custom']['spectrum_groups_differ']} 个谱图组不同：{', '.join(scanids)}。公共保留候选评分差异为 0，但候选集合不同，PSM SHA256 不同。",f"日志 reported matched：CUDA {a['reported_matched']:,}，RT-custom {c['reported_matched']:,}；这也是结果统计，不能替代峰级差异审计。",'本轮未采集全量逐候选/逐查询指纹或峰级 oracle。不能据先前小子集审计宣称本轮匹配完全等价；也不能未经定点验证就把差异全部归因于 float 边界。需要对这些 scan 定点审计，核心匹配算法本轮保持不变。','', '## 证据文件','','- primary_timings.csv / primary_analysis.json：统一计时口径和检查结果。','- profiles/manifest.json：NCU 命令、状态；profiles/selected_metrics.csv/json：原始指标单位。','- profiles/{cuda,rt-custom}/capture.ncu-rep、raw_metrics.csv、capture.log：完整 profiler 数据。','- result_discrepancies.json：保留结果差异。','- output/evaluation/results/accuracy/'+rid+'/：普通运行原始配置、日志与最终 PSM。','']
(D/'primary_report.md').write_text('\n'.join(lines));print(json.dumps(status,indent=2))
