"""Consolidate measured timing scopes and independent equivalence evidence."""
import json,hashlib,subprocess,csv,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';D=E/'diagnostics/continuous_control';A=E/'diagnostics/marine_query_audit'
t=json.loads((D/'analysis.json').read_text());audit=json.loads((A/'comparison.json').read_text());meta=json.loads((E/'diagnostics/marine_control/manifest.json').read_text())
replay=json.loads((E/'results/accuracy/marine_continuous_replay/manifest.json').read_text());normal=json.loads((E/'results/accuracy/marine_service_1_auto/manifest.json').read_text())
checks={r['backend']:r['psm_sha256']==next(x for x in normal['runs'] if x['backend']==r['backend'])['psm_sha256'] for r in replay['runs']}
assert all(checks.values())
assert all(r['canonical_candidate_differences']==0 and r['query_fingerprint_differences']==0 and r['preprocessed_spectra_identical'] and r['audit_psm_matches_native'] for r in audit['rows'])
service_csv=list(csv.DictReader((D/'service_runs.csv').open()));cache_consistency={b:len({r['psm_sha256'] for r in service_csv if r['backend']==b})==1 for b in ('cuda','rt-instanced','rt-custom')}
assert all(cache_consistency.values())
status={'status':'completed','large_benchmarks':'paused','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'replay_output_matches_native':checks,'cache_output_unchanged':cache_consistency,'audit_passed_for_cuda_and_rt_custom_on_this_subset':True,'rt_instanced_equivalence':False,'timing_and_audit_builds_separate':True}
(D/'status.json').write_text(json.dumps(status,indent=2))
lines=['# Marine 固定工作负载：连续计时与查询一致性诊断','','## 范围与方法','',
'固定 Pan X1 前 256 张谱图、Marine 前 2,048 个 target/decoy 家族（4,096 条序列）、multi 前体、batch 上限 2,000,000。生成 3,555,023 条肽段，分 2 批，150,120 个候选关联。',
f"输入 SHA256：FT2 `{meta['scans_sha256']}`；FASTA `{meta['fasta_sha256']}`。",
'独立 timing build 仅修改 host launch 驱动；离子公式、匹配、几何、MVH 算法源文件 SHA256 与构建前一致。每批交替 cache-on/off 与 off/on 顺序，每条件预热 3 次、测量 7 次；每次 cache-on 都重新分配和构建缓存。固定候选，重放只覆盖 Result，Top retention 在最后执行一次。重放最终 PSM 与普通运行逐后端 SHA256 一致。',
'连续区间从 theory 前已同步的 host 起点开始，到评分结束事件完成、计时读取后结束，使用单一单调 wall clock 直接测量，包含缓存分配/构建、同步、评分事件管理与提交、评分执行；不包含区间前的候选上传/RT geometry 准备、区间后缓存析构与 Top retention。起点前同步不计入区间。此边界不是 kernel GPU 时间戳，也不是完整服务。',
'另外使用未重放、未插入查询观察器的普通诊断二进制，做每条件 3 个独立新进程，全六条件串行，交替缓存条件并轮换后端顺序。完整服务采用现有连续时间戳 `executed - packed`（gpu_service_seconds），包括 executeScoringBatch 的上传、RT 准备、theory、评分、保留、压缩、下载与临时资源释放；不包括 host packing 与后续恢复。packing→恢复另列。新进程服务包含首次运行影响，不能当作同上下文热服务吞吐量。',
'', '## 同上下文连续计时', '', '| 后端 | 缓存 | 缓存准备中位数 ms | 融合 kernel 中位数 ms | 连续区间中位数 ms | 连续区间 min–max ms |','|---|---|---:|---:|---:|---:|']
for r in t['continuous']:lines.append(f"| {r['backend']} | {r['cache']} | {r['theory_median_ms']:.3f} | {r['fused_median_ms']:.3f} | {r['continuous_median_ms']:.3f} | {r['continuous_min_ms']:.3f}–{r['continuous_max_ms']:.3f} |")
lines+=['','连续列独立测量；各阶段中位数不能简单相加替代它。各批相同 trial 的区间相加后取 7 次中位数，批间生成/调度时间不计入。','', '## 完整服务对照（3 个独立新进程的中位数）','','| 后端 | 缓存 | GPU 评分服务 ms | packing→恢复 ms | MVH search ms |','|---|---|---:|---:|---:|']
for r in t['service']:lines.append(f"| {r['backend']} | {r['cache']} | {r['gpu_service_ms']:.3f} | {r['pack_service_restore_ms']:.3f} | {r['search_ms']:.3f} |")
lines+=['','packing→恢复由同次运行的连续相邻 start/packed/executed/restored 时间戳差组成，不拼接不同运行。MVH search 包含肽段生成等更大范围；不等于进程端到端耗时。RT 准备与首次加载对小负载的服务耗时影响很大，kernel 加速不能直接推广到服务加速。RT-custom 的 packing→恢复缓存开关几乎持平，不能声称该口径显著改善。','','## 逐候选、逐查询与峰级检查','','使用另一份独立非计时审计 build，沿实际融合评分路径观察请求，保留现有匹配结果，同时用严格 double 容差最近峰 oracle 对照。原项目算法文件未改动，审计副本只增加观察器，不替换结果。审计与非审计普通运行的最终 PSM 逐后端/缓存 SHA256 一致。',
'全部六组的预处理谱图/峰类别输出 SHA256 一致；150,120 条候选按顺序逐项核对原始 scan ID、charge、precursor mass、实际评分序列字符串一致。每个候选针对范围内 query 的原始 double bits 计算两个顺序相关 64-bit 指纹，包含查询计数；全部逐候选指纹一致。物理查询合计 3,112,061，不混用结果恢复后的 inrange。指纹一致是强证据，但不是保存每个 double 的完整逐项原始 dump，仍存在理论哈希碰撞可能。','','| 后端 | 缓存 | 物理查询 | 候选差异 | 查询指纹差异 | 峰身份差异 | 类别差异 |','|---|---|---:|---:|---:|---:|---:|']
for r in audit['rows']:lines.append(f"| {r['backend']} | {r['cache']} | {r['physical_queries']:,} | {r['canonical_candidate_differences']} | {r['query_fingerprint_differences']} | {r['peak_identity_mismatches']} | {r['class_mismatches']} |")
lines+=['', 'RT-instanced 实际命中 193,720 次，严格 oracle 命中 194,220 次；500 次差异均表现为本次命中总数减少。样本文件每批最多 128 条，完整差异数量由所有查询的计数器给出，不能据有限样本断言每种差异的分布。其最终 8 个谱图结果组不同，缓存开关均相同。CUDA 与 RT-custom 在该子集对照 oracle 峰身份/类别差异均为 0；此前全 X1 的 RT-custom 边界差异并未因此消失。',
'旧日志 RT-instanced 的 inrange 比 CUDA 多 47，但实际发出的查询及其序列指纹一致。因此 +47 不能解释为多发了 47 个匹配请求；差别属于后续结果计数/恢复口径，尚未沿各恢复分支逐项归因。',
'', '## 已验证结论','','1. 本次小负载中，缓存降低融合 kernel 时间，但每批重建缓存的成本超过收益；连续准备＋评分三个后端均更慢。CUDA 15.325→4.656 ms，RT-instanced 12.756→2.563 ms，RT-custom 12.652→2.417 ms（缓存开启→关闭）。',
'2. 在关闭缓存、相同实际查询的条件下，RT-custom 热 kernel 相对 CUDA 约 1.98×，连续准备＋评分区间约 1.93×；这些不是全评分服务或整个搜索的加速比。',
'3. RT-instanced 仍存在匹配正确性差异，不能作为等价实现的有效性能基准。',
'4. 所有三后端的缓存开关最终 PSM 不变。缓存不能解释本次 RT-instanced 正确性差异。',
'', '## 未解决与建议后续','','- 缓存是否在更多 scan、候选复用率更高时摊薄成本：未在本轮验证；当前不能概括为缓存永远无用，也不应据此修改默认策略。',
'- 缓存构建阶段约 10–12 ms 的分配、dense key 扫描、prefix sum、生成与同步各自贡献，需要将本次 Marine 的 GPU 时间线及 CUDA API 时间分开分析；仅凭本轮 wall 时间不能指定唯一瓶颈。',
'- 若需严格原始逐项证明，可保存完整 query double dump；当前为逐候选双指纹。',
'- RT-instanced 的 float 几何/射线精度与漏匹配因果验证仍需另一个小型边界实验；本轮不修改匹配算法。',
'- 大型 batch、DB-1、Soil DB-2、scan 扩展继续暂停。待匹配语义与计时口径验证后再决定是否恢复。',
'', '## 文件与复现','','- `continuous_timings.csv`、`service_runs.csv`、`service_summary.csv`：测量分组与服务原始重复。','- `build.json`、`runs.json`、`status.json`：构建指纹、命令、结果验证。初次驱动构建出现局部变量作用域错误，已修复；成功编译日志为 build_retry.log，保留失败日志备查。','- 相邻 `marine_query_audit/comparison.json` 与 `comparison.csv`：全部查询审计摘要。','- `output/evaluation/results/accuracy/marine_continuous_replay/`、`marine_service_*`、`marine_query_audit_*`：完整配置、日志、PSM、查询指纹及峰级样本。','- 驱动代码在 `evaluation/diagnostics/`；生成的源码副本、构建产物与全部数据均在 output。','- 每条件服务只测 3 个新进程，热重放只有一个上下文；GPU 时钟未锁定，桌面负载与温度影响未隔离，不声称统计显著或外推到全 Marine/Soil。','']
(D/'report.md').write_text('\n'.join(lines))
print(json.dumps(status,indent=2))
