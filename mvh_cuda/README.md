# CUDA MVH 搜索

输出路径统一约定见 [OUTPUT_LAYOUT.md](../OUTPUT_LAYOUT.md)。省略输出参数时自动写入项目 `output/` 下的分类运行目录；已有显式输出路径仍然有效。

在现有 `mvh_cuda/` 内直接演进的 GPU 实现。CPU 参考实现位于 `mvh/`，原始依赖及 CPU 校验函数位于 `original/`。本轮没有另建或保留一个 CUDA baseline 分支/副本。旧实验记录仍是历史记录，不代表当前代码。

当前不是“整个程序都运行在 GPU”：CPU 负责配置、文件读取、基础前体质量的最终浮点评估、对数阶乘表和结果对象/文件输出；默认 GPU 负责酶切、原子计数、PTM 枚举及质量偏移累加，以及谱峰预处理、候选质量查询与关联、肽段中性丢失预处理、理论峰生成、峰匹配、评分与 top 决策。`--peptide-generation cpu` 保留原肽生成路径供对照。

## 从入口读到子函数

```text
app/main.cpp → mvh_app::run()
├─ loadMassData()                         CPU：文件与 precursor 索引
├─ preProcessAllMs2Mvh()                  GPU：实验峰预处理
└─ searchDatabaseMvh()                    原母函数名称保留
   ├─ SearchPeptideGenerator            GPU：酶切、原子计数、PTM；CPU：读取/基础质量浮点评估
   └─ processPeptideArrayMvh()            按生成的肽段分批
      ├─ packPeptideBatch()               CPU：一次生成连续质量/文本/偏移数组
      ├─ assignPeptides2Scans()           GPU 批量分配
      │  └─ GetAllRangeFromMass()
      │     └─ GetRangeFromMass()
      ├─ preprocessingMVH()              GPU：中性丢失文本，留在显存
      └─ scorePeptidesMVH()
         ├─ packScoringBatch()           CPU：每个肽段/scan 的描述信息
         ├─ executeScoringBatch()
         │  ├─ CalculateSequenceIons()   GPU：批内理论峰缓存/直接生成
         │  ├─ ScoreSequenceVsSpectrum() GPU：每个候选一个线程
         │  │  ├─ findNear()
         │  │  └─ lnCombin()
         │  ├─ scorePeptidesMVH()        GPU：按原顺序合并及保留 top
         │  └─ gatherScoringEvents()     GPU：只收集合并/入选事件
         └─ restoreScoringResults()      CPU：仅保留最终 Top 的紧凑元数据
   └─ finishSearchResults()             CPU：搜索结束统一创建结果对象
```

为了批量调用 GPU，`assignPeptides2Scans()` 参数由单个肽段改为一批肽段，调用位置移入 `processPeptideArrayMvh()`。原 CPU `GetRangeFromMass()` / `GetAllRangeFromMass()` 方法仍保留供对照，生产 CUDA 路径调用 `assignment.cuh` 内的同名设备函数。没有为每个肽段启动一次 kernel。

## 文件职责

| 文件 | 阅读重点 |
|---|---|
| `src/database_search.cpp` | 数据库遍历、批次边界和母子函数关系 |
| `cuda/engine.h` | CPU 与 CUDA 模块的接口 |
| `cuda/engine.cu` | 显存所有权、批次组织、kernel 调用、CPU 校验和结果恢复 |
| `cuda/assignment.cuh` | 包含边界的质量查询、原窗口合并规则、关联生成和 scan 分组 |
| `cuda/theoretical.cuh` | 计数→前缀和→理论离子缓存 |
| `cuda/scoring.cuh` | 原理论离子公式、严格峰匹配、并行评分、按序 top 更新 |
| `cuda/preprocess.cuh` | 实验峰和肽段预处理 |
| `cuda/types.cuh` | 用整数 ID/数组偏移跨越 CPU/GPU 边界 |

关键的顺序约束、显存策略和计时边界使用英文注释。

## 这轮的数据组织

1. 肽生成器依原顺序交付一批肽段（默认 CUDA，CPU 模式供对照），一次性打包为 `PeptideBatch` 的质量、连续文本和偏移数组；同一批输入供关联、预处理和评分使用。
2. GPU 查询排序好的 precursor 表，使用计数和前缀和分配连续候选空间。
3. GPU 用 **稳定 radix sort** 按 scan 分组。每个 scan 内的肽段、窗口和 precursor 遍历顺序不变；不会去重多个 precursor 假设。
4. GPU 中性丢失后的文本保留在显存；正常模式不下载再上传。
5. 理论峰按 `(批内肽段对象 ID, charge)` 缓存，使用原来的 `CalculateSequenceIons()` 数值路径。不同对象即使序列相同，也暂不跨对象合并缓存；避免引入字符串去重和蛋白归属变化。
6. 每个候选由一个 GPU 线程评分。部分后来会被 merge 跳过的候选会被提前计算，但其结果不会进入原算法的逻辑统计或 top 更新。
7. 每个 scan 按原顺序应用 merge 与 top 50 更新。默认仅下载与本批最终 Top 有关的入选/合并事件，CPU 保存紧凑元数据，在搜索结束时统一恢复结果对象；逐批恢复模式保留作对照。

默认缓存 charge 0..8 的索引槽（0 电荷仍按原规则判无效）；更高电荷使用同一个公式直接在 GPU 计算，不截断、不回退 CPU。缓存槽数或显存预算不足时，整个批次使用直接 GPU 评分。缓存是可选的性能路径，不改变输出规则。

## 在现有 container 内运行

```bash
cd /workspace/sipros
cmake -S mvh_cuda -B build/mvh_cuda \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/mvh_cuda -j 4
ctest --test-dir build/mvh_cuda --output-on-failure --no-tests=error

build/mvh_cuda/bin/sipros_mvh_cuda \
  -f test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2 \
  -c experiments/Regular.cfg -fasta raw/Ecoli.fasta \
  -o output/mvh_cuda_new_run \
  --peptide-batch-size 2000000
```

输出目录必须不存在。输出为 `mvh_psms.tsv`、`run_summary.tsv`、配置副本。

`--peptide-batch-size` 默认 2,000,000，**现在计数的是生成的肽段，不是质量匹配成功的肽段**。减小它可以降低显存/内存峰值；跨批次保留 top 状态，scan 内候选顺序不变。`-t` 仅保留 CLI 兼容性，GPU 版本不使用 OpenMP 工作线程，它不是 GPU core 数量。

加 `--verify-cuda` 会逐项检查预处理、质量窗口、每次实际评分和 merge/top 结果；它还会下载所有候选用于 CPU 对照。这是正确性模式，不能作为性能模式。

CPU/GPU 三轮性能比较：

```bash
python3 mvh_cuda/tests/validate.py --real --cpu-threads 4 --repeats 3 \
  --output output/mvh_cuda_new_comparison
```

该脚本需要 `build/mvh/bin/sipros_mvh`。所有输出相同后保存一份压缩 PSM，删除本轮重复 TSV。原始数据、旧实验和 CPU 源码不会被删除。

## 日志计时和统计

- `run_summary.tsv/search_seconds`：完整搜索 wall，包括数据库遍历、酶切、PTM、赋值、GPU 工作和恢复；不包括读谱、实验峰预处理或导出。
- `[CUDA assignment]`：本批生成肽段数、命中肽段数、候选关联数及分配时间。
- `[CUDA scoring]`：打包、GPU 服务、恢复；GPU 服务内部再区分上传、理论峰缓存、候选评分 kernel、按序 top 更新、事件压缩和下载。
- `kernel_seconds` 是候选评分的 CUDA event 时间，**不包括**理论缓存构建和 top 更新；不要拿它单独与 CPU 完整搜索比较。
- `calls/success/inrange/matched` 保持原搜索的逻辑计数：已合并候选的提前评分不计入。
- 父区间包含子区间，不能重复相加。`cached_ions` 是本批缓存保存的理论离子数量，不是跨全部候选查询的峰数量。

## 保留的语义与限制

质量窗口包含端点；窗口索引合并仍使用原来的严格 `previous.last > next.first`，包括相接窗口重复访问同一个 precursor 的行为。峰匹配仍是严格 `abs(error) < tolerance`，等误差保留先遇到的峰，允许多个理论峰命中同一个实验峰。merge 仍只看当前 top 候选，保持同分排序和蛋白名称合并顺序。

仅支持 Regular；不运行 WDP/Xcorr、Percolator 或蛋白推断。设备容量仍为最多 128 个残基、512 字节中性丢失文本缓冲、8 个强度类别；超出时报错。单批关联数使用 CUB 的 int 索引，超出时需减小批次。缓存预算检查不能保证任意数据/任意显存均可容纳整个批次，显存不足时也应减小批次。

CPU `-ffast-math` 与 CUDA 显式浮点运算的精确对应已针对当前 GCC/CUDA 环境验证；更换工具链需重新校验。原始强度总和仅允许相对 `1e-12` 误差，它不参与 MVH 评分。

本轮结果见 `DEVICE_PIPELINE.md`；`OPTIMIZATION.md` 和旧 `VALIDATION.md` 是历史版本记录。

## 可选 RT 匹配后端

现有 GPU 数据流可通过 `MVH_CUDA_ENABLE_RT=ON` 接入三角形 RT，默认仍为 CUDA 桶搜索。
支持 `--match-backend cuda|rt-audit|rt-triangle|rt-instanced`。RT 精度问题尚未修复；audit 保留 CUDA 输出，
triangle 使用实际 RT 评分结果。构建、数据路径和验证范围见 [GPU RT 接入](../MVH_RT/gpu_bridge/README.md)。

纯 RT 模式会跳过查找桶的构建和上传；CUDA/audit 保留原桶路径。CPU 复算需要主机桶，但纯 RT 不再上传它。实现、日志字段与验证说明见 [RT 桶索引策略](../MVH_RT/gpu_bridge/README.md#rt-路径跳过桶索引)。

需要量化 RT 对 MVH 分数及最终 top 候选的影响时，使用 [评分影响验证脚本](../MVH_RT/gpu_bridge/README.md#用-mvh-分数和最终-top-候选评估差异)。`--score-impact` 为显式诊断开关，默认关闭，诊断耗时不作性能指标。

实验 RT 构建可选 `--match-backend rt-custom`，当前采用整数/小数拆分坐标内置 sphere 和沿 −z 的 closest-hit，保留原有两个三角形后端及默认 CUDA 路径。实现与验证说明见 [自定义匹配后端](../MVH_RT/gpu_bridge/CUSTOM_MATCHING.md)。

This branch retains only run-level timing and ordinary counters; application NVTX
and fine-grained timers have been removed. See [timing and build contract](PROFILING.md).

### Shared GPU input packing

All CUDA-based matchers (CUDA buckets, RT triangles, instanced triangles and
RT spheres) use the same optimized packing path. Fixed experimental peak arrays,
classes, lookup tables and scan metadata are packed once per dataset. Each batch
still updates its candidate ranges and current top list.

- `--spectrum-cache host` (default): reuse packed host arrays; allocate/upload and
  release their GPU copies for each scoring batch. This avoids keeping these
  device copies alive during the next candidate-association stage.
- `--spectrum-cache device`: also keep those GPU copies between batches. This can
  reduce transfers but may raise peak VRAM during association and influence the
  existing theoretical-ion cache policy. CUDA includes its mass buckets; pure RT
  does not. The scoring log records `spectrum_device_bytes` and `spectrum_cache`.

Both modes retain fixed host arrays for the dataset; they are reset before scan
preprocessing and at the end of the search scope. Candidate metadata remains
batch-local; Top entries and their counts stay on the GPU between batches, even
with `--spectrum-cache host`. The sphere geometry, tracing policy and scoring
formula are unchanged.

`BatchSequenceIds` in `include/sequence_ids.h` uses contiguous ID slots,
first-seen entries and owned key bytes. Linear probing with at most 50% occupancy
replaces per-key hash nodes; stored hashes avoid rehashing strings during growth.
Offsets keep keys valid when the byte buffer grows. Exact length/byte comparison
preserves duplicate handling, including when result restoration replaces a
source top-candidate string. After restoration, `retain()` keeps only sequences
referenced by the final Top lists. Their IDs stay stable across batches; IDs with
no remaining Top references are recycled. This bounds the dictionary by live Top
keys plus the current batch instead of all peptides ever generated. `reset()`
restarts IDs for a new dataset while preserving capacity when called explicitly.
Unique new keys still require hashing and copying, and retained keys are compacted
on the CPU after each batch.

A dataset-scoped host workspace reuses scoring input, initial/final Top, event and
count arrays, plus peptide preprocessing text/rule/length/error scratch. Peptide
pointers are borrowed from the caller through synchronous restoration instead of
copying the pointer array; preprocessing writes offsets directly into the prepared
batch. Dataset preprocessing/reset and search-scope exit release the workspace.
This retains host memory at the largest capacity reached during the dataset and
can increase peak RSS when later stages allocate; it is not a memory-reduction
claim. Step 4 also reuses bounded device workspace capacity; peptide objects still
follow their prior lifetimes.
The workspace is for serial batches; a future pipeline needs separate workspaces
for concurrently active batches.

Top reuse is shared by CUDA and RT-custom. The first scoring batch seeds a single
device Top buffer from the host; later batches read/update that buffer in place
and get their initial Top counts from persistent device counters. Each scan owns
its slice, preserving the original candidate order and equal-score sorting.
Dataset reset releases both buffers. This removes repeated initial-Top packing, upload and allocation. Top entries/counts
are still downloaded for ID retention and metadata maintenance. Step 4 defers
result-object creation to search completion and filters events to surviving Top
entries. Verification and `--result-restoration batch` retain the original CPU
`mergePeptide()`/`saveScore()` replay as an independent reference.

Validation for GPU Top reuse: all 15 CTest tests pass, including synthetic full
Top/tie/replacement/empty-batch cases, randomized live-ID retention, dataset reset,
and CUDA/RT-custom repeated-protein searches with one batch versus batches of 3
in both normal and verified modes. No Marine performance gain is claimed without
a fresh before/after benchmark.

Rebuild `build/mvh_rt/gpu_integration` before running the normal benchmark scripts.
Old `packing_optimized` and `profile_enabled` binaries may still contain profiling;
they are historical artifacts, not builds of the current clean source. Both spectrum
cache modes remain available through `--spectrum-cache host|device`.

## 连续肽输入（第 2 步）

`include/peptide_batch.h` 定义拥有数据的 `PeptideBatch`，包含连续的质量数组、
以 NUL 结尾的序列文本及带末尾哨兵的偏移数组。数组下标就是批内肽 ID，重复肽
保留独立条目和原始枚举顺序。`clear()` 复用容量，数据集重置释放工作区。

生产路径在 `processPeptideArrayMvh()` 中调用一次 `packPeptideBatch()`。
候选关联和中性丢失预处理接收 `const PeptideBatch&`，不再从 `Peptide*` 逐个提取
质量或复制字符串。评分输入的序列 ID 也从连续文本生成。无需展开时，GPU 在
上传的紧凑文本上直接处理；规则需要更大容量时，GPU 从紧凑输入复制到展开空间，
避免 CPU 构建含空白容量的文本数组。展开路径需要临时 GPU 输入文本和偏移缓冲。

第 2 步引入连续输入时保留了 CPU 肽生成；第 3 步将默认生成计算迁到 GPU（见下文）。
结果对象仍保留，尚未消除 `new Peptide/delete`。对象只用于
输入转换、原有合成测试的关联适配、CPU 校验和结果恢复；预处理后的长度和校验用
文本在恢复边界写回对象。关联、匹配公式、候选顺序及蛋白归属规则保持不变。
后续 GPU 肽生成可以基于这一数组接口继续演进。主机输入数组保留最大容量，
因此本轮不宣称峰值内存下降或 Marine 已提速。

验证覆盖连续数据所有权、容量复用、变长/重复/空序列、空批次保持 Top、GPU
中性丢失展开与删除，以及现有 CUDA/RT-custom 多批次结果对照。
本阶段容器 Release 构建完成，16/16 CTest 通过；CUDA contract 的
Compute Sanitizer memcheck 报告 0 errors。

## GPU 肽生成（第 3 步）

Regular 搜索默认使用 `--peptide-generation cuda`，与 `--match-backend` 独立，
因此 CUDA 和 RT-custom 评分均可使用。`--peptide-generation cpu` 运行原始
`ProteinDatabase`，用于回退和性能/正确性对照；运行摘要记录所选生成后端。

- CPU 流式加载/清理 FASTA，每块最多 256 条蛋白，达到约 1 MiB 序列后换块
  （单条蛋白可能超过该阈值）。GPU 处理起始 M 移除、酶切位点、漏切组合及长度过滤。
- GPU 统计每条基础肽的原子组成和 PTM 组合数量。CPU 对计数做前缀和并检查溢出，
  按每页最多 65,536 条生成结果分配空间，不一次展开整个数据库的修饰组合。
- GPU 按原始顺序生成：蛋白顺序、漏切数、酶切起点、原肽、PTM 数量、位点组合、
  修饰类型排列；同一组合中第一个位点的修饰类型变化最快，质量偏移从末位向前累加。
  末端修饰、每位点多种修饰和重复肽均保留。
- 为保持现有 CPU `-ffast-math` 下的质量窗口语义，基础前体质量的最终浮点评估仍在
  `src/generation_mass.cpp` 的 CPU 兼容层执行，输入是 GPU 计算的六种原子计数。
  它没有再次遍历肽序列；PTM 质量累加在 GPU 执行。本阶段尚非完全 GPU 常驻生成。
- CPU 将下载的生成页转换成现有结果对象，再进入第 2 步的连续输入路径。
  这保留了结果恢复兼容性，也仍有下载、对象创建和重新打包成本；尚不宣称搜索提速。

CUDA 生成的配置长度上限为 128，与当前评分残基容量一致；计数使用 64 位整数，
PTM 组合计数溢出会明确报错。未新增 Mutation/SIP 搜索支持。

`--verify-cuda` 在 CUDA 生成模式下，额外逐条调用未经修改的 CPU 生成器，精确比较
序列、原肽、质量、蛋白名称、位置及侧翼残基，最后检查总条数相同。
生成对照覆盖跨蛋白块、跨页、多类型/末端 PTM、零 PTM、漏切和起始 M 规则、
不同同位素元素、128 残基边界及计数溢出。实际 Ecoli 数据库的 **3,956,849** 条
肽已全部通过精确 CPU 对照（`experiments/Regular.cfg`）。

## 最终结果恢复与有界工作区（第 4 步）

默认 `--result-restoration final`，CUDA 和 RT-custom 共用此路径。每个 GPU Top
条目记录来源：本批入选候选索引，或前一批 Top 的名次。评分后，仅保留最终入选
事件，以及该次入选之后发生的蛋白归属合并事件；本批已被淘汰候选的事件不再下载。
若同一序列被淘汰后再次入选，旧蛋白合并历史不会混入新结果。

CPU 每张谱图最多保存 TopN 条紧凑元数据，按 GPU 返回的名次更新，不再逐批创建、
排序、删除 `PeptideUnit`。蛋白名称合并刻意保留原函数的字符串匹配语义。
`finishSearchResults()` 在搜索结束、输出前统一构建结果对象；不缓存整个搜索的
全部事件历史。生成用的 `Peptide` 对象以及每批 CPU 元数据维护仍然存在，尚非
完全 GPU 常驻搜索。

`--result-restoration batch` 保留旧逐批恢复路径，便于直接对照。
`--verify-cuda` 在 final 模式中同时运行旧路径，并逐批核对最终 Top 的序列、
分数、蛋白名、质量、电荷、长度和侧翼信息。回归测试包含合并→淘汰→重新入选。

扫描描述、肽描述、评分结果、选择索引、选择计数、事件和 CUB 选择临时空间使用
可复用设备缓冲区。每个缓冲区超过 **32 MiB** 时，在评分结束后释放，避免大块
评分存储常驻而挤占下一批关联阶段的显存。这七个工作缓冲的跨批保留上界为
224 MiB，不包括已有的 Top、谱图缓存和 RT GAS；数据集重置统一释放工作区。
理论离子缓存和候选关联等其他临时存储仍按原生命周期管理。

验证：17/17 CTest 通过；CUDA contract 和正常 final 模式多批次搜索的
Compute Sanitizer memcheck 均为 0 errors。实际 Ecoli 使用 batch=250,000，
共 16 批，CUDA 与 RT-custom 各自的 batch/final 恢复输出 SHA-256 完全一致，
每份输出 1,403,362 条 PSM。记录见
[恢复方式对照](../output/validation/deferred_results_ecoli_20261001T185301Z.json)。
该记录为单轮正确性对照，耗时不作为正式性能结论；尚未进行 Marine 对照。


## Peak selection: nearest mass

CPU and CUDA select the nearest experimental peak within strict
`abs(peakMz-queryMz) < tolerance`, regardless of intensity class. Equal distances
retain the first encountered peak. A nearest class-0 peak remains unscored.

RT-custom splits double m/z into integer and fractional float coordinates, adds
boundary copies, and traces downward from z=0.5 through equal-radius spheres.
Its float boundary and tie behavior still requires comparison with the double
reference. See [sphere matching](../MVH_RT/gpu_bridge/CUSTOM_MATCHING.md).

## Candidate reuse distribution

`--candidate-reuse-stats` prints an exact per-batch histogram of candidate counts
for each observed `(batch peptide ID, precursor charge)` pair. `[REUSE histogram]`
rows contain charge, uses and number of groups; `[REUSE batch]` verifies that the
weighted histogram equals the assignment association count. Unused pairs are
excluded. Duplicate precursor hypotheses and candidates on skipped scans remain
included, so this is candidate reuse opportunity, not distinct-scan counts or
actual ray counts. Equal sequences with different peptide IDs remain separate.
The diagnostic is off by default, runs before scan sorting, and does not change
scoring. Its runtime must not be treated as an uninstrumented benchmark.

### Instrument-only FT2 precursor

FT2 input uses the precursor m/z from the S line and the charge from the first
Z line's first item. The remaining Z fields and additional Z lines are ignored.
Each scan contributes exactly one neutral precursor mass: z * (m/z - proton mass).
Missing or nonpositive primary charge is an input error; no alternative charge
or isolation-window precursor is inferred. This changes candidate lists and
search results for files containing extra precursor hypotheses. MzML input
retains its existing behavior.
