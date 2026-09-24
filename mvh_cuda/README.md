# CUDA MVH 搜索

输出路径统一约定见 [OUTPUT_LAYOUT.md](../OUTPUT_LAYOUT.md)。省略输出参数时自动写入项目 `output/` 下的分类运行目录；已有显式输出路径仍然有效。

在现有 `mvh_cuda/` 内直接演进的 GPU 实现。CPU 参考实现位于 `mvh/`，原始依赖及 CPU 校验函数位于 `original/`。本轮没有另建或保留一个 CUDA baseline 分支/副本。旧实验记录仍是历史记录，不代表当前代码。

当前不是“整个程序都运行在 GPU”：CPU 负责配置、文件读取、FASTA 酶切/PTM 枚举、原始质量估计、对数阶乘表和最终对象/文件输出；GPU 负责谱峰预处理、候选质量查询与关联、肽段中性丢失预处理、理论峰生成、峰匹配、评分与 top 决策。

## 从入口读到子函数

```text
app/main.cpp → mvh_app::run()
├─ loadMassData()                         CPU：文件与 precursor 索引
├─ preProcessAllMs2Mvh()                  GPU：实验峰预处理
└─ searchDatabaseMvh()                    原母函数名称保留
   ├─ ProteinDatabase::getNextPeptide()   CPU：酶切、PTM、肽段质量
   └─ processPeptideArrayMvh()            按生成的肽段分批
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
         └─ restoreScoringResults()      CPU：原 mergePeptide()/saveScore()
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

1. CPU 依原顺序生成一批肽段，将质量数组上传。
2. GPU 查询排序好的 precursor 表，使用计数和前缀和分配连续候选空间。
3. GPU 用 **稳定 radix sort** 按 scan 分组。每个 scan 内的肽段、窗口和 precursor 遍历顺序不变；不会去重多个 precursor 假设。
4. GPU 中性丢失后的文本保留在显存；正常模式不下载再上传。
5. 理论峰按 `(批内肽段对象 ID, charge)` 缓存，使用原来的 `CalculateSequenceIons()` 数值路径。不同对象即使序列相同，也暂不跨对象合并缓存；避免引入字符串去重和蛋白归属变化。
6. 每个候选由一个 GPU 线程评分。部分后来会被 merge 跳过的候选会被提前计算，但其结果不会进入原算法的逻辑统计或 top 更新。
7. 每个 scan 按原顺序应用 merge 与 top 50 更新。正常模式仅下载会合并蛋白名称或进入 top 的事件，CPU 用原函数恢复结果对象。

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

实验 RT 构建可选 `--match-backend rt-custom`，当前采用 `(m/z, 原始 class, 0)` 内置 sphere 和向下 closest-hit，保留原有两个三角形后端及默认 CUDA 路径。实现与验证说明见 [自定义匹配后端](../MVH_RT/gpu_bridge/CUSTOM_MATCHING.md)。
