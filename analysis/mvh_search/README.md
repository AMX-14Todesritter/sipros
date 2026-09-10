# MVH 独立入口（A）

A 复用 `MS2ScanVector::searchDatabaseMvh()`：完整读取 FASTA、酶切/PTM、质量窗口分配、
`preprocessingMVH()`、理论碎片、`findNear()`、MVH 概率评分和每张 scan 的候选保留。
批次仍为 2,000,000；不运行 WDP/Xcorr、Percolator 或蛋白推断。
B 的预处理快照与同进程重复搜索见 [mvh 模块文档](../../mvh/README.md)，C 尚未实现。

## 环境与构建

所有编译、运行、测试均在 RX-104FF 的现有 `sipros-sipros-1` 容器中执行。
宿主机 `/home/ams098z/projects/sipros` 已 bind mount 到 `/workspace/sipros`。
先在宿主机检查 `git status --short`，不要用旧目录覆盖远程代码或实验输出。
进入现有容器：

```bash
docker exec -it sipros-sipros-1 bash
cd /workspace/sipros
cmake -S . -B build-mvh-only \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_CONDA=ON \
  -DSIPROS_MVH_ONLY_BUILD=ON -DSIPROS_MVH_REFERENCE_TEST=ON
cmake --build build-mvh-only \
  --target sipros_mvh_search sipros_mvh_reference -j 4
```

依赖或环境配置需要变更时先征得同意。已验证环境使用 GNU 11.4、CMake 3.22、OpenMP 4.5。
`BUILD_CONDA=ON` 在本项目中选择动态 OpenMP 链接和 `-ffast-math -march=native`，无需安装 Conda。
MVH 两个程序输出到 `build-mvh-only/bin/`；MVH-only 模式的依赖静态库输出到构建目录的 `lib/`，
不会覆盖旧 `bin/` 或 `MSToolkit/lib/`。只构建上述目标。
参考目标关闭时只需构建 `sipros_mvh_search`。

## 单次完整搜索

```bash
build-mvh-only/bin/sipros_mvh_search \
  -f /workspace/sipros/test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2 \
  -c /workspace/sipros/experiments/Regular.cfg \
  -fasta /workspace/sipros/raw/Ecoli.fasta \
  -o /workspace/sipros/output/mvh_NEW_RUN \
  -t 4
```

每次一个配置、一个 FT2/mzML、一份 FASTA，只允许 `Search_Type = Regular`。
输出目录必须不存在；失败运行也保留目录和诊断，重试请换新目录。
`-fasta` 覆盖配置中的数据库路径。配置副本保持原文，实际生效的 FASTA 路径记录在 summary。
`-t` 直接调用 OpenMP API，不修改环境变量；省略时使用现有 OpenMP 默认值。

输出：

- `mvh_psms.tsv`：每张 scan 已保留的候选，最多 50 个，不是全部评分尝试。
  `mvh_score` 只读取 `vdScores[2]`。`mvh_rank` 是原保留列表位置，保留同分顺序和蛋白名称顺序。
  `scan_index` 从 0 开始；同一 scan 的多个 precursor 假设由候选的质量、电荷记录。
- `run_summary.tsv`：scan、precursor、跳过数、候选数、线程上限和各阶段墙钟时间。
- `input_config.cfg`：原始配置副本。

计时定义：`config_and_load_seconds` 包含配置加载、配置副本/目录创建、读谱与 precursor 索引构建；
`preprocess_seconds` 包含峰排序/筛选与 MVH lnTable 初始化；二者相加为 `prepare_seconds`。
`search_seconds` 包含完整 `searchDatabaseMvh()` 调用（包括内部准备、清理、原 profiling 和日志），
不含 TSV 导出，也不含程序最终析构。不可拿原 `CLOCKSTOP` 的局部数字替代这一计时。
`omp_max_threads` 是配置上限，不是实测每个并行区域的工作线程数。

## 结果一致性验证

参考目标使用原始 `openmp/main.cpp` → `startProcessingMvh()`。
构建目录中的 `ms2scanvector_reference.cpp` 只在 `searchDatabaseMvh()` 返回后插入 TSV 导出和 return，
因此快照早于 WDP/Xcorr。生产源码不被插入提前返回。
`reference_main.cpp` 仅处理并移除 `-t` 参数，再调用原始 main；剩余参数沿原 CLI 解析。
两个目标在平台定义设置完成后复制编译选项、定义和链接库。
修改 `src/ms2scanvector.cpp` 会触发 CMake 重新生成参考副本。

以下全部在容器项目目录执行，输出名每次必须新建：

```bash
python3 analysis/mvh_search/validate.py --output output/mvh_synthetic_NEW --repeats 2
python3 analysis/mvh_search/validate.py --output output/mvh_unmatched_NEW --case unmatched
python3 analysis/mvh_search/validate.py --output output/mvh_skipped_NEW --case skipped
python3 analysis/mvh_search/check_cli.py \
  --binary build-mvh-only/bin/sipros_mvh_search \
  --fixture output/mvh_synthetic_NEW --output output/mvh_cli_NEW
python3 analysis/mvh_search/validate.py --output output/mvh_real_NEW \
  --input test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2 \
  --config experiments/Regular.cfg --fasta raw/Ecoli.fasta
```

默认比较 1 和 4 线程；`--threads 1 4 --repeats 2` 可显式控制组合。
脚本逐次运行两个入口，严格比较完整 TSV 字节，包括分数、行序、同分位置和蛋白来源，
并核查计数、有限分数、rank 范围、批次大小。任何不同即失败，不通过排序或放宽容差掩盖差异。
每次成功比较后将命令、SHA-256、行数、独立入口 summary 写入 `validation.json`；日志和 TSV 均保留。
合成 fixture 包含未排序 b/y 峰、内部 M 氧化、重复蛋白、无匹配质量和低峰数 skip。
`Try_First_Methionine=true` 当前会移除起始 M，fixture 因而使用内部 M，不改变原语义。

检查 CLI 同时只生成峰导出源码副本，确认 Expat 的 `lib/xmlparse.c` 存在且各钩子唯一。
旧峰导出工具仍减小批次并提前退出，不能拿其耗时作为完整搜索基准。

## 基准使用边界

完整真实输入、固定 FASTA/config、编译器/选项、CPU/WSL/容器限制、线程数和 profiling 状态需要一起记录。
采用独立新进程重复测量，分别报告前置和搜索耗时，并记录每次完整结果哈希。
保留所有重复值后报告中位数/范围；不将合成用例耗时作为性能结论。
这里没有关闭原 profiling、设置线程亲和性或改变环境变量；Windows/WSL 调度与其他负载仍可能影响结果。
1/4 线程在本次输入上一致不证明所有数据与平台均一致。
特别保留严格 `abs(error) < tolerance`、等距先遇峰、多离子共用峰、top 序列合并顺序、
多 precursor 和质量窗口合并的原行为。首次 GPU 重构不能隐式改变这些规则。

完成参考入口一致性验证后，串行重复独立入口进行计时，并记录身份信息：

```bash
python3 analysis/mvh_search/validate.py --output output/mvh_baseline_NEW \
  --input test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2 \
  --config experiments/Regular.cfg --fasta raw/Ecoli.fasta \
  --standalone-only --threads 1 4 --repeats 3
python3 analysis/mvh_search/summarize.py --output output/mvh_baseline_NEW \
  --reference output/mvh_real_NEW
python3 analysis/mvh_search/record_metadata.py --output output/mvh_baseline_NEW
```

`metadata.json` 保存输入/源码/二进制/构建配置 SHA-256、工具版本、CPU、cgroup 限制及相关现有环境变量。
元数据只记录当前状态，需保证测量期间没有其他人修改输入或重建程序；另外保存宿主机 Git revision/diff。
元数据文件同样拒绝覆盖。`--standalone-only` 不代替参考入口验证。

本次实际验证记录见 [VALIDATION_20260909.md](VALIDATION_20260909.md)。
