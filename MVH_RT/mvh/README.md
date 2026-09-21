# 独立的 MVH 搜索类

本目录只有一个入口：读取谱图并预处理，然后运行提取后的 `MvhScanVector::searchDatabaseMvh()`（函数名和函数体保持原样）。没有 A/B 分支、快照加载、GPU 或新增函数 profiling。

## 目录

```text
mvh/
├── include/mvh_scan_vector.h 独立类 MvhScanVector，仅保留所需成员
├── src/
│   ├── mvh_scan_vector.cpp    构造、析构与 MVH 工作区
│   ├── spectrum_input.cpp    读谱、precursor 索引、实验峰预处理
│   └── database_search.cpp   候选分配、批次处理与搜索主函数
├── extraction.json          21 个提取方法的来源映射
├── original/                 原始依赖及来源对照
│   ├── src/                  12 个原始实现文件
│   ├── include/              12 个原始头文件
│   └── manifest.json         来源提交和每个文件的 SHA-256
├── app/
│   ├── main.cpp              参数检查
│   ├── runner.h
│   └── runner.cpp            直接调用独立类方法、计时、TSV 导出
├── tests/                    源码一致性和单样本集成测试
├── CMakeLists.txt            独立构建入口
└── README.md
```

来源：`96161334d96c249abedef727962784653a90653b`。这不是当前带历史 profiling 的工作区源码。`original/src/` 和 `original/include/` 的 24 个文件均从该提交完整提取，逐字节相同，没有更改函数签名、函数体、宏、数据结构或注释。来源校验会比较实际 Git 内容，而不只是比较本地清单。

实际运行的管理类为 `MvhScanVector`。21 个方法从原 `MS2ScanVector` 提取，方法定义仅替换类名；构造/析构名称随类名同步改变，普通函数名完全保留。类声明调整了访问权限并移除了 WDP/Xcorr、SIP 与 task 调度成员。`original/src/ms2scanvector.cpp` 不再编译，仅保留来源对照。其他原始依赖暂时仍按原翻译单元编译。

原项目的类与翻译单元耦合较强，因此保留了其中 WDP/Xcorr/SIP 的定义及编译依赖，但本入口不调用这些后续流程。没有复制完整项目、第三方库、实验数据或构建目录；MSToolkit 使用仓库中的现有源码，与来源提交一致。当前模块需要父项目的 MSToolkit，不能单独拷走 mvh 后直接编译。

## 执行路径与代码位置

1. `app/main.cpp` 检查输入和输出路径；`app/runner.cpp` 加载配置并指定 FASTA。
2. `src/spectrum_input.cpp::loadMassData()` 读谱，建立 precursor 查询数据。
3. 同文件 `preProcessAllMs2Mvh()` 调用实验峰预处理，初始化概率查找表。
4. `src/database_search.cpp::searchDatabaseMvh()` 完整执行：
   - `proteindatabase.cpp`：FASTA、切点、肽段与 PTM 枚举；
   - `averagine.cpp`、`isotopologue.cpp`、`proNovoConfig.cpp`：原质量计算及配置；
   - `assignPeptides2Scans()`：质量窗口查询和 scan 关联；
   - `processPeptideArrayMvh()` → `peptide.cpp::preprocessingMVH()`；
   - `ms2scan.cpp::scorePeptidesMVH()`：已有候选合并检查；
   - `MVH.cpp::CalculateSequenceIons()`：理论碎片；
   - `ms2scan.cpp::PeakList::findNear()`：实验峰查询；
   - `MVH.cpp::ScoreSequenceVsSpectrum()`：命中分类计数及概率评分；
   - `ms2scan.cpp::saveScore()`：保留高分候选；
   - 原函数自身的批次和工作区清理。
5. 返回入口，导出 MVH 结果。不会调用 `startProcessingMvh()`，因为它还会执行 WDP/Xcorr 和 ensemble 输出。

### 如何阅读入口

`app/runner.cpp` 的 `mvh_app::run()` 直接执行：

```cpp
MvhScanVector spectra(input, output, config, true);
spectra.loadMassData();
spectra.preProcessAllMs2Mvh();
spectra.searchDatabaseMvh();
```

三个方法名与原项目一致。没有 `access()`、模板标签或私有成员绕行。预处理和搜索改为独立类的公开入口，内部辅助函数保持私有；原名 `vpAllMS2Scans` 作为公开结果集合供入口读取。类组织和访问权限是有意调整，函数定义只允许类名替换，由 `verify_extraction.py` 自动检查。阅读算法从 `src/database_search.cpp` 开始。

## 构建与运行

所有命令必须在现有 `sipros-sipros-1` 容器中执行。

```bash
cd /workspace/sipros
cmake -S mvh -B build/mvh -DCMAKE_BUILD_TYPE=Release
cmake --build build/mvh -j 4
ctest --test-dir build/mvh --output-on-failure --no-tests=error
```

只使用 `build/mvh/`，不再创建 build-mvh-only、build-mvh-gpu 等并列目录。无需 CUDA、MPI 或新增依赖。当前已验证环境是原容器 Linux/GCC；编译选项沿用原项目 BUILD_CONDA=ON 的 `-ffast-math -march=native` 和 OpenMP。

```bash
build/mvh/bin/sipros_mvh \
  -f test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2 \
  -c experiments/Regular.cfg \
  -fasta raw/Ecoli.fasta \
  -o output/mvh_run_01 \
  -t 4
```

输出目录必须不存在，父目录必须存在。`-t` 是搜索线程数；构建命令中的 `-j` 是编译并行度。不提供 `-t` 时使用 1 个线程。每次一个配置、一份 FT2/mzML、一份 FASTA，只支持 Regular。mzML 路径继承原读取实现，本轮真实数据验证使用 FT2。

输出：
- `mvh_psms.tsv`：scan 保留列表，读取已计算的 `vdScores[2]`，不读取 WDP/Xcorr。最多 50 个候选/scan；rank 是原列表位置，同分顺序不重新排序。
- `run_summary.tsv`：配置/读谱、预处理、完整搜索调用、导出分别计时，外加 scan/skip/PSM 数量。不是函数内部 profiling。
- `input_config.cfg`：配置副本。

输入路径会写入 TSV；跨目录比较时应注意这个身份字段。峰匹配严格小于 tolerance、同误差保留先遇到的峰、多离子可命中同峰、候选合并的顺序依赖、批次大小 2,000,000 均继承原始实现。原代码本来存在的 CLOCKSTART/CLOCKSTOP 宏也没有删除。

## 验证

`verify_original.py` 检查 24 个原始对照/依赖文件与 Git 来源逐字节一致。`verify_extraction.py` 检查 21 个提取的方法定义，除类名外逐字一致，并确认入口使用直接调用。`check_run.py` 检查有效匹配/PTM、单线程和四线程一致、无匹配、跳过谱图，以及无效线程数和拒绝覆盖输出。测试使用保留的 scan 1004 单样本，仅验证集成行为；不能将它作为完整数据集的性能基准。

独立类改造后的验证见 `CLASS_EXTRACTION_VALIDATION.md`；此前原样移植记录见 `VALIDATION.md`。旧 A/B/C 代码备份仍在父项目 `.reset-backup/`，不参加当前构建。
