# B：MVH 预处理快照与可重复搜索

在同一个 Sipros 仓库内提供 `mvh::SearchSession`，两种输入均调用原始
`MS2ScanVector::searchDatabaseMvh()`，保持 FASTA、酶切、PTM、质量分配、理论碎片、峰匹配、MVH 评分与 top 候选保留。
批次为 2,000,000，使用原 CPU/OpenMP 实现；没有实现 GPU 或改变评分规则。

## 构建（仅在现有容器中）

宿主机 RX-104FF，项目 `/home/ams098z/projects/sipros` 已挂载为 `/workspace/sipros`。
使用现有 `sipros-sipros-1`，不启动新容器：

```bash
docker exec -it sipros-sipros-1 bash
cd /workspace/sipros
cmake -S . -B build-mvh-only \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_CONDA=ON \
  -DSIPROS_MVH_ONLY_BUILD=ON -DSIPROS_MVH_REFERENCE_TEST=ON \
  -DSIPROS_MVH_SNAPSHOT_TESTS=ON
cmake --build build-mvh-only \
  --target sipros_mvh_snapshot sipros_mvh_search sipros_mvh_reference mvh_snapshot_contract_test -j 4
```

依赖沿用 C++17/OpenMP 和仓库自带 zlib，无新增安装。依赖、环境变量和容器/连接配置变更仍须先征求用户同意。
所有产物位于 `build-mvh-only/`，实验输出使用新建 `output/` 子目录。
源代码位于 `mvh/include/mvh/`、`mvh/src/`，测试位于 `mvh/tests/`。
A 的旧输出、峰导出和其他实验结果不会被覆盖。

## 使用

以下均在容器项目目录执行，每次替换为新的输出目录名称。

**只准备快照，不搜索，也不要求 FASTA 存在：**

```bash
build-mvh-only/bin/sipros_mvh_snapshot \
  -f test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2 \
  -c experiments/Regular.cfg -o output/B_prepare_NEW -t 4 --prepare-only
```

**从快照加载，在同一进程、同一个对象上连续搜索两次：**

```bash
build-mvh-only/bin/sipros_mvh_snapshot \
  --snapshot output/B_prepare_NEW/preprocessed.mvh \
  -c experiments/Regular.cfg -fasta raw/Ecoli.fasta \
  -o output/B_loaded_NEW -t 4 --repeat 2
```

**原始读谱/预处理 → 保存快照 → 连续搜索两次：**

```bash
build-mvh-only/bin/sipros_mvh_snapshot \
  -f test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2 \
  -c experiments/Regular.cfg -fasta raw/Ecoli.fasta \
  -o output/B_raw_NEW -t 4 --repeat 2
```

只允许 `Regular`。每次一个配置、一个谱图或快照；搜索必须显式提供 `-fasta`。
快照不含数据库候选，加载后仍完整读取指定 FASTA，可通过 `-fasta` 换数据库。
快照中的原始谱图路径仅用于身份和输出，不会打开或检查其存在；TSV 因而可与 A 逐字节比较。
加载需要与生成时配置文件的**内容字节完全相同**，文件路径可以不同。配置中的 FASTA 项由 CLI 参数覆盖。

输出根目录和每轮结果目录均要求不存在，失败目录也保留，重试请换新名字。

```text
output/B_raw_NEW/
├── input_config.cfg
├── preprocessed.mvh           原始输入模式生成；快照模式不再复制
├── preparation.tsv            一次性配置/输入准备计时和来源
├── run_1/
│   ├── mvh_psms.tsv
│   └── run_summary.tsv
└── run_2/
    ├── mvh_psms.tsv
    └── run_summary.tsv
```

`mvh_psms.tsv` 直接复用 A 的 writer，只导出已保留的最多 50 个候选和 `vdScores[2]`，保持顺序。

## 计时与状态

`preparation.tsv` 分别记录：

- `config_seconds`：原有配置解析。
- `spectrum_load_seconds`：原有读谱和 precursor 索引构建；快照模式为 0。
- `preprocess_seconds`：原有峰排序、筛选、分类、索引、统计及首次 lnTable 的创建/释放；快照模式为 0。
- `snapshot_save_seconds`：提取字段、校验、序列化、校验和及写文件。
- `snapshot_load_seconds`：读文件、CRC/版本/配置/结构校验、分配对象及重建关联；不包含 lnTable 初始化。

每轮 `run_summary.tsv` 分别记录：

- `state_prepare_seconds`：清理上一轮 top 候选及计分状态、恢复相关全局边界、重新创建 lnTable。
- `search_seconds`：完整原 `searchDatabaseMvh()`，包含原 profiling、日志和内部清理；不含 TSV 导出。
- `export_seconds`：调用原 MVH TSV writer 的时间。

`--repeat` 每次都重新读 FASTA 并完整搜索，清空上轮结果，不使用候选缓存。
预处理快照保持不变；临时肽段、线程工作数组及 lnTable 仍按原函数生命周期分配/释放。
输出计数核对 scan、precursor、skip 和保留候选数量。
B 额外把 lnTable 创建单列，因此比较准备成本时要包含 `state_prepare_seconds`；A 将首次创建计入预处理。
快照不保存原始实验强度数组，加载后的内存占用可能小于原始输入路径；不要将这个差异当作评分算法优化。

## 文件契约与兼容性

见 [snapshot.h](include/mvh/snapshot.h) 和 [snapshot.cpp](src/snapshot.cpp)。格式 v1：

- 明确的 little-endian u32/u64、i32、IEEE754 binary64、长度前缀字符串/数组；不写对象内存、指针或 STL 布局。
- 原始 scan 顺序、scan ID、parent scan ID、保留时间/类型、分辨率、precursor 属性和假设。
- `pPeaks`、`pClasses`、整数桶索引、分类计数、m/z 边界、总桶数、skip 及强度统计。
- 排序后的 `(mass, charge, scan_index)` 列表；保留同质量顺序和重复项，由此精确重建查询质量数组及指针关联。
- 原始输入路径、配置原文、全局 m/z/质量/电荷边界、格式版本及代码/构建指纹。
- 整个文件末尾的 CRC32 用于发现意外损坏，不作为密码学身份认证。

`scan_index` 是位置 ID，不是仪器 scan ID；允许多个对象拥有相同仪器 scan ID。
空 PeakList 的未初始化桶边界不读取，以规范化的零值保存。
未提供 parent scan ID 时原构造函数现初始化为 0；不参与搜索评分。

默认拒绝代码/构建指纹不同或配置内容不同的快照，不提供强制忽略选项。
当前指纹保守地覆盖核心代码、B 模块、CMake 和主要编译设置；修改这些内容并重建后需重新生成快照。
C 阶段若要跨后端/搜索版本共享快照，应先建立经过验证的预处理兼容版本规则，而非直接跳过校验。

加载先检查 CRC、格式版本、指纹和配置，再检查长度、有限数值、排序、分类/桶一致性和 ID 范围，之后才构造搜索对象。
当前读取器限制文件不超过 8 GiB；峰索引遵守现有 `short` 的范围。它是本研究的内部缓存格式。

## 验证

```bash
build-mvh-only/bin/mvh_snapshot_contract_test output/B_contract_NEW
python3 mvh/tests/validate.py --output output/B_synthetic_NEW
python3 mvh/tests/validate.py --output output/B_real_NEW \
  --input test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2 \
  --config experiments/Regular.cfg --fasta raw/Ecoli.fasta \
  --reference output/mvh_A_real_20260909/sipros_mvh_search_t1_r0/mvh_psms.tsv \
  --threads 4 1 --raw-threads 4 --repeat 2
```

真实对照路径应指向已验证的 A TSV，旁边需要 A 的 `run_summary.tsv`。
脚本比较所有 TSV 字节和关键计数，绝不排序、舍弃 rank 或放宽浮点容差。
合成测试覆盖混合匹配/PTM、全无匹配、全跳过、多 precursor、重复仪器 ID，
1/4 线程原始预处理快照相同；两条路径各在同一进程连续搜索两次。
加载测试暂时移走自己创建的合成输入以证明不依赖原谱图，完成后恢复；不会移动真实实验文件。
另检查 prepare-only、输出拒绝覆盖、配置差异、版本差异、校验和、截断和非法长度。

## 当前边界

`SearchSession` 尚依赖 Sipros 全局配置和 MVH 表，每个进程只允许一个活跃 session；
调用方须先加载匹配的 Regular 配置，并且在 session 生命周期内不改变配置。
会话方法由一个控制线程串行调用，不支持多个调用者并发操作同一 session；内部搜索仍使用 OpenMP。
失败搜索不可复用 session。加载快照不支持后续 WDP/Xcorr、蛋白推断或恢复原始峰强度。
GPU、数据布局优化和算法拆分属于 C，目前均未实现。

上述完整验证完成后，可以记录最终结果与环境/文件哈希：

```bash
python3 mvh/tests/record_results.py \
  --output output/B_real_NEW --synthetic output/B_synthetic_NEW
```

该汇总命令对应默认 32 次合成搜索及示例中的 6 次真实搜索组合；要求统一构建指纹，
生成拒绝覆盖的 `metadata.json`，包含原始重复计时、输入/快照/二进制哈希、CPU 和环境信息。

本次通过验证的实际快照与复跑命令见 [2026-09-09 验证记录](VALIDATION_20260909.md)。
