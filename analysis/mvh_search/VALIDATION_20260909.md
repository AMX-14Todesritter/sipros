# A 验证记录：2026-09-09

## 环境和变更

宿主机 RX-104FF，现有容器 sipros-sipros-1，挂载项目 `/workspace/sipros`。
开始时宿主机仓库干净，HEAD `a9691be`，A 初稿及 `plot_export.py` 均已存在。
本次直接修改同一个远程工作区，没有覆盖同步、安装依赖、改环境变量或容器/连接配置。
容器 root 无法直接读取该仓库的 Git 状态（dubious ownership），Git 状态在宿主机检查；未修改 safe.directory。

修正参考目标编译属性继承和源码重新配置依赖；增加原 main 的线程参数包装；
新增目标/依赖库置于独立构建目录；修复峰导出复制规则及多入口下的唯一插入点。
`src/`、`include/` 与 `openmp/mvh_search_main.cpp` 保持原有 A 版本，原搜索算法没有改动。
既有 `output/peak_export_*`、`experiments/` 结果和 `plot_export.py` 保留。

构建：Release，GNU 11.4，CMake 3.22，OpenMP 4.5。
两个入口实际编译定义均为 `GCC _FILE_OFFSET_BITS=64 _NOSQLITE`，
编译选项均为 `-O3 -DNDEBUG -ffast-math -march=native -fopenmp`。
CPU：Intel Core Ultra 9 275HX，WSL 可见 12 CPU；容器 CPU quota、memory.max 均为 max；WSL 内存约 23 GiB。
未调整线程亲和性或关闭原 profiling。

## 正确性结果

| 数据/检查 | 运行组合 | 结果 |
|---|---|---|
| 合成混合输入 | 两入口 × 1/4 线程 × 2 次 | 8 次完整 TSV 字节相同；有效匹配、内部 M 氧化、重复蛋白、无匹配 scan、skip |
| 整份无匹配 | 两入口 × 1/4 线程 | 4 次相同，只有表头 |
| 整份跳过 | 两入口 × 1/4 线程 | 4 次相同，只有表头；skip=1 |
| 完整真实 FT2/FASTA | 两入口 × 1/4 线程 | 4 次完整 TSV 字节相同 |
| CLI | 已有目录、线程 0、SIP、缺失输入 | 均非零退出，已有文件不变，不创建被拒绝的新目录 |
| 峰导出副本 | prepare() | Expat xmlparse.c 存在，各插入钩子唯一；未重跑峰导出搜索 |

真实输入：`test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2`，
配置 `experiments/Regular.cfg`，数据库 `raw/Ecoli.fasta`。
读入 46,066 scans、275,099 precursor 条目，跳过 1 scan，保留 1,403,362 候选。
比较的是原始入口 MVH 返回时快照，尚未执行 WDP/Xcorr。
行序、rank、蛋白来源、所有导出数值也相同，没有排序后比较或容差归一化。

首轮真实验证的独立入口时间（仅作初测）：1 线程准备 11.2575 s、搜索 50.4940 s；
4 线程准备 8.07785 s、搜索 20.4445 s。该轮期间还运行过小型辅助验证，正式基准另行串行重复。

合成首版失败记录保留在 `output/mvh_A_synthetic_20260909`：
当时 fixture 误以为起始 M 会保留；原实现的 `Try_First_Methionine=true` 实际会先移除起始 M。
改用内部 M 后通过，未更改算法以迎合 fixture。

## 可复核文件

所有目录均位于项目 `output/`，新建且保留，不纳入 Git：

- `mvh_A_synthetic_v2_20260909`：8 次验证、配置、fixture、构建/配置日志。
- `mvh_A_unmatched_20260909` 和 `mvh_A_skipped_20260909`：边界用例。
- `mvh_A_cli_20260909`：拒绝输入日志、保留文件和峰导出副本。
- `mvh_A_real_20260909`：4 次完整真实输入对照，`validation.json` 含命令和哈希。
- `mvh_A_baseline_20260909`：独立入口 1/4 线程各 3 次串行重复、计时、TSV 和元数据。

## 串行重复基准

每个线程数 3 次独立进程运行；测量期间未并行执行其他验证。单位：秒。

| 线程 | 搜索三次值 | 搜索中位数 | 搜索范围 | 前置中位数 |
|---|---|---|---|---|
| 1 | 50.73394, 50.28884, 49.82518 | 50.28884 | 49.82518–50.73394 | 11.11509 |
| 4 | 19.01271, 18.61752, 19.52877 | 19.01271 | 18.61752–19.52877 | 8.07562 |

搜索中位数比值（1/4 线程）：2.645×。
六次结果均与前面四次真实参考验证的完整 TSV 相同。

真实数据十次运行的 TSV SHA-256：

```text
2f96a76b931b89c64054026533e4de5379fca959848a0a314c8ea42332f31520
```

原始重复值见 `output/mvh_A_baseline_20260909/benchmark_summary.json`。

## 范围

这是当前硬件、编译参数、profiling 和输入下的 CPU 基线，不是去掉 profiling 后的纯评分性能。
快照保存的是 top 候选，不是所有评分尝试。此次未验证 mzML、Windows 编译或其他线程数。
结果相同不等于所有数据均无顺序依赖；原始 findNear/top/多 precursor 语义保持不变。
B/C 尚未开始。
