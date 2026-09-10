# B 验证记录（2026-09-09）

全部构建和测试在 RX-104FF 的现有 `sipros-sipros-1` 容器中执行，项目路径 `/workspace/sipros`。
没有安装依赖、调整环境变量、修改容器/连接配置或覆盖 A/峰导出的历史实验结果。
工作区保留原 A 的未提交改动；B 新增在 `mvh/`，目前也未提交 Git。

## 最终实现

- `mvh::SearchSession` 支持原始谱图预处理和快照加载两条路径，均调用完整原 `searchDatabaseMvh()`。
- 格式 v1 显式写入标量/数组、配置和构建版本、CRC32；整数 scan 位置 ID 重建 precursor 指针和查询数组。
- 每轮清理已保留候选、检查临时关联已结束、重置累计状态并重建 MVH lnTable。
- 单列配置、读谱、预处理、快照保存/加载、每轮状态准备、完整搜索、TSV 导出时间。
- 原扫描类补上缺失 parent scan ID 的默认值 0，防止读取未初始化元数据。
- 原搜索函数、理论碎片、findNear、概率评分和 top 候选算法均未修改，批次仍为 2,000,000。

最终构建指纹：

```text
bcb776cf0e8cee3df39460c2a71591433ee6cde197fb6b3e4e9bf58871f1f14c
```

核心搜索使用与 A 相同的 `-O3 -DNDEBUG -ffast-math -march=native -fopenmp`，
宏定义 `GCC _FILE_OFFSET_BITS=64 _NOSQLITE`。快照 IO/校验库单独使用 `-fno-fast-math`，
确保 NaN/Inf 校验有效；不在该库中计算搜索分数。B 目标显式链接 OpenMP，不依赖其他子目录的变量作用域。

## 正确性验证

| 检查 | 结果 |
|---|---|
| 快照契约测试 | 往返、重复 precursor、非法 ID/排序/桶/分类/数值、配置不符、损坏及拒绝覆盖均通过 |
| 混合匹配/PTM、全无匹配、全跳过、多 precursor | 4 组；每组 1/4 线程 × 原始/快照路径 × 同进程 2 轮，共 32 次 B 搜索 |
| 上述 32 次搜索 | 全部与对应 A 的 TSV 逐字节一致，关键计数一致 |
| 预处理线程一致性 | 同一个合成输入用 1/4 线程生成的整个快照文件字节一致 |
| 原谱图缺失 | 移走自己创建的合成输入后，快照加载与搜索仍通过；测试后恢复输入 |
| 重复仪器 scan ID | 使用位置 ID 关联，不混淆两个同名 scan 对象；结果与 A 一致 |
| prepare-only | 不执行搜索，不要求 FASTA；生成的快照与直接搜索路径相同 |
| 读取错误 | 截断、CRC 损坏、格式版本、构建版本、非法长度、配置差异均拒绝 |
| 完整真实数据 | 4 线程原始路径 2 轮，4 线程快照路径 2 轮，1 线程快照路径 2 轮，共 6 次均与已验证 A 一致 |

真实输入：`test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2`；
配置 `experiments/Regular.cfg`；FASTA `raw/Ecoli.fasta`。
46,066 scans，275,099 precursor 条目，1 张 skipped scan，1,403,362 个保留候选。

六份真实结果均与 A 的以下 SHA-256 一致，包含分数、候选顺序、rank 和蛋白来源：

```text
2f96a76b931b89c64054026533e4de5379fca959848a0a314c8ea42332f31520
```

## 本次计时

每条路径在同一进程连续搜索两次；未更改原 profiling。单位：秒。

| 路径 | 线程 | 一次准备/加载 | 每轮状态准备中位数 | 两次完整搜索 |
|---|---|---|---|---|
| 原始读谱与预处理 | 4 | 8.11293 | 0.12338 | 17.65289, 17.02997 |
| 快照加载 | 4 | 1.20302 | 0.10841 | 17.69657, 17.42147 |
| 快照加载 | 1 | 1.22394 | 0.14061 | 45.49819, 44.98020 |

原始路径另花 1.74312 秒保存快照；这项一次性开销不计入搜索。
快照文件 606,073,827 bytes（约 578 MiB）。加载时间包含文件读取、CRC/结构校验和对象/关联重建。
上述计时是本次验证运行，不是对历史 A 的严格加速实验；CPU 负载、缓存和快照内存占用均可能影响搜索耗时。
B 的收益主要是省去重复读谱/排序/筛选，不代表新增了评分算法优化。

## 实验文件与复跑

最终有效记录：

- `output/mvh_B_contract_final_20260909/`
- `output/mvh_B_synthetic_final_20260909/`：32 次搜索、错误用例、A 对照、validation.json。
- `output/mvh_B_real_final_20260909/`：6 次完整搜索、validation.json、metadata.json、构建日志。
- `output/mvh_B_real_final_20260909/real/raw_t4/preprocessed.mvh`：可直接用于当前最终构建的快照。

`metadata.json` 包含输入、配置、FASTA、快照、二进制和源码哈希、实际 CPU/工具版本/cgroup 与现有相关环境变量。
首轮验证目录 `mvh_B_*_20260909` 也保留，但它们的快照属于前一次构建指纹；当前程序应拒绝加载。
使用包含 `final` 的目录。源代码/构建设置再次变化后，应按当前严格兼容规则重新生成快照。

容器内从最终快照复跑（输出目录须不存在）：

```bash
cd /workspace/sipros
build-mvh-only/bin/sipros_mvh_snapshot \
  --snapshot output/mvh_B_real_final_20260909/real/raw_t4/preprocessed.mvh \
  -c experiments/Regular.cfg -fasta raw/Ecoli.fasta \
  -o output/B_next_NEW -t 4 --repeat 2
```

程序退出后结果位于 `output/B_next_NEW/run_1/` 和 `run_2/`。
完整构建与测试说明见 [README.md](README.md)。

## 未覆盖与下一阶段

本次验证覆盖 FT2/Regular 和 CPU 1/4 线程，未新增 mzML 测试或 GPU 测试。
仍使用全局配置和 MVH 表，只支持一个活跃 session、串行调用会话方法；不支持加载后运行 WDP/Xcorr。
C 的独立 CPU/GPU 计算阶段、跨搜索版本快照兼容规则和特殊优化尚未实现。
