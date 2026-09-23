# 导出一份实验峰和对应候选理论峰

输出路径统一约定见 [OUTPUT_LAYOUT.md](../../OUTPUT_LAYOUT.md)。省略输出参数时自动写入项目 `output/` 下的分类运行目录；已有显式输出路径仍然有效。

本工具只支持 `Search_Type = Regular`，使用当前项目的真实读谱、数据库酶切、质量窗口分配和 `MVH::CalculateSequenceIons()`。
它不修改原始 `src/`、`include/`、CMake 文件或现有二进制；在输出目录复制源码，注入导出钩子并独立编译。

## 在服务器 container 中运行

先检查远程 Git 状态，再逐文件合并改动；不要覆盖远程 `plot_export.py` 或已有实验结果。
当前 RX-104FF 使用现有容器，从宿主机进入：

```bash
docker exec -it sipros-sipros-1 bash
```

安装依赖、修改环境变量、挂载或容器配置前，先征求用户同意。

容器中（替换为实际输入路径）：

```bash
cd /workspace/sipros
python3 analysis/peak_export/export_one.py \
  --input /workspace/sipros/data/sample.ft2 \
  --config /workspace/sipros/configTemplates/Regular.cfg \
  --fasta /workspace/sipros/data/database.fasta \
  --scan-id 120 \
  --output /workspace/sipros/output/peak_example_120 \
  --jobs 4
```

也支持 `--input sample.mzML`。不支持直接读取 vendor `.raw`，需先用现有流程转换。
**配置必须使用该样本搜索时的配置**，模板仅用于展示参数位置；质量容差、酶切、PTM 和同位素设置影响候选和理论峰。
`--fasta` 显式覆盖配置中的数据库路径。所有输入路径必须在容器内可见。
默认 compose 只将项目目录挂载到 `/workspace/sipros`；项目外数据需要另加 bind mount。

`--scan-id` 是文件中的 scan 编号，不是第几行。不指定时选择预处理遇到的第一张原始峰数达到最低门槛的 scan；
它可能没有候选，因此更建议指定已知参与 Regular 搜索的 scan。

`--jobs` 只控制编译并行度。提取运行固定为单线程，防止选择和文件写入竞争。
输出目录必须不存在，防止覆盖已有实验数据。每次运行独立编译，日志分别为 `build.log`、`run.log`。

## 输出

| 文件 | 内容 |
|---|---|
| `observed_unsorted.tsv` | 一张 scan 在 `sortPeakList()` 前的 m/z、intensity、charge、输入索引 |
| `theoretical_unsorted.tsv` | 同一 scan 第一个实际进入 MVH 评分的候选肽段，按生成顺序保存所有理论 m/z |
| `observed_mvh.tsv` | 该 scan 经 MVH 筛选后的实验峰及 class，按 m/z 排序，便于对照真实匹配对象 |
| `candidate.tsv` | scan、候选电荷、数据库肽段、neutral-loss 处理后序列、当前蛋白来源、肽段质量、碎片容差 |
| `run.json` | 输入命令和选择规则 |

索引从 0 开始。“未排序”指**未经过 Sipros 的排序**；如果 FT2 文件或 mzML 读取器已经按 m/z 提供数据，
实验峰可能本来就是有序的。工具不会故意打乱它，也无法恢复上游转换前的仪器顺序。

理论数组只含 m/z，没有预测 intensity 或离子类型标签。`fragment_index` 保留实际生成顺序，
`in_spectrum_range=0` 的离子也保留。预测电荷取自该候选对应的 precursor entry。

这是质量窗口筛选后的**第一个待评分候选**，不是最佳匹配、真实肽段或经 FDR 验证的鉴定。
`protein_source` 只是当前枚举来源，不是所有包含该肽段的蛋白集合。
工具在理论数组生成成功后主动退出，不计算这对数据的 MVH 分数，不执行后续 WDP/Xcorr，不生成完整搜索结果。
退出时无需担心缓冲数据：四个 TSV 均显式关闭后才退出。

## 如何检查和取回数据

```bash
head -n 6 output/peak_example_120/observed_unsorted.tsv
head -n 6 output/peak_example_120/theoretical_unsorted.tsv
cat output/peak_example_120/candidate.tsv
wc -l output/peak_example_120/*.tsv
```

默认 compose 将输出保存在服务器宿主机项目下的 `output/peak_example_120/`，退出 container 不会丢失。
可以在自己电脑执行（替换用户名、地址、宿主机实际路径）：

```bash
scp 'user@server:/server/project/sipros/output/peak_example_120/*.tsv' ./
```

如果原有 container 没有挂载输出目录，从服务器宿主机使用：

```bash
docker cp CONTAINER_NAME:/path/to/peak_example_120 ./peak_example_120
```

## 失败和规模说明

- 编译依赖：Linux C++17 编译器、CMake、OpenMP；Python 只用标准库。项目 Dockerfile 已包含所需软件。
- 没有目标 scan、scan 被过滤、没有质量匹配候选：运行非零退出；可能只生成实验峰文件，请查看 `run.log` 并换 scan。
- 一开始仍读取整个谱图文件并预处理全部 scan，以保留原流程的全局 m/z 范围；不是只读取一个 scan 的微型程序。
- 数据库按原流程遍历。独立副本把批次从 2,000,000 改为 1,000 个已分配肽段，减少导出首个候选前的等待和内存。
- 配置、质量窗口、实验峰预处理、理论离子公式不变。改变批次大小用于提取，不用于性能基准或完整搜索结果对照。
- 脚本对插入位置做唯一性检查，源码版本变化时可能报 `Source version mismatch`，不会静默插错位置。
- 本地没有完整 Linux 构建环境和真实输入时，只能验证副本生成与导出逻辑；服务器首次运行需以构建日志和四个 TSV 为准。

## 数据对应关系

FASTA 蛋白 → 酶切肽段（含 PTM 变体）→ precursor mass 匹配 scan → 根据肽段和电荷生成理论碎片 →
与该 scan 经筛选的实验峰匹配 → 得到 peptide–spectrum 的 MVH 分数 → 肽段通过数据库来源关联回蛋白。

实验 scan 不是一个蛋白的所有峰。它可能主要来自一个肽段，也可能包含共分离肽段/噪声，DIA 中尤其可能是混合谱。
MVH 在当前代码中不直接给整条蛋白与整张实验谱计算分数。

## 2026-09-09 修复

复制规则不再按目录名排除 `lib`，以保留 `MSToolkit/src/expat-2.2.9/lib` 中的 Expat 源码；
仍排除 `.git`、对象文件和静态库。提前退出钩子锚定原 `startProcessingMvh()` 搜索后的
后处理注释，避免新增 MVH-only 入口造成多处匹配。已有输出及 `plot_export.py` 不需要替换。
完整 MVH 搜索基准请使用 [独立入口 A](../mvh_search/README.md)。
