# MVH_RT：数据样本与独立 OptiX 入门

这里包含两项互相独立的内容：真实 MVH 中间数据，以及用于学习 ray tracing 的两个球体示例。OptiX 例子不读取实验峰，不链接 Sipros，也不声称已经实现峰匹配。

**当前验证状态（2026-09-21）：数据导出与重放通过；OptiX 教学示例已在现有 container 编译并以文件输出模式运行，76,800 条射线全部通过 CPU 几何对照。复用了主机已有 SDK/runtime 的构建目录副本，未安装系统依赖、重启容器或替换 WSL 驱动文件。**

## 先看哪几个文件

| 文件 | 用途 |
|---|---|
| `test_data/experimental_peaks.tsv` | 101 个已分类、尚未建整数桶的实验峰 |
| `test_data/theoretical_ions.tsv` | 对应肽段的 13 个原算法理论离子 |
| `test_data/scan_state.json` | scan 身份、边界、容差、类别计数等原算法状态 |
| `test_data/peptide_state.json` | 肽段、前体电荷、理论峰生成模式 |
| `optix_example/scene.h` | primitive、每条 ray 的结果、launch 参数布局 |
| `optix_example/device_programs.cu` | ray generation、精确求交、closest-hit、miss |
| `optix_example/main.cpp` | 场景、AABB、OptiX BVH/GAS 构建、pipeline、SBT、launch、结果验证 |
| `LEARNING.md` | 按顺序理解代码和进行修改的教程 |

## 数据来自哪里

- 原始文件：`test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2`。
- 配置：`experiments/Regular.cfg`，保留副本 `test_data/search.cfg`。
- scan：1004（索引 0），原始峰 111 个，保留峰 101 个。
- 对应历史 PSM：`[LDNM~ATK]`，原始序列 `[LDNMATK]`，蛋白 `sp|P69776|LPP_ECOLI`，前体电荷 2。
- 原函数生成 13 个理论离子；8 个匹配；`mvhKey = [6, 1, 1, 5]`。
- 原始 C++ 评分：`58.564677834510803`，与历史 PSM double 值精确相同。

导出器链接未修改的 `mvh/` CPU 实现，执行：

```text
读取完整 FT2（建立与原运行相同的全局 m/z 边界）
  → 定位 scan 1004
  → 原 sortPeakList()
  → 原 MVH::Preprocess()
  → 保存 experimental_peaks.tsv + scan_state.json
     此时 scan->pPeakList == nullptr，尚未构造整数桶

历史 PSM 对应的 peptide
  → 原 Peptide::preprocessingMVH()
  → 原 MVH::CalculateSequenceIons()
  → 保存 theoretical_ions.tsv + peptide_state.json

以上快照保存完成后，才构造 PeakList 并调用原评分函数验证
```

没有重跑数据库搜索来重新挑选 PSM，也没有使用 Python 重新估算理论质量。历史 PSM 和原函数重算结果吻合是这个样本的对应关系证据。

## 快照字段与注意事项

`experimental_peaks.tsv` 的列是 `peak_index, mz, intensity_class`。这正对应分类结果 `map<m/z, class>` 的内容，已经按 m/z 有序；**没有 `pMassHub` 桶索引**。序号只是导出数组的下标，不是桶号。

原分类结果不保存每个峰的强度数值，只保存类别。完整原始 m/z、强度和峰电荷另外放在 `raw_peaks.tsv`，原始 scan 文本在 `raw_scan.ft2.txt`，不要把原始数组下标与筛选后下标直接对应。

`scan_state.json` 中包含：

- `mzLowerBound`、`mzUpperBound`：来自整份文件，为 `84.03363 .. 1999.964111`；不能仅用这一张 scan 重算替换。
- `fragment_tolerance = 0.01`、`minimum_matched_fragments = 5`、`bSkip`。
- `intenClassCounts = [14, 29, 58, 95696]`。
- `totalPeakBins = 95797`：MVH 概率模型的离散空间数量，**不是整数桶索引**。
- `preprocessing_parent_neutral_mass = 1612.872994136`：原读谱逻辑对同 scan 多个 precursor 假设取最大质量，用于预处理筛选。
- `matched_precursor_neutral_mass = 807.379489068`：所选 PSM 使用的那个 precursor 假设。它与上一个字段不同是原逻辑，不是导出错误。

`theoretical_ions.tsv` 保留原生成顺序 `b1,y1,b2,y2,...,b6,y6,y7`，未按 m/z 排序。这份 charge-2 样本的碎片电荷均为 1，没有理论强度分类。离子标签只针对当前样本生成模式添加；数值来自原函数。

辅助验证文件：

- `selected_psm.tsv`：从保留的历史 PSM 文件选出的原始一行。
- `reference_matches.tsv`：原桶查询与线性查询交叉检查后保存的匹配结果，`-1` 表示未匹配。
- `ln_factorial_table.bin`：验证阶段保存的原始对数阶乘表，little-endian float64，索引 `0..totalPeakBins`。它是复现评分的辅助输入，不是桶数据，也不是声称在上述捕获时刻已经存在的 scan 成员。
- `validation.json`：原 C++ 验证结果。
- `provenance.json`：输入、配置、原源码与导出程序的 SHA-256。

只想研究匹配时，加载两个主要 TSV，再读取 `scan_state.json` 的容差与边界即可。需要 MVH 评分时再加载类别计数和对数阶乘表。

## 数据重放与重新导出

以下命令在现有 container 中运行：

```bash
cd /workspace/sipros
python3 -B MVH_RT/scripts/check_data.py
```

该脚本仅用标准库，不创建整数桶，直接遍历实验峰，保留严格 `< tolerance` 与等误差先遇到的峰，验证匹配、类别计数和分数。

重新导出到一个不存在的目录：

```bash
cmake -S MVH_RT -B build/mvh_rt -DCMAKE_BUILD_TYPE=Release
cmake --build build/mvh_rt --target export_mvh_sample -j 4
python3 -B MVH_RT/scripts/export_data.py --output MVH_RT/test_data_second_export
ctest --test-dir build/mvh_rt -R '^mvh_rt_snapshot_replay$' --output-on-failure
```

不要用单张 `raw_scan.ft2.txt` 代替完整输入进行重新预处理，否则全局 m/z 边界会改变。该导出脚本专门选择本次样本，不是任意电荷/肽段的通用导出工具。

## OptiX 构建与运行

本轮实际使用：现有 container 的 **CUDA 12.8**、已有 **OptiX 9.0.0** 头文件、用户提供的 **Linux 610.57.04** OptiX runtime 副本。无需升级 container 到主机的 CUDA 13.3 即已验证成功。

检查时主机 `/usr/lib/wsl/lib/libnvoptix.so.1` 是约 14 KB 的 shim，而不是记录中的 48 MB runtime；container 也没有挂载 `/opt/optix` 或 `/usr/lib/wsl/lib`。因此使用以下已存在的文件作为隔离构建/运行输入：

| 原文件位置（主机） | 现有项目挂载内的副本 |
|---|---|
| `/home/ams098z/opt/optix/include` | `build/mvh_rt/optix_sdk/include` |
| `/home/ams098z/packages-DL/NVIDIA-Linux-x86_64-610.57.04/` 下的四个 runtime 文件 | `build/mvh_rt/optix_runtime/` |

四个文件是 `libnvoptix.so.610.57.04`、`libnvidia-rtcore.so.610.57.04`、`libnvidia-gpucomp.so.610.57.04` 和 `nvoptix.bin`，并在副本目录创建 `libnvoptix.so.1` 符号链接。**没有复制 Linux 安装包中的 libcuda，也没有改写 WSL 原生 libcuda。** 副本在被 Git 忽略的 build 目录，不随教学源码提交。

在现有 container 中运行：

```bash
cd /workspace/sipros
bash MVH_RT/scripts/run_optix.sh MVH_RT/runs/spheres_02
```

输出目录必须不存在。脚本只为启动的进程设置 runtime 路径，不改变调用者环境；若发现 CUDA stubs 路径则报错。本例无图形窗口，不使用 WSLg/GLFW。

修改教学源码后重建：

```bash
cmake -S MVH_RT/optix_example -B build/mvh_rt/optix_example \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPTIX_ROOT=/workspace/sipros/build/mvh_rt/optix_sdk
cmake --build build/mvh_rt/optix_example -j 4
```

独立构建不编译任何 MVH 代码。若未来容器已按用户建议挂载 SDK，可以直接把 `OPTIX_ROOT` 改成 `/opt/optix`；仍须确认运行进程能找到完整 runtime，不能只看 `nvidia-smi`。

这次成功运行输出：

```text
Built GAS: primitives=2 bytes=1152
PASS: CPU geometry reference; hits=19640 misses=57160
```

实际结果在 `runs/spheres_01/spheres.ppm` 和 `runs/spheres_01/rays.tsv`。前者是 320×240 图像，后者每像素一行，记录 primitive ID 与 t。未命中 ID 为 -1，t 留空。

`build/` 清理后这些 SDK/runtime 副本会消失；应重新使用上述已有文件或已配置挂载，不要自动下载驱动或更改 `/usr/lib/wsl/lib`。运行依赖的源文件与副本哈希见 `validation/optix_runtime_provenance.json`。

## 验证记录与官方资料

- `validation/data_tests.txt`：三个已执行测试通过：快照重放、原源码身份、提取方法身份。
- `validation/optix_configure.txt`、`optix_build.txt`：复用 SDK 后的成功配置和编译。
- `validation/optix_environment.json`：首次检查的缺依赖状态；`optix_environment_staged.json`：使用副本后的状态。
- `validation/optix_run.txt`：首次实际 GPU 运行和 CPU 几何对照通过的日志。
- `validation/optix_script_run.txt`：2026-09-21 容器重新启动后，运行脚本复现通过。
- `validation/optix_loaded_libraries.txt`：实际动态库加载路径；CUDA 经容器已有路径加载 WSL 驱动，OptiX/rtcore/gpucomp 来自隔离副本。
- `validation/status.json`：最终验证状态。
- [NVIDIA OptiX 官方编程指南](https://raytracing-docs.nvidia.com/optix9/guide/index.html)
- [NVIDIA 官方最小头文件仓库](https://github.com/NVIDIA/optix-dev)
- [OptiX 9.0 API 声明](https://github.com/NVIDIA/optix-dev/tree/v9.0.0/include)
