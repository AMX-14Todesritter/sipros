# C：保持搜索语义的 CUDA 移植基准

本目录由 `mvh/` 复制而来。CPU 基准仍在 `mvh/`，没有修改；复制时的文件哈希见 `CPU_BASELINE.json`。这里是独立构建目标，复用仓库中的 MSToolkit，不能单独脱离仓库构建。旧 CPU 文档保存在 `cpu_provenance/`，不代表本目录的 CUDA 实现。

目标是将原有四处 OpenMP **线程并行循环**移到 GPU，保留原搜索顺序和评分规则。并非整个程序都在 GPU 上执行，也没有增加理论谱缓存、质量分组或新搜索算法。

## 按这个顺序阅读

| 位置 | 作用 |
|---|---|
| `app/main.cpp`、`app/runner.cpp` | 参数、配置、读谱、预处理、搜索、导出和计时 |
| `include/mvh_scan_vector.h` | 与 CPU 基准相同的提取类和函数名称 |
| `src/database_search.cpp` | 原 `searchDatabaseMvh()`、酶切/PTM/质量分配；批次处理接入 CUDA |
| `src/spectrum_input.cpp` | 原读谱和 precursor 索引；预处理入口接入 CUDA |
| `cuda/engine.cu` | CPU 对象展开为数组、传输、启动 kernel、回收结果和一致性检查 |
| `cuda/preprocess.cuh` | 实验峰预处理、强度求和、肽段预处理的设备实现 |
| `cuda/scoring.cuh` | 理论离子、`findNear()`、MVH 概率评分、top 候选的设备实现 |
| `cuda/types.cuh` | 设备数组结构、显式容量和 CUDA 内存管理 |
| `original/` | 未修改的原实现，提供依赖和验证用 CPU 参考 |

函数名称保留；同名的 CUDA 实现在 `mvh_cuda` 命名空间中。设备不能直接使用原对象内的 STL 容器和主机指针，因此内存表示和执行载体进行了改写。

## 哪些工作移到了 GPU

| 原 OpenMP 循环 | CUDA 工作分配 | 保留的计算 |
|---|---|---|
| `preProcessAllMs2Mvh()` 的 scan 预处理 | 每个线程一个 scan | 排序、TIC/峰数筛选、失水峰处理、强度分类 |
| 同函数的 `sumIntensity()` 循环 | 每个线程一个 scan | 强度总和与最大强度 |
| `processPeptideArrayMvh()` 的 `preprocessingMVH()` 循环 | 每个线程一个 peptide | 长度统计与中性丢失字符串处理 |
| 同函数的 `scorePeptidesMVH()` 循环 | 每个线程一个 scan，内部按原顺序遍历候选 | 理论离子、匹配、概率评分、merge/top 决策 |

CPU 仍负责读文件、配置、FASTA 酶切/PTM、precursor 质量查询/关联、原始对数阶乘表、数组打包以及 STL 结果恢复。GPU 决策返回后，CPU 使用原 `mergePeptide()`/`saveScore()` 按原顺序恢复蛋白名称和结果对象，正常模式只对 GPU 标记的合并调用 `mergePeptide()`，对评分成功的候选调用 `saveScore()`，并检查最终 top 顺序；未成功且未合并的候选没有状态副作用，不再重新扫描 CPU top 列表。`--verify-cuda` 仍逐候选复核所有 merge 决策并复算原 CPU 评分。

原依赖 `isotopologue.cpp` 中两处 `omp simd` 未改动；仍链接 OpenMP 以支持原计时/依赖。移植替换的是上述四处线程并行循环，不能称为完全消除所有 OpenMP 指令。

每个 block 固定 128 个线程，由 CUDA 调度到硬件。`-t` 仅为兼容 CPU 命令行而接受，CUDA 版本把主机 OpenMP 线程数设为 1；它不设置 GPU core 数。

## 构建与运行（现有 container 内）

```bash
cd /workspace/sipros
cmake -S mvh_cuda -B build/mvh_cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/mvh_cuda -j 4
ctest --test-dir build/mvh_cuda --output-on-failure

build/mvh_cuda/bin/sipros_mvh_cuda \
  -f test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2 \
  -c experiments/Regular.cfg -fasta raw/Ecoli.fasta \
  -o output/mvh_cuda_my_run
```

输出目录必须不存在。输出 `mvh_psms.tsv`、`run_summary.tsv` 和配置副本。加 `--verify-cuda` 可逐项复算原 CPU 预处理和每次实际评分，发现差异立即失败；该模式用于正确性检查，不用于性能比较。

运行 CPU/GPU 成对测试：

```bash
python3 -B mvh_cuda/tests/validate.py --real --cpu-threads 4 --repeats 2 \
  --output output/mvh_cuda_my_comparison
```

不加 `--real` 则使用小样本；需要 CPU 二进制 `build/mvh/bin/sipros_mvh`。脚本保存命令、输入/二进制/PSM 哈希、原始日志、汇总 JSON/TSV。输出全部一致后，默认保存一份压缩 PSM 并删除重复 TSV；加 `--keep-psms` 保留全部。

## 语义与适用范围

- `Search_Type = Regular`；不执行后续 WDP/Xcorr、Percolator 或蛋白推断。
- 批次大小仍为 2,000,000；保留完整数据库枚举和候选分配。
- 匹配使用严格 `abs(error) < tolerance`；等误差保留先遇到的峰；允许多个理论离子命中同一实验峰。
- 保留原候选顺序、只在已有 top 内合并重复序列的行为、top 50 和同分排序行为。初始化时用 1,000 组样本检查设备排序是否与实际主机 `std::sort` 一致。
- GPU 使用原 CPU 生成的对数阶乘表。针对本容器 GCC 的 `-ffast-math` 数值行为显式匹配了高电荷分支的倒数乘法和 FMA；不能据此宣称任意编译器/平台都保证逐位一致。更换环境后应重新执行验证。
- 显式设备容量：最多 128 个残基、512 字节字符串缓冲、8 个强度类别。超过容量或 CUDA 出错会报错，不截断，也不静默回退 CPU。中性丢失扩展规则还受保守缓冲容量检查限制。
- 原始强度总和因 CPU 编译器重排允许相对 `1e-12` 校验误差；该总和不参与 MVH 评分。峰、类别、桶索引及成功评分要求精确一致。
- 当前在移植基准上减少了主机重复工作并预分配打包数组。暂未做理论峰缓存复用、线程间协作、传输重叠或细粒度负载均衡。优化前源码和二进制保存在 `output/mvh_cuda_optimization_01/`。

## 计时口径

`run_summary.tsv` 分开记录加载、预处理、完整搜索和导出。搜索不包含前置实验峰预处理和 TSV 导出。

日志中 `CUDA scoring` 的 `pack_seconds` 为主机打包，`gpu_service_seconds` 包括设备分配、传输、kernel 和同步，**不是纯 kernel 时间**；`replay_verify_seconds` 包含主机结果恢复/检查，验证模式还包含原 CPU 评分复算。原 CPU 内部 profiling 的累计线程时间不能直接当作 wall time 相加比较。验证结果见 `VALIDATION.md`。


## 本轮优化的阅读顺序

`cuda/engine.cu` 的 `scorePeptidesMVH()` 现在只负责组织三个步骤：

1. `packScoringBatch()`：按原候选顺序打包，提前统计容量并预分配数组。
2. `executeScoringBatch()`：设备内存分配、上传、kernel、下载；使用 CUDA events 测量 kernel。
3. `restoreScoringResults()`：按原顺序恢复必要的状态变化；验证模式保留全量 CPU 复核。

关键注释使用英文，解释为什么不能重排候选、为什么不能只从最终 top 重建蛋白名称，以及哪些重复检查可从正常路径移除。`ResultStatus` 用明确名称代替设备/主机共享的状态数字。

新增 `allocation_upload_seconds`（分配与上传的主机耗时）、`kernel_seconds`（CUDA events 记录的设备 kernel 时间）、`download_seconds`（下载及接收数组分配的主机耗时）。这些是 `gpu_service_seconds` 的子项，不要重复相加；服务总时间还包含释放资源等开销。

可以额外比较保存的优化前 CUDA 二进制：

```bash
python3 -B mvh_cuda/tests/validate.py --real --cpu-threads 4 --repeats 3 \
  --cuda-baseline-binary output/mvh_cuda_optimization_01/sipros_mvh_cuda_before \
  --output output/mvh_cuda_my_optimization_comparison
```

脚本依次运行 CPU、优化前 CUDA、当前 CUDA，并在每轮轮换执行顺序。性能实验期间不要同时编译或执行其他 GPU/CPU 测试。详细优化记录见 `OPTIMIZATION.md`。
