# 第一轮主机开销优化

本轮不修改酶切、质量窗口、理论离子公式、匹配条件、评分公式或 GPU top 排序算法。CPU 基准 `mvh/` 和 `original/` 依赖保持不变。优化前源码与可执行文件分别保存在：

- `output/mvh_cuda_optimization_01/before_source.tar.gz`
- `output/mvh_cuda_optimization_01/sipros_mvh_cuda_before`

## 为什么可以减少 CPU 恢复工作

GPU 已经按原候选顺序判断是否合并、评分是否成功，并维护 top 列表。原来的主机恢复阶段又为每个候选运行一次字符串 top 查询，产生大量重复工作。

当前正常路径按设备返回的状态处理：

| 状态 | CPU 操作 | 保留的语义 |
|---|---|---|
| 已合并 | 原 `mergePeptide()`，并检查它确实返回 true | 保留蛋白名称追加和去重规则 |
| 评分成功 | 原 `saveScore()` | 保留 top 50、替换和同分排序规则 |
| 评分不足 | 不修改 top | 原流程在这种情况下也没有结果状态变化 |
| 设备错误 | 报错停止 | 不静默忽略或回退 |

恢复仍逐 scan、逐候选保持原顺序，不能只读取 GPU 最终 top 并创建对象：中途发生的蛋白名称合并，以及候选离开 top 后再次进入的行为，都与顺序有关。

正常模式在每批结束检查 GPU 与主机保留候选的序列 ID、分数及顺序。`--verify-cuda` 保留逐候选的原 CPU merge 查询和每次实际评分复算，用于独立验证正常路径依赖的 GPU 决策。该开关不是性能优化开关；性能对照双方均不启用它。

## 可读性与文件变化

- `cuda/engine.cu`：`PackedScoringBatch` 明确拥有批次数组；`packScoringBatch()` 预分配容量；`executeScoringBatch()` 负责传输/执行；`restoreScoringResults()` 恢复状态。原 `scorePeptidesMVH()` 名称保留，成为简洁的步骤组织函数。
- `cuda/engine.cu`：增加接受 `const char *` 的检查函数重载，正常检查通过时不创建临时字符串；带 scan/candidate 信息的错误文本仅在失败时构造。
- `cuda/types.cuh`：增加有名称的 `ResultStatus` 和管理 CUDA event 生命周期的 `CudaEvent`。
- `cuda/scoring.cuh`：仅将返回状态数字替换为枚举名；设备评分与候选处理顺序不变。
- `tests/check_run.py`：增加正常模式与完整验证模式的输出一致性检查。
- `tests/validate.py`：整理成职责清楚的函数，支持保存的旧 CUDA 二进制；轮换各版本执行顺序，保存每批 profiling 数据。

关键语义位置采用英文注释。没有为了打包增加原 `Peptide` 类字段，也没有改变原有函数名称。

## 计时解释

`kernel_seconds` 使用 CUDA events，测量设备上的评分 kernel。分配/上传、下载、主机打包、结果恢复分别记录。`gpu_service_seconds` 是较大的包含关系，不应再加上其中的 kernel/传输子项来计算总时间。完整搜索时间仍以 `run_summary.tsv` 的 `search_seconds` 为准。

本轮运行记录集中于 `output/mvh_cuda_optimization_01/`。真实数据验证和性能结果见下方完成记录。

## 完成记录（2026-09-20）

全部编译、测试和性能运行在原有 container 中完成，没有安装依赖或改变容器配置。

- 4 项 CTest 全部通过，包含 CPU 原文件身份、转换范围、CLI/样本、CUDA 语义边界测试。
- CUDA compute-sanitizer：0 errors、0 bytes leaked（语义测试程序）。
- 真实数据 `--verify-cuda`：37,899,128 次实际评分尝试通过原 CPU 复算校验，1,403,362 条保留 PSM 与 CPU 输出逐字节相同。
- 性能比较：三个版本各运行 3 次，轮换顺序，共 9 次搜索，全部 PSM 完全一致。性能测试不启用 `--verify-cuda`，运行期间没有同时执行本任务的编译或测试。

| 版本 | 第 1 次搜索/秒 | 第 2 次搜索/秒 | 第 3 次搜索/秒 | 平均/秒 |
|---|---:|---:|---:|---:|
| CPU 4 线程 | 15.4499 | 15.9630 | 15.1609 | 15.5246 |
| 优化前 CUDA | 29.2646 | 30.0536 | 30.3882 | 29.9021 |
| 优化后 CUDA | 18.3179 | 18.9016 | 18.3241 | 18.5146 |

完整搜索时间较旧 CUDA 减少约 **38.1%**，约为 **1.62 倍加速**；仍比当前 CPU 4 线程慢约 **19.3%**。这是当前数据与设备上的三次观察，不能外推为所有数据库/谱图的加速比。

两批评分阶段合计、每次运行的平均分段时间：

| 环节 | 优化前/秒 | 优化后/秒 |
|---|---:|---:|
| 主机打包 | 5.1436 | 4.1613 |
| GPU 服务（分配/传输/计算/释放等） | 6.1596 | 6.1377 |
| 主机结果恢复/检查 | 11.9147 | 2.3608 |

新增 CUDA-event kernel 测量平均为 5.5206 秒。不同计时源和边界不保证子项严格相加等于主机 wall time；GPU 服务与其子项也不能重复累计。此次改善主要来自减少主机重复工作，不能解读为 GPU 评分 kernel 本身获得 1.62 倍加速。

PSM SHA-256（与历史基准一致）：

```text
2f96a76b931b89c64054026533e4de5379fca959848a0a314c8ea42332f31520
```

报告与日志：

- `output/mvh_cuda_optimization_01/ctest.log`
- `output/mvh_cuda_optimization_01/memcheck.log`
- `output/mvh_cuda_optimization_01/verified/report.json`
- `output/mvh_cuda_optimization_01/performance/report.json`
- `output/mvh_cuda_optimization_01/performance/comparison.tsv`

后续尚可研究打包中的候选映射以及 GPU 评分工作分配。本轮没有继续引入新的 kernel 算法或理论峰缓存，以便明确归因本轮收益。
