# MVH CPU 基线与 RT 接入位置

## 当前状态

`mvh/` 是从仓库根目录的 `mvh/` 当前工作区复制的独立 CPU 搜索模块，包括现有 `app/runner.cpp` 修改。原目录未改动。当前模块已演进为独立 `MvhScanVector`；它不是历史 A/B/C 备份，也没有预处理快照加载功能。

副本中的算法和入口均保持原样，仅调整 `CMakeLists.txt` 和 `tests/verify_original.py` 的仓库根目录路径。逐文件来源哈希见 `cpu_baseline_copy.json`。副本内 README 保留原始来源说明，其旧构建路径请改用下列命令。

在现有容器中独立构建副本：

```bash
cd /workspace/sipros
cmake -S MVH_RT/mvh -B build/mvh_rt/cpu_baseline -DCMAKE_BUILD_TYPE=Release
cmake --build build/mvh_rt/cpu_baseline -j 4
ctest --test-dir build/mvh_rt/cpu_baseline --output-on-failure --no-tests=error
```

该副本仍只运行 CPU。现有 `optix_example/` 仍是独立峰匹配原型。顶层 `MVH_RT/CMakeLists.txt` 中的数据导出器仍链接仓库原 `mvh/`，以保留独立的参考数据来源。本轮没有将两个 CPU 模块同时添加到同一个 CMake 构建树。

## 建议接入顺序

1. **预处理后构建每张 scan 的 GAS。** 在 `mvh/app/runner.cpp` 的 `preProcessAllMs2Mvh()` 之后、`searchDatabaseMvh()` 之前创建 RT 上下文、共享 pipeline 和每 scan 资源。跳过 `bSkip` 或空峰列表。此时 `peakData` 已销毁，读取 `scan->pPeakList->pPeaks` 和 `pClasses`；以 scan 数组下标标识资源，不仅依赖可能重复的 `iScanId`。GAS 输出内存在所有候选批次结束前保持有效。重用构建 scratch 前必须确认对应构建已完成。
2. **抽出可调用匹配服务。** 将 `optix_example/main.cpp` 中的资源管理、GAS 构建和 launch 从演示入口抽出。接口输入为 scan 索引和理论 m/z 数组，输出峰索引/类别。测试文件读取和 TSV 输出留在演示入口。保留 CPU double 峰数组作为正确性基准。
3. **先替换匹配，保留 CPU 评分。** 以 `mvh/original/src/MVH.cpp::ScoreSequenceVsSpectrum()` 为语义边界，保留原理论峰生成、有效 m/z 范围、`mvhKey` 及概率公式，用 RT 返回结果替换 `findNear()` 循环。不要逐离子 launch；先用完整离子数组建立正确性路径，再考虑跨候选批量。
4. **保持候选顺序。** `mvh/src/database_search.cpp::processPeptideArrayMvh()` 按 scan 调用 `MS2Scan::scorePeptidesMVH()`，后者依次执行 `mergePeptide()`、评分、`saveScore()`。RT 批量化不能改变合并与 top 候选更新的顺序。当前 OpenMP 会并行处理 scan，需明确 GPU 调度及每次 launch 的参数、结果缓冲区所有权，不能让多个线程共享并覆盖同一组缓冲区。
5. **匹配验证后再迁移评分。** 新 CUDA kernel 按 peptide–scan 对归约分类计数并应用原 lnCombin 公式。携带每组 offset/count、scan 索引、类别容量、总 bins、全局 m/z 范围、最小匹配数和对数阶乘表。保持“评分无效”与有效分数分开，先不改候选保留策略。

## 接入前需要解决的正确性问题

- 当前三角形坐标和射线原点均为 float；原匹配使用 double。直接最邻近查询不保证在舍入碰撞、近邻或容差边界处等价。只对返回的一个峰做 double 复核，也无法恢复已漏掉的真实候选。需要保守候选覆盖及 double 筛选，或其他可证明等价的坐标方案。
- 原条件为严格 `abs(error) < tolerance`；当前 tracing 区间不能代替这一最终判断。
- 原桶查询等误差保留先遇到的峰；当前双向结果合并等 t 保留右侧，且重合三角形的选择无顺序保证。
- 全局 m/z 范围外的理论峰不进入 MVH 计数；范围内未命中的峰计入未匹配。多个离子允许重复命中同一实验峰。
- 当前单样本通过不代表整库一致，也没有测得端到端 RT 加速。

先以已有小样本核对峰索引、类别和完整分数，再覆盖双向命中、无匹配、完全重合、严格边界、等距离和浮点碰撞。最后比较 CPU 基线与 RT 完整 PSM 输出，并将 GAS 构建、打包、传输、trace、评分、结果恢复分别计时。
