# MVH CPU 基线与 RT 接入位置

## 逐候选 RT 对照（2026-09-22）

已接入可选的 `MVH_ENABLE_RT=ON` 路径：预处理后构建各 scan 的 GAS，在每次 Regular 理论离子生成后调用 RT，对照 double 线性最近峰索引和原桶查询类别。CPU 匹配、MVH 评分与候选保留仍负责正式输出。

`mvh/src/mvh_scoring.cpp` 是原 `MVH.cpp` 的工作副本，仅增加 observer include 和回调；`original/` 保持原样。`mvh/include/rt_match_observer.h` 声明回调和清理作用域；`mvh/app/runner.cpp` 负责资源映射、串行 GPU 调用、错误捕获和统计。OpenMP 工作线程通过 mutex 串行提交 RT，异常保存后在搜索退出并行区域后重新抛出，不会静默忽略失败。若发现差异，不输出成功 PSM 结果。

验证通过：4 项 CTest，包括评分副本改动范围检查、单 scan 与四份复制 scan 的 1/4 线程一致性及实际 observer 计数。单 scan 的 CPU 基线、RT 对照版 1/4 线程输出逐字节一致，保存于 `runs/observer_validation_4_bp1ykq/report.json`。单样本实际评分候选为 1 个，有效理论离子为 12 个；单独 FT2 的上界为 779.359802，生成的 y7=808.3869334668799 被原规则排除。此前完整 FT2 导出的 13 离子快照使用不同全局范围，不能混用有效离子计数。

此版本尚未替换 CPU 匹配，未执行完整数据库 RT 验证，也不是性能版本。当前搜索时间包含双重匹配和 CPU 线性对照。启动时需要沿用已有进程级 OptiX runtime 路径：

```bash
cd /workspace/sipros
cmake -S MVH_RT/mvh -B build/mvh_rt/integration \
  -DCMAKE_BUILD_TYPE=Release -DMVH_ENABLE_RT=ON \
  -DOPTIX_ROOT=/workspace/sipros/build/mvh_rt/optix_sdk
cmake --build build/mvh_rt/integration --target sipros_mvh -j 4
LD_LIBRARY_PATH=/workspace/sipros/build/mvh_rt/optix_runtime:/usr/local/cuda/lib64 \
  ctest --test-dir build/mvh_rt/integration --output-on-failure --no-tests=error
```

## 初始 CPU 基线复制记录

`mvh/` 是从仓库根目录的 `mvh/` 当前工作区复制的独立 CPU 搜索模块，包括现有 `app/runner.cpp` 修改。原目录未改动。当前模块已演进为独立 `MvhScanVector`；它不是历史 A/B/C 备份，也没有预处理快照加载功能。

首次复制时算法和入口均保持原样，仅调整 `CMakeLists.txt` 和 `tests/verify_original.py` 的仓库根目录路径；后续接入改动见上方记录。初始逐文件来源哈希见 `cpu_baseline_copy.json`。副本内 README 保留原始来源说明，其旧构建路径请改用下列命令。

在现有容器中独立构建副本：

```bash
cd /workspace/sipros
cmake -S MVH_RT/mvh -B build/mvh_rt/cpu_baseline -DCMAKE_BUILD_TYPE=Release
cmake --build build/mvh_rt/cpu_baseline -j 4
ctest --test-dir build/mvh_rt/cpu_baseline --output-on-failure --no-tests=error
```

默认关闭 `MVH_ENABLE_RT` 时副本仅运行 CPU；开启后的当前行为见上方对照模式说明。现有 `optix_example/` 同时提供独立示例与可链接的 RT 支持库。顶层 `MVH_RT/CMakeLists.txt` 中的数据导出器仍链接仓库原 `mvh/`，以保留独立的参考数据来源。没有将两个 CPU 模块同时添加到同一个 CMake 构建树。

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
