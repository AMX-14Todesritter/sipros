# GPU baseline 的 RT 匹配接入

本目录把现有未偏移三角形算法接入 `mvh_cuda`，不复制整套 CUDA baseline。
默认仍使用 CUDA 桶搜索。CPU 版 `MVH_RT/mvh` 与独立三角形示例继续保留。

## 数据路径

`mvh_cuda/cuda/engine.cu` 继续完成 GPU 候选分配、理论峰缓存和批次缓冲区管理。
启用 RT 时，每批调用一次 OptiX launch，每个 raygen 线程处理一个候选肽段。
它直接读取显存里的理论峰，或者在缓存不可用时使用原 GPU 理论峰生成函数。
每个理论峰仍执行原来的左右两次 trace，保留第二次范围缩短及 float t 比较。

`scoring.cuh::scoreCandidate<Counter>` 为 CUDA 和 OptiX 共用的评分函数，
只替换 Counter 的匹配行为。RT 结果直接写入显存中的 Result 数组，
后续仍使用现有 GPU merge/top 与事件压缩，只回传最终所需事件。
没有逐候选调用 CPU 版 `traceRays()`，也没有逐理论峰回传 CPU。

`bridge.cpp` 复用 `optix_example` 的 OptiX 初始化和 pipeline 创建。
`geometry.cu` 从已上传的 double 峰数组在 GPU 生成顶点或实例。
直接三角形后端为每个可用 scan 构建一次 GAS；实例后端共享一份含标准三角形的 GAS，
每个峰使用 x 平移实例，每个 scan 构建一次 IAS。closest-hit 通过实例 ID 返回 scan 内峰索引。
两个后端均使用一块 AS 输出池和一块可复用 scratch，在同一 stream 顺序构建，最后统一同步。
句柄表、SBT 和 launch 参数缓冲区跨肽段批次复用。
每个数据集预处理时重置资源，退出 run 时在 CUDA 运行时清理前释放。
现有 CUDA baseline 的大批次峰数据上传仍保留，本次没有修改其打包策略。

## 后端

| 参数 | 行为 |
|---|---|
| `--match-backend cuda` | 默认 CUDA 桶搜索 |
| `--match-backend rt-audit` | GPU 同时计算 CUDA/RT 候选结果，只回传差异计数；后续评分保留与输出使用 CUDA 结果 |
| `--match-backend rt-triangle` | 实际使用 RT 候选结果参与 GPU merge/top，输出为实验结果 |
| `--match-backend rt-instanced` | 共享单三角形 GAS，每峰一个实例，每 scan 一个 IAS；输出为实验结果 |

三角形后端仍存在已知的零距离漏命中和 float 边界问题，不是准确等价替代。
`--verify-cuda` 继续检查用于恢复的结果与 CPU 一致：audit 模式检查 CUDA 路径，
triangle/instanced 模式遇到真实评分差异会报错；不会静默切回 CPU 或 CUDA 匹配。
实例变换仍使用 float，不解决零距离或容差精度问题。当前 SDK 的实例结构为 80 字节，
三个 float3 顶点为 36 字节；共享基础几何不代表整体显存或遍历成本更低。

audit 的 differing_results、matched_count_changes、score_changes 和
predicted_count_changes 是 merge/top **之前**的候选级计数，包含之后可能被合并的候选。
它们不是逐离子差异数，也不是最终鉴定准确率。

## 构建与运行

在宿主终端执行，使用已有容器与 SDK：

```bash
docker exec sipros-sipros-1 cmake \
  -S /workspace/sipros/mvh_cuda \
  -B /workspace/sipros/build/mvh_rt/gpu_integration \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DMVH_CUDA_ENABLE_RT=ON \
  -DOPTIX_ROOT=/workspace/sipros/build/mvh_rt/optix_sdk

docker exec sipros-sipros-1 cmake --build \
  /workspace/sipros/build/mvh_rt/gpu_integration -j4

docker exec sipros-sipros-1 env \
  LD_LIBRARY_PATH=/workspace/sipros/build/mvh_rt/optix_runtime:/usr/local/cuda/lib64 \
  /workspace/sipros/build/mvh_rt/gpu_integration/bin/sipros_mvh_cuda \
  -f /workspace/sipros/MVH_RT/runs/real_database_8scans_qrjuzn16/subset.ft2 \
  -c /workspace/sipros/experiments/Regular.cfg \
  -fasta /workspace/sipros/raw/Ecoli.fasta \
  -o /workspace/sipros/MVH_RT/runs/gpu_rt_audit_example \
  --match-backend rt-audit
```

输出目录必须不存在。`MVH_CUDA_ENABLE_RT` 默认 OFF；关闭时不需要 OptiX，
请求 RT 后端会明确报错。运行摘要记录 `match_backend`。

## 验证范围

- RT 构建的 7 项 CTest：原 4 项 CUDA 测试，加两种 GPU RT contract 和集成测试。
- 固定小样本：四个后端、默认和 100 肽段小批次，PSM 完全一致并通过 CPU 对照。
- contract：缓存/直接 GPU 理论峰路径、重复 GAS 查询、已知零距离 miss 和正常非零命中。
- compute-sanitizer memcheck：0 errors。
- 八个真实 scan + 完整 E. coli FASTA：三后端正常退出；audit PSM 与 CUDA 完全一致。
  两批共 2,764 个 speculative candidates，186 个 matched-count 变化、38 个 score 变化，
  predicted-count 变化为 0；实际 triangle PSM 与 CUDA 不同，符合已有精度限制。
  原始记录：`MVH_RT/runs/gpu_bridge_8scan_ukjd12oy/`。

本轮没有重新启动已停止的全量审计，也没有声称性能加速。
`[RT GPU setup]` 单列构建和初始化时间，但它也包含在外层 gpu_service/search 时间中，
不可重复相加。audit 的 kernel_seconds 包含两种评分、比较和相关等待，不能当成纯 RT 时间。
不同验证开关下的运行时间不可直接用于后端性能比较。

## 完整数据性能对照（400 万批次）

GPU 几何生成/输出池修改前，已完成 CPU 四线程、CUDA、GPU-RT 各三轮交替测试。
搜索平均分别为 16.47、8.05、19.37 秒。这是旧实现的结果。
GPU 两个后端指定 400 万生成肽段批次；该批次走直接 GPU 理论峰生成。默认批次参数仍为 200 万。
结果、内存口径和限制见 `MVH_RT/runs/full_three_backend_batch4m_20260922/REPORT.md`。
可用本目录 `benchmark.py --root /workspace/sipros --output NEW_DIRECTORY --batch 4000000` 复现。
可加 `--backends cuda rt-triangle rt-instanced --repeats 1` 比较 GPU 几何和实例后端。

本次 GPU 几何/共享 GAS 的全量单轮结果见
`MVH_RT/runs/gpu_geometry_instancing_20260922/REPORT.md`。
该轮 OptiX context 开启 validation，这些时间含验证开销。
直接三角形与实例后端的完整 PSM 相同，并与此前三角形输出相同。

当前 `rt_support.cpp` 将 `validationMode` 设为 `OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF`，
性能运行不再启用 OptiX 调试验证。需要排错时可改回 `ALL` 并重新编译；该设置与
用于 CPU 结果对照的 `--verify-cuda` 开关相互独立。

关闭 validation 后的全量三轮结果：CUDA、直接三角形、共享 GAS 的平均搜索时间分别为
8.08、14.29、11.57 秒；评分 kernel 分别为 0.909、0.840、0.708 秒。
7 项 CTest 通过，9 次全量运行的输出均与各自修改前一致。
详细计时、显存口径及限制见 `MVH_RT/runs/gpu_validation_off_20260922/REPORT.md`。

## 修改代码后的自动测试入口

在 WSL 宿主的项目根目录执行（容器必须已启动，沿用现有 build 目录和 SDK）：

```bash
# 默认：重编译 CPU/GPU、跑两套 CTest，再串行跑四个后端的小样本。
bash MVH_RT/gpu_bridge/run_benchmark.sh --dataset smoke

# 完整 scan + Marine，每个后端一轮；可能需要一小时以上。
bash MVH_RT/gpu_bridge/run_benchmark.sh --dataset marine --batch 4000000 --repeats 1

# E. coli 完整测试，轮换顺序跑三轮。
bash MVH_RT/gpu_bridge/run_benchmark.sh --dataset ecoli --repeats 3

# 只测选定后端；也支持 --fasta /容器路径、--scans 和 --config。
bash MVH_RT/gpu_bridge/run_benchmark.sh --dataset marine --backends cuda rt-instanced
```

`--skip-build` 和 `--skip-tests` 可显式跳过编译或检查，默认不跳过。
`--output` 为容器内尚不存在的目录；默认自动生成 `output/benchmarks/gpu_bridge/数据集_UTC时间/`。
`SIPROS_CONTAINER` 环境变量可覆盖容器名称。脚本顺序执行，各运行不重叠。
CPU 保留原有批次机制（四线程）；`--batch` 仅控制 GPU 生成肽段的批次。
脚本不会自动改变代码中的 validation 设置，会将设置源码记录在 metadata.json。

输出包括：
- build/test 日志，失败即停止，性能测试不混入 CTest 的时间。
- metadata.json、source_hashes.json：参数、设备信息、源码哈希；容器可访问 Git 时保存 Git 信息与 diff。
- benchmark/report.json：逐批时间、输入/二进制哈希、退出状态与内存统计。
- benchmark/REPORT.md：均值/标准差、构建次数、输出哈希比较。
- 每个后端的日志与 PSM 输出。RT 与 CUDA 的已知结果差异会显示，不会当成脚本执行失败。

设备显存为 NVML 全设备采样值，含其他进程。单轮无法判断时间波动。
已有 report.json 可直接重新生成 Markdown，不需要重跑搜索：

```bash
docker exec sipros-sipros-1 python3 /workspace/sipros/MVH_RT/gpu_bridge/benchmark_report.py \
  /workspace/sipros/MVH_RT/runs/某次运行/benchmark
```

## RT 路径跳过桶索引

普通 `rt-triangle` / `rt-instanced` 不再构建主机 `PeakList` 桶索引，也不打包、分配或上传设备桶。预处理得到的有序峰和强度类别保存在紧凑的 `UnindexedSpectrum` 数组中，与当前数据集的谱图顺序对应；预处理新数据集或退出搜索作用域时释放；RT 继续通过 GPU bridge 使用同一份峰数据构建加速结构。

| 模式 | 主机桶索引 | 设备桶索引 |
|---|---|---|
| `cuda` | 保留 | 保留 |
| `rt-audit` | 保留 | 保留，用于 CUDA 对照 |
| 普通 `rt-triangle` / `rt-instanced` | 不构建 | 不分配、不上传 |
| RT 加 `--verify-cuda` | 保留，用于原 CPU 评分复算 | 不分配、不上传 |
| RT 加 `--score-impact` | 保留，用于 CUDA 分数对照 | 保留，用于 CUDA 分数对照 |

实现位于 `mvh_cuda/cuda/engine.cu`：`needsHostBuckets()` 与 `needsDeviceBuckets()` 决定各侧是否需要索引；`storeProcessedSpectrum()` 管理两种峰数据表示，`unindexedSpectrum()` 检查谱图身份与顺序；批次打包在 CUDA/audit 模式或显式开启 `--score-impact` 时附带桶数据。空的 `Buffer<short>` 不调用 `cudaMalloc` 或 `cudaMemcpy`，传给 RT 的桶指针为 `nullptr`。RT 的匹配实现不解引用该指针。

`totalPeakBins`、类别计数、对数表及峰筛选仍然保留，它们属于 MVH 评分而非查找索引。原始 CPU 源码、CUDA 评分内核、OptiX 几何及匹配规则不变；已有 RT 浮点边界差异也不属于本次修改范围。

日志新增 `host_bucket_entries`（预处理完成后保留的主机桶元素数）和 `device_bucket_entries`（本批打包并上传的桶元素数），单位是 `short` 元素，不是峰数或概率模型的 bins。

`check_integration.py` 覆盖四个后端、普通/CPU 复算、微小批次、空峰谱图，检查结果一致性及实际桶数量。真实性能与修改前后逐字节结果对照见项目 `output/benchmarks/rt_bucket_elision/` 下的运行报告。

本次完整输入验证：20 次最终版/基线运行的同后端 PSM 均逐字节相同；CUDA 两种构建的机器指令完全一致。具体性能、主机峰值内存及统计限制见 [验证报告](../../output/benchmarks/rt_bucket_elision/before_after_20260923T154801_995975Z/confirmation/REPORT.md)。

## 用 MVH 分数和最终 top 候选评估差异

`validate_score_impact.py` 不报告逐 peak 准确率，而是以 CUDA 桶匹配为计算参照，依次执行 CUDA 正常搜索、各 RT 后端正常搜索和显式开启的候选评分诊断。默认使用完整 E. coli 数据：

```bash
# 宿主机执行，使用现有容器；输出自动位于 output/validation/score_impact/ecoli_<时间>/
docker exec sipros-sipros-1 python3 -B MVH_RT/gpu_bridge/validate_score_impact.py --dataset ecoli

# 快速检查；可用 --backends rt-triangle 只检查一个后端
docker exec sipros-sipros-1 python3 -B MVH_RT/gpu_bridge/validate_score_impact.py --dataset smoke

# 使用已完成结果重新生成报告，不再运行数据库搜索（路径为容器内路径）
docker exec sipros-sipros-1 python3 -B MVH_RT/gpu_bridge/validate_score_impact.py \
  --analyze-only /workspace/sipros/output/validation/score_impact/ecoli_<时间>
```

支持 `--scans`、`--config`、`--fasta`、`--binary`、`--output`、`--peptide-batch-size` 和 `--score-tolerance`。默认每批 100 万生成肽段，限制额外诊断缓冲区的内存；输入、二进制哈希、命令和批次大小保存在 `manifest.json`。报告与逐 scan 明细位于 `analysis/`，重分析写入新的 `analysis_<时间>/`。

比较口径：

- 同一候选出现实例在 merge/top **之前**分别计算 CUDA 和 RT 的 MVH 分数；仅输出分数或评分资格改变的实例，并保存包含未变化实例的完整分母。
- 分数差为 RT − CUDA；默认 `1e-9` 用于区分实质分数变化及同分集合，同时保留精确不等计数。未达到匹配数门槛的分数留空，不当作 0 分。
- 最终比较从独立正常搜索的完整 PSM 列表计算：top-1 肽段、top-1 完整前体假设、top-N 成员、成员相同时的排序变化、同分集合变化。PSM 按唯一的 `scan_index` 分组，不假定 `scan_id` 永远唯一。
- 同一肽段若选择了不同前体电荷/质量，另记为假设变化，不混入同一候选的分数差。
- 诊断搜索最终 PSM 必须与相应的 RT 正常搜索逐字节一致；不满足则终止分析。最终筛选变化的 scan 必须能找到候选分数或资格变化，避免遗漏诊断数据时给出结论。
- 候选统计包含重复及随后可能被合并的候选。与最终筛选的关联在 scan 层进行，不宣称某个重复实例单独导致了结果变化，也不将 CUDA 参照等同于生物学鉴定真值。

底层 `--score-impact` 仅允许 `rt-triangle` / `rt-instanced` / `rt-custom`；诊断模式需要临时恢复 CUDA 桶及额外 CUDA 评分结果，故不能用来测 RT 省桶性能。普通搜索不启用该选项，仍走原先的省桶路径。内核/写出实现放在独立 `mvh_cuda/cuda/score_impact.*`，分析与调度分别位于 `score_impact_report.py` 和 `validate_score_impact.py`。

默认将本次生成的 PSM 压缩并校验解压哈希后移除明文副本；`--keep-psms` 可保留明文。`--analyze-only` 同时支持两种形式。

完整 E. coli 验证（46,066 个 scan）：两种 RT 后端结果一致；与 CUDA 相比，30,737 个双方可评分的候选出现实例分数改变，42,341 个实例失去评分资格；627 个 scan 的 top-1 肽段改变（其中 114 个为同分集合内换序），12,879 个 scan 的最终 top-N 成员改变。诊断与正常 RT 搜索的最终 PSM 逐字节相同。分数分布、完整分母及明细说明见 [验证报告](../../output/validation/score_impact/ecoli_20260923T184046_786090Z/analysis/REPORT.md)。

## 独立实验后端 rt-custom

`rt-custom` 当前为内置 sphere 基础框架：球心 `(m/z, class, 0)`，半径取质量容差，从上方向下发射并选择 closest-hit。class 不反转，编号较大的组优先；class 0 不在几何阶段过滤，但现有评分仍不把它作为有效评分组。

原有 `rt-triangle` / `rt-instanced` 保留作旧算法对照，默认后端仍为 CUDA。新规则不要求与旧 CUDA/三角形输出相同；正常与诊断运行、同一后端的批次一致性仍须通过检查。旧 AABB 版本的零差异报告不适用于当前 sphere 实现。

普通新后端继续省略桶索引。`validate_score_impact.py` 可用 `--backends rt-custom` 单独比较新旧规则，性能脚本同样支持该选项。

人工设计入口、资源复用约定和验证命令见 [CUSTOM_MATCHING.md](CUSTOM_MATCHING.md)。

Sphere 场景资源与构建已独立到 `sphere_backend.cpp`；旧三角形场景在 `triangle_backend.cpp`。`bridge.cpp` 只分发后端，公共 GAS/SBT/上传逻辑在 `scene_resources.cpp`。射线设计继续修改 `custom_device.cu`。
