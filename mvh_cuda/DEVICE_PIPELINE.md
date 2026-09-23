# GPU 数据路径优化验证（2026-09-22）

本轮直接修改现有 `mvh_cuda/`，保留 CPU `mvh/` 与 `original/` 不变；未新建 CUDA baseline。所有编译、运行和测试在现有 `sipros-sipros-1` 容器中执行。没有安装依赖或更改容器配置。

## 改动和读取顺序

1. `src/database_search.cpp`：保留 `searchDatabaseMvh → processPeptideArrayMvh`；批次现在按生成的肽段计数。把质量分配移到批次内部，避免单肽段启动 CUDA。
2. `cuda/assignment.cuh`：保留同名 `GetAllRangeFromMass → GetRangeFromMass`；GPU 二分查询、原窗口合并、前缀和、稳定 scan 分组。关联直接留在显存。
3. `cuda/engine.cu`：肽段预处理中性丢失文本留在显存；打包不再遍历每一条关联，只处理肽段和 scan 描述。
4. `cuda/theoretical.cuh`：原公式生成的理论峰按肽段对象和电荷在批内复用。显存不足/高电荷使用直接 GPU 计算，绝不改变质量或评分公式。
5. `cuda/scoring.cuh`：候选评分并行；随后每个 scan 按原顺序 merge/top。正常模式只回传状态改变事件，CPU 保留原结果对象构建。
6. `app/main.cpp`：新增 `--peptide-batch-size`；原 CLI 继续可用。
7. 测试：增加质量窗口端点/重复交界、稳定关联顺序、小批次切换、缓存及直接路径、高电荷路径测试。CPU 基准指纹更新只接受已存在的 runner 行首单空格，无 CPU 计算变化。

## 验证

- 4 项 CTest 全部通过。
- compute-sanitizer memcheck：0 errors，0 bytes leaked（边界语义测试程序）。
- 完整真实数据 `--verify-cuda` 与 CPU 逐评分/merge/top/预处理/质量窗口对照通过。
- 3 轮 CPU/GPU 交替顺序性能运行，6 份 PSM 与历史基准逐字节一致。
- 每次保留 1,403,362 条 PSM；每次逻辑关联与评分计数也与历史一致。

SHA256：`2f96a76b931b89c64054026533e4de5379fca959848a0a314c8ea42332f31520`。

## 搜索性能

| 版本 | 第一次/秒 | 第二次/秒 | 第三次/秒 | 平均 ± 标准差/秒 |
|---|---:|---:|---:|---:|
| CPU 四线程 | 15.3565 | 16.0557 | 15.1452 | 15.5191 ± 0.4766 |
| 当前 CUDA | 7.7336 | 7.5888 | 7.7352 | 7.6859 ± 0.0841 |

本轮 CPU/CUDA 完整搜索加速比：**2.02×**。此前同日旧 CUDA 平均 19.9951 秒，当前相对该历史测量为 2.60×；该旧值不是与本轮交替重跑的结果。

搜索时间包括数据库遍历、酶切/PTM、质量分配、GPU 执行和结果恢复，不含读谱、实验峰预处理和 TSV 输出。完整输出和原始分段时间见 performance/report.json。未锁定频率或控制 Windows 侧负载，三次结果只说明当前真实样本、配置和硬件的情况。

## 当前分段平均

| 区间 | 每次搜索平均秒 |
|---|---:|
| pack_seconds | 2.107111 |
| gpu_service_seconds | 0.824969 |
| allocation_upload_seconds | 0.111191 |
| theory_seconds | 0.164110 |
| kernel_seconds | 0.320128 |
| retention_seconds | 0.105662 |
| compaction_seconds | 0.008918 |
| download_seconds | 0.091136 |
| replay_verify_seconds | 1.494244 |
| assignment | 0.145684 |
| peptide_preprocessing | 0.273557 |

GPU 服务包含 theory、kernel、retention、compaction、传输及资源管理，父子项不可重复相加。kernel_seconds 仅为候选评分 CUDA event 时间。缓存计数不等于所有评分尝试访问的离子总数。

## 仍留在 CPU 的部分

FASTA 读取、原 `ProteinDatabase` 酶切/PTM 枚举、肽段质量估计、配置/对数表、结果对象和导出仍在 CPU。此轮主要目标是消除大规模关联在 CPU 上的构建/打包、让评分具有候选级并行度，不是声称整段搜索已完全 CUDA 化。

批次大小由命令行控制；默认 2,000,000 个生成肽段。设备仍有 128 残基、512 字节文本、8 强度类别的显式边界。详细结构和运行命令见 README.md。

日志与结果：`output/mvh_cuda_device_pipeline_20260922/`。

## 追加批次与内存检查

真实数据改为每批 250,000 个生成肽段，PSM 仍与上述 SHA256 完全一致。该次搜索耗时 16.573249516999567 秒：较小批次降低内存需求，但增加重复打包/传输及 top 状态恢复，默认仍采用 2,000,000。

正常模式的事件压缩路径另通过 compute-sanitizer：0 errors、0 bytes leaked；与校验模式语义测试的内存检查分别记录。
