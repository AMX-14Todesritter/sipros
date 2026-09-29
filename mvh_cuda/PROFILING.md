# Detailed RT / CUDA function timing with Nsight Systems

`MVH_ENABLE_PROFILING` defaults to `OFF`. When enabled, host code emits NVTX3 ranges. It does not change matching, MVH arithmetic, candidate ordering, ray generation, geometry, or synchronization. Device kernels and OptiX PTX receive no NVTX instrumentation.

The interface is isolated in `include/profiling.h`. Disabled macros expand to `((void)0)`: no range objects, label evaluation, NVTX headers, or NVTX linkage are required. Existing CUB/OptiX library NVTX ranges can still appear in OFF captures; the switch controls only the application ranges added here. Enabled builds use the CUDA toolkit's header-only NVTX3 and its platform loader (`libdl` on Linux).

## 细粒度函数耗时

启用 `-DMVH_ENABLE_PROFILING=ON` 后，RT 后端及 CUDA 共用路径会输出嵌套 NVTX 标记。
重新编译后可直接运行：

```bash
# 小数据验证；每次运行自动创建新的输出目录。
bash MVH_RT/gpu_bridge/run_profile.sh --dataset smoke --tools nsys --batch 50 --range-trace

# 实际数据统计；省略 --range-trace 可以减少导出文件的体积。
bash MVH_RT/gpu_bridge/run_profile.sh --dataset marine --tools nsys --batch 6000000
```

每次 NSYS 运行自动导出：

| 文件 | 含义 |
|---|---|
| `timings_nvtx_sum.csv` | 函数/阶段的调用次数、总耗时、平均值、中位数、最小值、最大值、标准差 |
| `timings_cuda_gpu_kern_sum.csv` | CUDA / OptiX kernel 的 GPU 执行耗时统计 |
| `timings_cuda_api_sum.csv` | CUDA API 的主机端耗时统计 |
| `timings_cuda_gpu_mem_time_sum.csv` | GPU 内存传输/填充耗时统计 |
| `timings_nvtx_pushpop_trace.csv` | 仅 `--range-trace`：每次调用的时间、父调用、线程、嵌套层级和自身耗时 |

时间列默认单位为 ns。逐调用表的 `Duration` 包含子调用，`DurNonChild` 扣除了
已插桩的子调用；未插桩函数的耗时仍计入其父调用的自身耗时。
父子时间不能相加，NVTX 汇总中的百分比也不是整个程序运行时间的占比。
模板 Buffer 的统计合并所有元素类型；upload 构造函数的标记不包含先执行的委托分配构造函数。

新增覆盖范围：

- `mvh/function/MvhScanVector::*`：读谱、预处理入口及搜索前后处理。
- `mvh/function/*`：设备初始化、配置提取、谱图恢复、exclusive prefix sum、谱图设备缓存上传/释放及评分入口。
- `mvh/preprocess/scans/*`、`mvh/preprocess/peptides/*`：主机打包、预处理 kernel 与已有等待、结果下载/恢复/校验。
- `mvh/assignment/*`：候选关联的主机打包、质量窗口 kernel 提交、候选生成和稳定排序、校验。
- `mvh/memory/Buffer/*`：CUDA 共用 buffer 分配、上传、下载和释放。
- `mvh/rt/bridge/*`、`mvh/rt/sphere/*`、`mvh/rt/triangle/*`：后端派发、准备/复用、配置检查、几何输入准备及 launch。
- `mvh/rt/SceneResources::*`、`mvh/rt/accel/*`：布局校验、加速结构内存计算、构建及已有等待、SBT 初始化。
- `mvh/rt/optix/*`：OptiX 初始化、PTX 读取、module 创建、program group 创建、pipeline 链接、stack size 计算及资源销毁。
- `mvh/rt/geometry/*`、`mvh/rt/DeviceBuffer/*`：几何生成的主机提交函数，以及 RT buffer 分配/传输/释放。
- `mvh/rt/launch/*`：launch 参数上传、`optixLaunch` 提交；`mvh/gpu/scoring_completion_wait` 单独标记已有的评分完成等待。
- `mvh/gpu/synced`、`mvh/diagnostic/rt_audit`：已有 device synchronize 和 RT audit 成本。

这是**主机函数/阶段耗时**，异步提交函数返回不代表 GPU 已完成。
`*_submit` 和几何生成 wrapper 的时长不能当作 GPU kernel 时长；后者应查看 kernel CSV 或 GPU 时间线。
未向 device 内联函数、逐离子匹配逻辑或原始 CPU 源码副本中插入计时，也没有新增 GPU 同步。
肽生成继续按批次统计，避免产生数百万逐肽事件；GPU 函数以实际 kernel 为统计单位。
细粒度事件会增加采集开销，正常性能基线仍应使用 OFF 构建。

## Separate builds

Run from WSL. Keep the normal benchmark binary in `build/mvh_rt/gpu_integration` unchanged while investigating performance:

```bash
docker exec sipros-sipros-1 cmake \
  -S /workspace/sipros/mvh_cuda \
  -B /workspace/sipros/build/mvh_rt/profile_enabled \
  -DCMAKE_BUILD_TYPE=Release \
  -DMVH_CUDA_ENABLE_RT=ON \
  -DOPTIX_ROOT=/workspace/sipros/build/mvh_rt/optix_sdk \
  -DMVH_ENABLE_PROFILING=ON

docker exec sipros-sipros-1 cmake --build \
  /workspace/sipros/build/mvh_rt/profile_enabled -j4
```

For a fresh normal build, use a different directory (e.g. `profile_disabled`) and `-DMVH_ENABLE_PROFILING=OFF`. An existing CMake directory retains its cached option, so explicitly set `OFF` when disabling instrumentation, then rebuild. An instrumented binary still executes range calls when NSYS is not attached; use the OFF build for performance baselines.

## Parameterized capture script

From the WSL project directory, run:

```bash
bash MVH_RT/gpu_bridge/run_profile.sh --dataset marine --batch 6000000
```

Defaults: sphere (`--backend rt-custom`), the `profile_enabled` binary, NSYS then NCU in separate processes. NSYS runs the complete search and captures CUDA/NVTX/OS runtime events without CPU sampling. NCU uses `--set basic`, selects `optixLaunch`, profiles one matching launch and terminates its search early. NCU replay time is not an end-to-end benchmark. Its result directory is intentionally incomplete unless `--ncu-complete-search` is set.

Common variants:

```bash
# Preview only; no execution or output-directory creation.
bash MVH_RT/gpu_bridge/run_profile.sh --dry-run

# Only the complete timeline / host-stage capture.
bash MVH_RT/gpu_bridge/run_profile.sh --tools nsys --batch 6000000

# Only GPU counters: skip two matching scoring launches and collect the next three.
bash MVH_RT/gpu_bridge/run_profile.sh --tools ncu --batch 6000000 \
  --ncu-launch-skip 2 --ncu-launch-count 3 --ncu-set detailed

# Keep the NCU-launched search running to completion after collecting counters.
bash MVH_RT/gpu_bridge/run_profile.sh --tools ncu --ncu-complete-search

bash MVH_RT/gpu_bridge/run_profile.sh --help
```

`--ncu-launch-skip` counts matching scoring launches, not all CUDA kernels. A batch with zero candidates has no scoring launch, so the count need not equal the generated-peptide batch index. Larger metric sets replay kernels more times and increase profiling overhead. If fewer launches exist than requested, the script reports failure and preserves diagnostics.

Additional options: `--dataset smoke|ecoli|marine`, `--backend cuda|rt-triangle|rt-custom`, `--spectrum-cache host|device` (optimized binary only), `--binary`, `--fasta`, `--scans`, `--config`, and `--output`. File overrides and output paths refer to the container filesystem. `SIPROS_CONTAINER` selects another existing container with the same project mount. The script does not build code or install profilers; rebuild `profile_enabled` after code changes.

Outputs are created under `output/benchmarks/profiling/<dataset>_<backend>_batch<size>_<timestamp>/` unless overridden, and existing directories are never overwritten. It stores `manifest.json` (parameters, commands, tool versions via logs, input/binary/PTX/cache hashes and completion status), `nsys_capture.nsys-rep`, `ncu_capture.ncu-rep`, `nsys_summary.log`, `ncu_summary.log`, and per-tool logs/results. Only selected tools produce their corresponding files. Ctrl+C requests cancellation and saves an interrupted manifest.

## Small sphere capture

The following example uses only the repository's small test input. Choose new output names if they already exist. It does not run Marine.

```bash
docker exec sipros-sipros-1 mkdir -p /workspace/sipros/output/validation/profiling

docker exec \
  -e LD_LIBRARY_PATH=/workspace/sipros/build/mvh_rt/optix_runtime:/usr/local/cuda/lib64 \
  sipros-sipros-1 nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output=/workspace/sipros/output/validation/profiling/sphere_nvtx \
  /workspace/sipros/build/mvh_rt/profile_enabled/bin/sipros_mvh_cuda \
  -f /workspace/sipros/mvh_cuda/tests/data/sample.ft2 \
  -c /workspace/sipros/mvh_cuda/tests/data/search.cfg \
  -fasta /workspace/sipros/mvh_cuda/tests/data/proteins.fasta \
  -o /workspace/sipros/output/validation/profiling/sphere_sample \
  --match-backend rt-custom --peptide-batch-size 3

docker exec sipros-sipros-1 nsys stats \
  --report nvtx_sum,cuda_gpu_kern_sum \
  /workspace/sipros/output/validation/profiling/sphere_nvtx.nsys-rep
```

The current container can collect NVTX and CUDA timelines with CPU sampling disabled. The existing `run_benchmark.sh` selects `gpu_integration/bin/sipros_mvh_cuda`, not this separate profiling binary. Use the explicit binary above for captures.

## Range meanings

| Range | Includes |
|---|---|
| `mvh/run` | Entire application runner, including resource destruction |
| `mvh/run/config_and_load` | Configuration, output setup, scan loading |
| `mvh/run/preprocess_scans` | Experimental spectrum preprocessing |
| `mvh/search/database` | Database search orchestration and cleanup |
| `mvh/search/load_database`, `mvh/search/prepare` | Database loading and pre-search preparation |
| `mvh/search/generate_peptides` | Database iteration, digestion, peptide construction/allocation and batch assembly, excluding batch processing |
| `mvh/batch/process` | One batch's association, preprocessing, scoring/restoration and peptide deletion |
| `mvh/batch/assign_scans` | Host setup and GPU candidate association |
| `mvh/batch/preprocess_peptides` | Peptide text preparation, transfer, GPU preprocessing and validation |
| `mvh/pack/all` | All host scoring-data packing |
| `mvh/pack/setup` | Batch ownership transfer, vector/map reservation, contract-only association fallback |
| `mvh/pack/sequence_ids` | Peptide sequence hashing/ID assignment and peptide metadata array |
| `mvh/pack/spectra_and_top` | Prepare/reuse fixed spectra and construct current top candidates (including their sequence IDs) |
| `mvh/pack/prepare_or_reuse_spectra` | First-batch fixed spectrum/lookup-table packing, or subsequent dataset/configuration checks |
| `mvh/pack/ln_table` (earlier baseline only) | Lookup table copy; the optimized version includes this once in prepare_or_reuse_spectra |
| `mvh/gpu/service` | Scoring service, including temporary-buffer destruction |
| `mvh/gpu/allocate_upload` | Device allocation, upload and candidate-range setup |
| `mvh/rt/prepare_or_reuse` | First RT setup or subsequent layout/configuration checks; does not mean BVH is rebuilt every batch |
| `mvh/gpu/theory_cache` | Optional theoretical-ion cache preparation |
| `mvh/gpu/match_and_score` | Submission and existing wait for scoring completion; includes direct ion generation when uncached |
| `mvh/gpu/retain_top`, `mvh/gpu/compact_results`, `mvh/gpu/download_results` | Candidate retention, result compaction and download |
| `mvh/gpu/release_temporaries` | Return handling and destruction of scoring device temporaries |
| `mvh/host/restore_results` | Host restoration/merge of scored candidates |
| `mvh/host/release_packed_batch` | Return handling and destruction of packed arrays, sequence map and retained GPU batch resources |
| `mvh/batch/delete_peptides` | Deletion of CPU peptide objects and clearing their vector |
| `mvh/search/finalize` | Post-search processing and lookup-table cleanup (earlier local destructors remain in the outer database range) |
| `mvh/run/export_psms` | PSM file export |

Ranges measure host elapsed time, including waits, allocation and synchronization already present in the program. They do not measure pure CPU execution time. Parent ranges include children: do not add parent and child totals. The default `nvtx_sum` percentage uses overlapping range totals, so it is not a percentage of end-to-end wall time. Compare GPU kernel durations on the GPU timeline separately.

Generation uses one range per batch-sized chunk, paused while processing that batch. It does not emit events for individual peptides or ions; spectrum storage and RT build-input helpers do emit per-scan events. An empty terminal chunk can appear for an exact batch-size multiple; count `mvh/batch/process` for processed batches. Cleanup ranges are declared before the objects they cover and activated at function exit, so their destructors close the ranges after those objects are destroyed.

## Disable or remove

For continued optimization, simply use/rebuild the OFF configuration; the markers can remain in source with no NVTX calls. The shared `mvh_profiling` interface propagates the option and NVTX dependency to the RT support library and bridge as well as the CUDA engine. Standalone OptiX tutorial builds keep these macros disabled. No search logic depends on the profiling wrapper.

Keep instrumentation changes separate from algorithm optimizations in version control. This change does not automatically commit the pre-existing workspace changes. Validate enabled/disabled PSM equality on the same inputs; use non-profiled runs for timing comparisons because capture adds overhead.
