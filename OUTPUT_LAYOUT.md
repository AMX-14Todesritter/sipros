# 输出目录约定

新运行默认写入项目根目录的 `output/`，按用途、模块和运行名称分层：

```text
output/
├── search/{mvh,mvh_cuda,mvh_rt}/<输入名>_<UTC时间>/
├── workflow/sipros/<标记元素>_<UTC时间>/
├── benchmarks/gpu_bridge/<数据集或benchmark>_<UTC时间>/
├── validation/mvh_cuda/<smoke或ecoli>_<UTC时间>/
├── profiling/mvh_cuda/ecoli_<UTC时间>/
├── analysis/regular_mvh/ecoli_<UTC时间>/
├── exports/{peak_export,mvh_rt}/<输入名或scan_1004>_<UTC时间>/
└── examples/optix/spheres_<UTC时间>/
```

时间格式为 `20260923T153012_123456Z`（UTC，微秒）。输入名中的非 ASCII 字母、数字、下划线、连字符会替换为下划线。每次运行生成新目录，日志、报告、结果和该次运行的配置仍放在同一目录；文件名和文件格式保持原样，方便现有读取脚本继续使用。

## 使用方式

以下命令在项目开发容器内运行：

```bash
# 自动生成 output/search/mvh/sample_<UTC时间>/
build/mvh/bin/sipros_mvh -f mvh/tests/data/sample.ft2 \
  -c mvh/tests/data/search.cfg -fasta mvh/tests/data/proteins.fasta

# 指定路径：保持相对于当前工作目录的解释方式；支持自动创建父目录
build/mvh/bin/sipros_mvh -f mvh/tests/data/sample.ft2 \
  -c mvh/tests/data/search.cfg -fasta mvh/tests/data/proteins.fasta \
  -o output/search/mvh/my_experiment

python3 mvh_cuda/tests/validate.py
python3 MVH_RT/gpu_bridge/run_suite.py --dataset smoke
python3 analysis/regular_mvh/analyze_regular_mvh.py
bash MVH_RT/scripts/run_optix.sh
```

`-o` / `--output` 优先于自动目录。`SIPROS_OUTPUT_ROOT` 可统一更换默认输出根目录，例如 `/workspace/sipros/output/session_a`。相对的环境变量路径按项目根目录解析，默认目录不会随当前工作目录改变。C++ 入口的项目根目录由 CMake 提供；移动构建产物后应重新构建或设置绝对的 `SIPROS_OUTPUT_ROOT`。

搜索入口和实验脚本拒绝复用已有运行目录。工作流 `script33/main.py` 显式指定已有目录时仍保留其原有覆盖警告行为；省略参数则生成新目录。

## 实现与兼容

- Python 路径规则集中在 `shared/output_paths.py`；C++ 搜索入口使用 `shared/output_paths.h`。路径函数只计算路径，目录创建由调用方负责，避免导入模块就产生输出。
- 已接入独立 CPU/CUDA/RT 搜索、Python 工作流、GPU benchmark suite、CUDA 验证/NCU、Regular MVH 分析、两种数据导出及 OptiX 启动脚本。
- 原始底层 `bin/sipros` / MPI 程序保留原有参数语义。直接运行它们时请显式指定 `-o output/search/sipros/<实验名>`；通过 Python 工作流运行时会收到统一后的输出路径。
- `build/`、`bin/`、`tools/` 是构建或工具目录，不并入运行结果。
- 临时集成测试仍使用自动清理的临时目录。版本化测试样本、既有 `MVH_RT/test_data/`、`test_output/` 和历史实验资料仍作为输入保留，不自动移动或删除。
- `export_data.py` 新导出的快照放入 `output/exports/mvh_rt/`；检查新快照时运行 `check_data.py <新快照目录>`。省略目录的检查命令及 OptiX 示例仍读取保留的基准样本。
- 历史验证文档中的旧结果路径是当时的记录，不批量改写。现有 `output/` 下的旧文件也不迁移，以免破坏引用。

## 验证

```bash
python3 -B shared/tests/check_output_paths.py --binary build/mvh/bin/sipros_mvh --component mvh
ctest --test-dir build/mvh --output-on-failure
ctest --test-dir build/mvh_cuda --output-on-failure
```
