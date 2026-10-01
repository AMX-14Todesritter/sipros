# rt-custom：整数/小数拆分 sphere

每个实验峰的 double m/z `d` 先在 double 精度下计算 `n=floor(d)`、`f=d-n`，再生成球心 `(float(n),float(f),0)`。球半径为 `float(fragmentTolerance)`，class 保存在原始峰数组中，不参与几何位置或匹配优先级。

当 `f <= tolerance` 时补 `(n-1,f+1,0)`；当 `1-f <= tolerance` 时补 `(n+1,f-1,0)`。例如 `d=2.9,tolerance=0.2` 生成 `(2,0.9,0)` 和 `(3,-0.1,0)`。补点边界使用包含等号的判断，避免遗漏边界几何。

理论峰 `c` 同样在 double 中拆分，射线从 `(float(floor(c)),float(c-floor(c)),0.5)` 沿 `(0,0,-1)` 发射，t 范围为 `[0,1]`。同半径球的入口距离为 `0.5-sqrt(radius²-fraction_error²)`，因此 closest-hit 选择几何距离最近的实验峰，不考虑 class。

当前要求 `0 < float(tolerance) < 0.5`，以使固定 z=0.5 的原点在球外，并隔离相邻整数列。内置 sphere intersection 使用 float；拆分降低了大 m/z 的转换误差，但严格容差边界、极近邻及相同距离的命中次序仍不保证与 double CPU/CUDA 完全一致。未增加双精度复核或 fallback。

## 数据与资源

`custom_geometry.cu` 在 GPU 计数、前缀和并生成紧凑球心数组及原始峰 ID 数组。每个非空 scan 构建一个 GAS，按峰的前缀和定位 primitive 范围。球心、偏移和 ID 映射在整个数据集内复用，批次之间无需重建树。

`__closesthit__record()` 将 GAS 内 primitive ID 通过全局偏移和映射还原为 scan 内原始峰 ID；基本点和补点共用同一峰身份。`SphereCounter::add()` 再从原始 class 数组读取 MVH 类别。class 0 可成为最近峰，但不贡献匹配分数；`NoPeak` 单独表示未命中。

CPU `PeakList::findNear` 与 CUDA 桶查询均恢复为严格 `abs(error)<tolerance` 的最近峰规则，不以 class 筛选；相同距离保留先遇到的峰。RT 同距离命中仍由 OptiX 遍历决定。

## 构建与验证

```bash
docker exec -w /workspace/sipros sipros-sipros-1 cmake --build build/mvh_rt/gpu_integration -j4
docker exec -w /workspace/sipros -e LD_LIBRARY_PATH=/workspace/sipros/build/mvh_rt/optix_runtime:/usr/local/cuda/lib64 sipros-sipros-1 ctest --test-dir build/mvh_rt/gpu_integration --output-on-failure
bash MVH_RT/gpu_bridge/run_benchmark.sh --dataset marine --batch 8000000 --repeats 1 --backends cpu cuda rt-custom
```

`custom_contract.cpp` 检查拆分坐标、补点数量和身份、跨整数边界命中、最近峰而非 class 优先、大质量小数精度、class 0、无命中、GAS 复用和配置变化拒绝。小样本集成另行比较 CPU/CUDA/RT-custom 输出与批次一致性。完整数据计时关闭诊断与 CPU 复算，报告独立记录 PSM 哈希是否一致。

marine benchmark 采用 `raw/Marine_fw_3rev.fasta`、`test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2` 和 `experiments/Regular.cfg`。CPU 使用 4 线程，三条路径均按生成肽段计数分批。
