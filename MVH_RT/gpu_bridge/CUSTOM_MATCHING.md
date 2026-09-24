# rt-custom：内置 sphere 基础框架

当前 `rt-custom` 已切换到内置 sphere，用于人工继续设计射线与命中策略。主程序、数据预处理、理论离子生成、MVH 评分和最终候选筛选继续共用现有代码。`rt-triangle`、`rt-instanced` 保留作旧算法对照，默认后端仍为 `cuda`。

## 当前可运行的设计

- 一个观测峰对应一个球，球心为 `(float(mz), float(class), 0)`，不反转 class、不排除 class 0。
- 统一半径为 `float(fragmentTolerance)`。目前要求半径在 `(0, 1)` 内，以保持单位 class 间距下的组间优先级。
- 射线起点为 `(float(query_mz), C + radius + 1, 0)`，沿负 y 方向发射，长度覆盖到 class 0 球体下方。`C` 为实际峰组数，输入约定 class 在 `[0, C]` 内；当前预处理只生成 `1..C`。
- 使用内置 sphere 求交和 closest-hit，返回 scan 内峰索引。编号较大的 class 优先；这不等于更强的峰，当前 class 1 是最强组。
- 几何允许 class 0，但现有 MVH 仍将它作为不计分匹配处理。`NoPeak` 独立表示没有命中，不用 class 0 充当 primitive 未命中标记。
- 不添加双精度复核、边界修正、额外 ray 或 fallback。这是基础实现，不能据此宣称与双精度参考等价。

统一半径下，理想球面入口距离为 `H - class - sqrt(radius² - mass_error²)`。实际执行使用浮点几何和命中距离：边界、极近邻和同距离候选不保证遵循双精度次序；当前也不强制以峰索引打破同分。仅在 closest-hit 后丢弃误命中，不会自动找到下一个有效候选。

## 后续人工修改入口

| 文件 / 函数 | 职责 |
|---|---|
| `custom_geometry.cu / sphereCentersKernel()` | 修改球心坐标映射，保持球与峰索引对应 |
| `custom_geometry.h / generateSphereCenters()` | 主机调用入口；kernel 本身不暴露给 C++ 调用方 |
| `custom_device.cu / makeSearchRay()` | 射线起点、方向和 t 范围 |
| `custom_device.cu / tracePeak()` | trace flags、payload、额外 trace 或查询流程 |
| `custom_device.cu / __closesthit__record()` | 命中后的峰索引回传 |
| `custom_device.cu / __miss__background()` | 无命中结果 |
| `custom_device.cu / SphereCounter::add()` | 理论离子查询与 class/MVH 统计衔接 |
| `custom_device.cu / __raygen__camera()` | 每个 launch index 处理一个候选；每个理论离子在 add 中发射 ray |
| `sphere_backend.cpp / prepare()` | 独立 sphere 场景资源、build input 和 GAS 复用 |
| `sphere_backend.cpp / launch()` | 校验配置并将射线范围加入 Params |
| `scene_resources.cpp` | 通用 GAS 内存构建、SBT、参数上传与 optixLaunch |
| `bridge.cpp` | 仅后端分发及 reset，不构建几何 |
| `triangle_backend.cpp` | 旧三角形/实例化场景，独立保留作对照 |
| `bridge.h / Params` | 主机与设备参数布局，改动会触发两套 PTX 重编译 |
| `../optix_example/rt_support.cpp / createPipeline()` | 内置 sphere intersection 模块、hitgroup 和 pipeline 注册 |

**当前没有自写的 `__intersection__peak()`。** 求交由 `optixBuiltinISModuleGet()` 得到的 sphere 模块执行。要继续使用内置 sphere，应在 ray/closest-hit 等步骤设计；若决定自行编写 intersection，则需另行切换 custom primitive/AABB 的 build input 和 pipeline，不能直接将用户 IS 替换到内置 sphere 上。

## GAS/BLAS 复用范围

每个非空、未跳过的 scan 构建一个 GAS，多个 GAS 共用大块显存分配和临时 scratch。每个 scan 的 GAS 在肽段批次之间复用。这不是多个峰共享一个基础球 BLAS 的实例化设计。

`SphereState` 持有球心、统一半径、类别和射线范围；其 `SceneResources` 成员持有加速结构、handles、pipeline 和 SBT。`prepare()` 校验 scan 布局、几何类型、半径和类别数量；`launch()` 再次校验半径和类别数量。主机 build-input 地址一直存活到所有构建完成，内置 intersection 模块在 program groups/pipeline 释放后销毁。

同一数据集内质量和 class 必须保持不变：当前复用检查不是数组内容哈希。重新预处理、变更坐标数据时调用 `reset()`。每批上传的峰/class 指针不作为永久指针缓存，持久球心资源由 SphereState 自己拥有。

普通搜索不创建桶。`--score-impact` 额外计算旧 CUDA 参考；`--verify-cuda` 使用原 CPU 规则复算，不能作为新 class 优先规则的正确性判据。

## 验证和构建

```bash
docker exec -w /workspace/sipros sipros-sipros-1 \
  cmake --build build/mvh_rt/gpu_integration -j2

docker exec -w /workspace/sipros \
  -e LD_LIBRARY_PATH=/workspace/sipros/build/mvh_rt/optix_runtime:/usr/local/cuda/lib64 \
  sipros-sipros-1 ctest --test-dir build/mvh_rt/gpu_integration --output-on-failure

# 先验证可运行的主程序接入及正常/诊断一致性。
docker exec -w /workspace/sipros sipros-sipros-1 \
  python3 -B MVH_RT/gpu_bridge/validate_score_impact.py --dataset smoke --backends rt-custom

# 人工设计完成后，可比较完整 E. coli 的分数和最终候选。
docker exec -w /workspace/sipros sipros-sipros-1 \
  python3 -B MVH_RT/gpu_bridge/validate_score_impact.py --dataset ecoli
```

`custom_contract.cpp` 验证布局、class 优先、class 0、不匹配、3/4 个实际峰组、半径/类别变化检查和 GAS 复用。测试采用远离浮点临界边界的案例；它不代表完整精度验证。两种旧三角形后端的已有测试保留，sphere 的批次和诊断一致性独立验证。

MVH 分数与 top 候选报告是与旧 CUDA 规则的比较，包含规则变化和数值误差。要区分二者，还需按最终人工设计建立同规则双精度参考。

历史 `ecoli_20260923T202854_899020Z` 中的 rt-custom 零差异报告来自之前的 AABB/双精度初版，不能用于证明当前 sphere 实现的准确性。新运行的 manifest 会记录匹配规则及可定位到的 PTX 哈希；源码仍应由 Git 版本保存。

## Params 中新增字段的传递

`bridge.h` 声明 `rayOriginY` 和 `rayTmax`。`sphere_backend.cpp::prepare()` 根据类别数量与球半径计算它们，保存在 SphereState；每批 `sphere_backend.cpp::launch()` 填入传入的 Params 副本。`SceneResources::launch()` 再填入 handles，使用 cudaMemcpy 上传整个 Params，并将地址及 sizeof(Params) 传给 optixLaunch。设备端 `__constant__ Params params` 由 OptiX 的 launch 参数机制提供，`makeSearchRay()` 从中读取范围。主程序不需要重复传入这两个派生值。

仅改变射线范围时可修改 sphere_backend.cpp 的计算；如果新增字段，则同步修改 bridge.h、主机赋值和设备端读取。不要只改一侧结构布局。sphere 场景文件没有三角形几何分支；通用 pipeline 工具仍为各后端共用。
