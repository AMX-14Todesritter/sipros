# 从两个球体开始理解 OptiX

本例先让你理解射线、primitive、BVH 和程序回调。它没有把实验峰映射成几何体，所有坐标都是普通三维场景坐标。代码的重要注释使用英文。

## 1. 先读场景：primitive 是什么

看 `optix_example/scene.h` 的 `Sphere`，再看 `main.cpp` 的 `makeScene()`。

一个球体包含中心 `center`、半径 `radius`、显示颜色 `color`。这里使用 **custom primitive**，即告诉 OptiX：“有这样的一个几何对象，但具体怎么求交由我写程序定义。”

这与 OptiX 内置 triangle primitive 不同。我们故意选 custom primitive，方便后续学习自定义形状。

## 2. 提供 AABB，让 OptiX 自己建立 BVH

`makeBounds()` 为每个球体产生一个轴对齐包围盒：

```text
min = center - radius
max = center + radius
```

主机把 AABB 数组复制到 GPU，然后填写 `OptixBuildInput`：

```cpp
input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
input.customPrimitiveArray.aabbBuffers = &aabbAddress;
input.customPrimitiveArray.numPrimitives = spheres.size();
input.customPrimitiveArray.numSbtRecords = 1;
```

随后：

```text
optixAccelComputeMemoryUsage()  → 查询构建临时空间和结果空间
申请 scratch 和 gasMemory
optixAccelBuild()               → OptiX 构建加速结构
得到 OptixTraversableHandle gas
```

你不需要手工划分树节点，也不需要先按 x/y/z 排序 primitive。OptiX 根据输入和 build options 构造内部加速结构。`gas` 是不透明的访问句柄，不是让应用遍历节点的 C++ 树指针；构建结果的 GPU 内存必须存活到 tracing 结束。

GAS 是 Geometry Acceleration Structure。本例只有一个 GAS，没有实例层 IAS，因此 pipeline 使用 `ALLOW_SINGLE_GAS`。

**AABB 不是球体本身。** 射线经过 AABB 只说明这个球体值得进一步检查。包围盒必须包住 primitive，否则可能错误地漏掉真实交点。

## 3. 理解 host 如何组装 tracing pipeline

`createPipeline()` 依次创建：

1. **Module**：从 NVCC 生成的 PTX 加载设备程序。
2. **Program groups**：分别绑定 raygen、miss、hitgroup 的入口名称。
3. **Pipeline**：把这些程序组连接成可启动的 tracing 流程。
4. **Stack sizes**：根据实际程序计算栈空间。本例最多调用一层 `optixTrace()`。

这里的 hitgroup 包含 intersection 和 closest-hit；没有使用 any-hit，也没有递归反射/折射。

`SbtRecord` 和 `OptixShaderBindingTable` 指定这些程序的记录位置以及用户数据。SBT（Shader Binding Table）并不是 BVH，它连接“遍历事件”与“执行哪个程序、使用什么数据”。

本例所有球体共享同一个 hitgroup 记录。记录中的 `HitGroupData` 保存 GPU sphere 数组指针，`optixGetPrimitiveIndex()` 再选择具体球体。

## 4. Ray generation：每个像素发出一条射线

看 `device_programs.cu` 的 `__raygen__camera()`。

`optixLaunch(..., width, height, 1)` 启动二维 raygen 工作；`optixGetLaunchIndex()` 得到像素位置。本例使用正交相机：

```text
origin    = 对应像素的 (x, y, -3)
direction = (0, 0, 1)
ray(t)    = origin + t * direction
```

`optixTrace()` 的主要参数是：

| 参数 | 本例含义 |
|---|---|
| traversable | 前面建立的 GAS 句柄 |
| origin / direction | 射线起点与方向 |
| tmin / tmax | 只考虑这个 t 范围内的交点 |
| ray time | 本例为 0，无运动模糊 |
| visibility mask / flags | 可见性和遍历行为 |
| SBT offset / stride / miss index | 选择对应程序记录 |
| payload | 把命中结果返回给 raygen 的寄存器 |

这不同于自己启动普通 CUDA kernel 后手写 BVH 遍历：这里用 `optixLaunch()` 启动，raygen 内调用 `optixTrace()`，遍历由 OptiX 负责。

## 5. 精确求交：判断是否真的碰到球面

看 `__intersection__sphere()`。

OptiX 遍历到可能命中的 AABB 时，调用我们的 intersection 程序。读取当前球体后，求解：

```text
|origin + t * direction - center|² = radius²
```

如果二次方程判别式为负，则射线经过了包围盒但没有碰到球。否则求两个根：近交点和远交点。有效的根用 `optixReportIntersection(t, hitKind)` 报告给 OptiX。

代码先报告近交点；近交点不在有效范围时尝试远交点，因此也处理从球内出发的射线。不要仅判断 AABB 相交就报告“命中 primitive”。

对于 custom primitive，BVH 遍历可利用 RT 硬件，但这里的球面求交是你编写的设备程序，不能把整段自定义求交理解为自动转换成专用 RT 硬件操作。

## 6. Closest-hit 和 miss：返回结果

- `__closesthit__record()`：返回最近命中的 primitive ID 和交点参数 t。
- `__miss__background()`：返回未命中标记。

两个 payload 都是 32 位整数寄存器。t 是 float，所以使用 `__float_as_uint()` / `__uint_as_float()` 保存其位表示，而不是进行数值取整转换。

`optixTrace()` 返回后，raygen 写入：

- `RayResult` 数组：primitive ID 和 t。
- 像素数组：命中球体的颜色，或者背景颜色。

主机同步后下载结果，用独立 CPU 几何计算逐像素检查 primitive ID 和 t，然后写 PPM 图像和 TSV。CPU 检查会增加教学程序耗时，因此本例也不是 ray tracing 性能基准。

## 7. 完整路径

```text
CPU: Sphere → AABB → 上传 → optixAccelBuild → GAS handle
CPU: PTX → Module → Program groups → Pipeline + SBT
CPU: 上传 LaunchParams → optixLaunch

GPU: raygen → optixTrace → BVH traversal
                            ├─ AABB 候选 → intersection → 报告有效 t
                            ├─ 找到最近命中 → closest-hit
                            └─ 没有命中     → miss
GPU: raygen 接收 payload → 写结果和像素
CPU: 同步 → 下载 → CPU 对照 → 保存图像/TSV
```

## 8. 建议的修改顺序

原例子已在当前 container 中跑通。先用 `bash MVH_RT/scripts/run_optix.sh NEW_OUTPUT_DIRECTORY` 复现，再每次只改一项。

1. **改颜色**：修改 `makeScene()` 的 `color`，确认不影响 `rays.tsv` 的 ID/t，只改变图像。
2. **移动球体或改半径**：修改 `center/radius`；AABB 由同一组参数自动更新，再运行检查。
3. **增加第三个球体**：向 `makeScene()` 增加一项，观察 primitive ID；不需要手工增加树节点或 SBT 记录。
4. **改图像分辨率**：修改 `width/height`，观察 ray 数量与输出行数。
5. **改相机**：当前 CPU 对照利用了射线沿 +z 的条件。若改射线方向，需同步修改 CPU 对照，不能直接把对照失败当成 OptiX 错误。
6. **换 primitive**：之后再增加自己的 shape 数据、保守 AABB 和 intersection 方程。保留 raygen/pipeline 框架，先用小场景验证。

暂时不建议把 MVH 的 tolerance 直接当几何半径后宣称等价。峰匹配还有严格边界、等误差选择、多个离子复用实验峰、float/double 精度等语义；这些留到你熟悉基础 API 后单独设计。

官方资料：[OptiX 编程指南](https://raytracing-docs.nvidia.com/optix9/guide/index.html)、[NVIDIA OptiX headers](https://github.com/NVIDIA/optix-dev)。本例已使用 OptiX 9.0、container CUDA 12.8 与 610.57.04 runtime 完成实际编译和运行验证。
