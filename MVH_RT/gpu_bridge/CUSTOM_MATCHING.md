# RT-custom：前体质量分组、混合 class BVH 与完整命中归约

设计基线：分支 `rtMVH_v0.2`，提交 `a2b3bede990a928b259a0420aa5c07ac8635ef71`。
2026-10-07 版本替换上一版连续 scan 分组、class 分树和 CPU 共享任务排序。
前体 candidate 区间匹配仍使用原 CUDA 二分查找，不在本版改为 RT。

## 分组与几何

`--rt-scan-group-size K` 为每组不同 scan 的数量上限，任意正整数，默认 8；
可以大于数据集 scan 数以构建一组，末组允许不足 K。
每个 scan 用**所有有效前体假设中最小的中性质量**作为锚点，CUDA 求最小值、
GPU 稳定排序后每 K 个 scan 分组。相同锚点保持 scan 原顺序。
没有前体条目的 scan 锚点为 infinity，排在末尾；其本身不会产生 candidate。

这是单归属的折中：每张 scan 的几何只存一份，不按每条前体假设复制。
所有前体假设继续参与原 candidate 筛选；查询通过 candidate 的 scan→group 映射路由，
绝不只查肽质量对应的锚点桶，因而其他假设也不会丢失。
不能保证一个 scan 的每条假设都与同组 scan 的质量接近；锚点策略可作为后续研究参数。
当前 charge 是 candidate/theory 键及返回元数据，不作为 BVH 分组轴。

每组仅一棵 GAS，包含该组非 skip scan 的全部 class 1、2、3 峰，class 0 不入树。
GPU 计数、prefix sum 与打包；CPU 只读取组偏移并调用 OptiX 构建接口。
常驻几何与 GAS 跨肽批次复用。

仍在 double 中拆分峰 `n=floor(mz)`、`f=mz-n`，球心 `(float(n),float(f),0)`。
`f<=tolerance` 复制到 `(n-1,f+1,0)`；`1-f<=tolerance` 复制到 `(n+1,f-1,0)`。
所有副本保留原始全局 peak ID 与 scan ID，class 从原峰数组读取。
理论峰同样拆分，射线沿 −z，起点 z=0.5，t 区间 [0,1]。
半径 `nextafter(float(tolerance)+8*float_epsilon,+infinity)` 仅为粗筛 padding；
最后以原始 double 严格判断 `abs(theory-experiment)<tolerance`。
该表示不是超出 float 精确整数范围等任意 m/z 数据的数学精度保证。

## GPU 任务与 batch 内理论峰去重

CUB 稳定排序 candidate 索引，键为 `(peptideId,charge,groupId)`。
candidate 本体及原顺序不改，不下载到 CPU 排序。
键编码为 64 位：peptide 32 位，剩余 32 位动态分给 charge 和 group；
超出编码容量时明确报错，不截断身份。
CUDA 相邻键比较与 GPU prefix sum 生成共享任务，只涉及实际有 candidate 的组。

按排序后的 `peptideId+charge` 连续键分块，默认候选 tile 65536；
末尾延伸到该键结束，避免同一键跨块重复生成理论峰。
CUDA 为每个键独立计数、prefix sum、生成，组任务引用 compact 理论数组。
生产 RT-custom 不再尝试原 dense theory cache；契约测试仍可提供外部缓存验证读取语义。
不跨 peptide 对象 ID 去重，也不按近似 m/z 合并理论峰。
所有理论峰序号与重复理论峰的计数保持原数值模板的语义。
理论峰生成不嵌入 OptiX raygen，沿用此前非法访问路径的处理经验。

## 完整 any-hit 收集与 CUDA 归约

每个共享组任务读取其理论峰数组。第一次 OptiX 遍历只统计原始几何命中数；
GPU prefix sum 后分配命中空间，第二次相同查询写入 `(theoryPeakIndex,scanId,peakId)`。
回调始终 `optixIgnoreIntersection()`，不接受交点缩短查询区间，也不提前终止。
OptiX 不参与最佳峰选择或 MVH。

CUDA 检查计数与写入遍历一致；越界或计数不一致明确失败，拒绝部分结果。
完整收集之后过滤不相关 scan、skip、有效 m/z 范围及严格 double 容差。
每个 `(theoreticalPeakIndex,scanId)` 维护一个 winner：

1. class 优先 **3→2→1**（用户指定）；
2. 同 class 选择原 double 质量距离最小；
3. 距离相同选择原始 peak ID 最小。

重复 any-hit 调用和 split 副本指向相同 peak ID，幂等选择不重复计分。
不同理论峰序号可复用同一实验峰；相同 m/z 的不同理论峰仍分别计数。
归约完成后 CUDA 计算 predicted、matched、未命中 histogram 和 MVH，回填原 candidate。
原 Top 更新、相同序列合并及蛋白归属恢复保留原顺序。

## 显存与日志

`--rt-workspace-mib N`，默认 512，最小 16，控制局部 theory/task/hit/winner 工作区。
理论块按预算减小；命中及 winner 矩阵按实际 prefix 数量进一步拆分组任务。
单键理论或单组任务仍超出预算时明确失败，提示降低 K 或增大工作区。
此预算不包含常驻谱图/GAS、原 candidate/Result、整批排序键索引及 CUB/Thrust scratch，
不代表整个程序显存峰值；整批 GPU 排序的峰值仍可能触发显存不足。

`[RT shared tasks]` 记录 candidate、组任务、去重 theory 键、生成理论峰、原始命中数、
store launch 数及局部工作区估计高水位。
`collected_hits` 为 store 遍历所需条目数，含无关 scan 和重复几何，非最终匹配数。
`workspace_max_bytes` 不含临时库 scratch、排序和原程序常驻资源。
`[RT stages]` 给出排序、理论/任务整理、两次 trace 收集、归约评分的同步墙钟时间。
阶段之和不等于整段搜索时间，也不包含原 candidate 分配/排序、消化、Top 与导出。
`run_summary.tsv` 记录 K、min-neutral-mass 分组、class 优先级和 workspace MiB。

## 验证与运行

```sh
docker exec -w /workspace/sipros sipros-sipros-1 cmake --build build/mvh_rt/gpu_integration -j4
docker exec -w /workspace/sipros -e LD_LIBRARY_PATH=/workspace/sipros/build/mvh_rt/optix_runtime:/usr/local/cuda/lib64 sipros-sipros-1 ctest --test-dir build/mvh_rt/gpu_integration --output-on-failure
```

`--verify-cuda` 为 custom 使用独立 CPU 穷举 class 优先参考；并保留预处理、candidate、
Top/protein 等验证。其他后端仍使用原规则。
契约覆盖 K1/2/8/32/128、非 scan 顺序质量分组、多个假设、混合 class、近距/等距峰身份、
严格边界、split 副本、重复理论峰与 candidate、skip/range、跨块映射及缓冲不足重试。
集成比较跨 K/批次的逐字节 PSM，含无缓存路径。

完整 Marine 示例（仅一次）：

```sh
docker exec -w /workspace/sipros sipros-sipros-1 python3 -B MVH_RT/gpu_bridge/benchmark.py \
  --root /workspace/sipros --fasta /workspace/sipros/raw/Marine_fw_3rev.fasta \
  --batch 8000000 --backends rt-custom --rt-scan-group-size 64 --rt-workspace-mib 512 --repeats 1
```

K64 是本次试验点，不是已确定最优值。旧故障记录见
`output/validation/shared_anyhit/ROOT_CAUSE.md`，旧性能报告不描述本版实现。
