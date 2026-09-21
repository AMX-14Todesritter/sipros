# CUDA 移植验证记录

本文记录首次直接移植版本。后续主机开销优化及当前版本的验证/耗时见 [OPTIMIZATION.md](OPTIMIZATION.md)。

日期：2026-09-20。基准提交：`aa1ebb7fe180707187a39a39f99ea07c39ac8dae`。
执行环境：现有 `sipros-sipros-1` container、GCC 11.4.0、CUDA 12.8、RTX 5070 Ti Laptop GPU；Release，CUDA architecture 120。没有安装依赖或改变容器配置。

## 正确性

四项 CTest 全部通过：

1. `mvh_original_source_identity`：24 份原始依赖源码身份验证。
2. `mvh_cli_and_sample`：样本 CPU 逐项验证、有效匹配、无匹配、跳过谱图、参数和输出保护。
3. `cuda_conversion_scope`：`mvh/` 文件哈希不变；提取类仅三处接入函数变化，其余方法保持一致；四处 OpenMP 线程循环被替换。
4. `cuda_semantic_contract`：严格 tolerance、等误差顺序、电荷 1–8、两种离子模型、末端 PTM、长肽段、重复峰与候选；逐项 CPU 参考比较。

`compute-sanitizer --tool memcheck --error-exitcode 99 --leak-check full` 对语义测试程序执行，结果 **0 errors、0 bytes leaked**。这不是对全部真实数据执行内存检查，也不是所有输入的形式化证明。

最终真实数据启用 `--verify-cuda`，对 46,066 张谱图进行预处理比较、对两批次实际评分尝试进行原 CPU 复算：

| 指标 | 结果 |
|---|---:|
| 实际评分尝试 | 37,899,128 |
| 成功评分 | 1,934,901 |
| 跳过 scan | 1 |
| 保留 PSM | 1,403,362 |
| 完整搜索（包含 CPU 逐项复算） | 62.0873 秒 |

最终 PSM SHA-256 与历史 CPU 基准完全一致：

```text
2f96a76b931b89c64054026533e4de5379fca959848a0a314c8ea42332f31520
```

详细日志：`output/mvh_cuda_validation_1Y8uew/final_verified.log`；内存检查：同目录 `memcheck.log`。初期失败日志保留作排查记录；它们不是最终结果。

开发过程中发现并修复了高电荷理论离子的浮点运算差异：CPU 编译器在不同分支使用倒数乘法或 FMA。设备端匹配本容器实际运算顺序后重新通过全量验证，未采用放宽评分容差或 CPU 静默回退。

## 正常运行耗时

`tests/validate.py --real --cpu-threads 4 --repeats 2` 的输出在 `output/mvh_cuda_real_comparison/`。关闭 CPU 数值复算，CPU 和 CUDA 顺序执行。

| 版本 | 次数 | 预处理（秒） | 完整搜索（秒） |
|---|---:|---:|---:|
| CPU，4 线程 | 1 | 1.0699 | 15.6476 |
| CUDA | 1 | 1.9234 | 30.7846 |
| CPU，4 线程 | 2 | 1.0760 | 15.8445 |
| CUDA | 2 | 2.0749 | 31.5646 |

这是功能移植的初步性能观察。第二轮 CPU 运行期间曾执行约 1.25 秒的 CTest，第二轮不作为隔离的性能基准。第一轮比较 CUDA 搜索约为 CPU 的 1.97 倍耗时；没有得到加速结论。后续正式性能实验应独占设备、增加重复次数并控制热状态。

第一轮 CUDA 搜索中，两批合计：主机打包 5.145 秒，GPU 服务 6.620 秒，主机结果恢复/检查 11.909 秒。其余时间包含数据库枚举、候选关联、肽段预处理、清理等；这里未进一步分解这些函数，不能将 GPU 服务时间称为纯 kernel 时间。

小样本对照在 `output/mvh_cuda_sample_comparison/`。对照脚本检查完整 TSV 字节相等，并保存输入/二进制哈希、命令、逐次 summary 和日志；仅在全部一致后压缩保留一份 PSM，删除同组重复 TSV。

## 覆盖范围与后续复验

已经验证当前真实数据、测试配置及上述合成边界场景。尚未系统覆盖所有可能的中性丢失规则、任意 mzML 数据、其他编译器和 GPU，不能把单一真实数据的一致性外推为所有输入保证。新数据/配置先使用 `--verify-cuda` 验证，再关闭该选项测量性能。

CPU 基准、原始输入和已有实验结果未覆盖。新增构建仅在 `build/mvh_cuda/`，新增结果使用独立目录。
