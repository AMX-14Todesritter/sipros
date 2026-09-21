# 独立类提取验证（2026-09-17）

按用户要求，将 MS2ScanVector 中 MVH 所需部分提取为 MvhScanVector。

- 新类声明：include/mvh_scan_vector.h。
- 实现分为 src/mvh_scan_vector.cpp（生命周期/工作区）、src/spectrum_input.cpp（输入/预处理）、src/database_search.cpp（查询/搜索）。
- 提取 21 个方法定义，除类名（包含构造/析构）替换外与来源逐字一致；普通函数名全部保留。
- 类声明改变访问权限、去掉非 MVH 工作流声明和成员；vpAllMS2Scans 保留原名供结果读取。
- 原 MS2ScanVector 实现不再参与编译，original/ 中该文件仅作来源对照。其余原始依赖仍按原文件编译。
- app/runner.cpp 移除 access、Preprocess、Search、Scans 模板适配；直接调用原名方法。

三项 CTest 全部通过：原始依赖身份、有效匹配/PTM/线程一致/无匹配/跳过/CLI、提取方法定义一致性。

容器内四线程真实数据：46,066 scans，跳过 1，保留 1,403,362 PSM；搜索耗时 16.4565 秒（单次诊断，不是加速结论）。输出 SHA-256 与历史参考一致：

```
2f96a76b931b89c64054026533e4de5379fca959848a0a314c8ea42332f31520
```

原始日志和完整命令：output/mvh_class_validation_20260917_182107/。
相同 PSM 数据已经保存在 output/reference/historical_mvh_psms.tsv.gz，因此本轮验证只保留日志和摘要，不再保留重复的大型明文 TSV。

权限修复：此前通过容器 root 创建 mvh 文件，编辑器 ams098z 无写权限；将 mvh/ 的所有权恢复为 1000:1000，文件保留 644、目录 755。没有修改容器配置或开放全员写权限。
