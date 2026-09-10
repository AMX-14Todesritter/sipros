# Precursor 中性质量分布

在现有容器中运行，使用已有 NumPy；SVG 绘图不依赖 matplotlib，不改搜索代码。

```bash
docker exec sipros-sipros-1 python3 /workspace/sipros/analysis/precursor_mass/plot_snapshot.py \
  /workspace/sipros/output/mvh_B_real_final_20260909/real/raw_t4/preprocessed.mvh \
  /workspace/sipros/output/precursor_mass_NEW --bin-da 100
```

输出目录必须不存在。输出 SVG、分箱 TSV、含源快照 SHA256 的 summary.json。
脚本按 snapshot.cpp 的 v1 布局读取，验证 CRC32、记录边界、precursor 排序及 scan 关联。

横轴是搜索使用的 precursor 中性质量，不是 m/z。每个柱是 100 Da 的左闭右开区间，颜色区分电荷。
统计非 skipped scan 的全部 precursor 假设，不按 scan 或质量去重；一张 scan 可被多次计数。
保留完整质量范围，不裁掉高质量尾部。分箱宽度不是匹配容差，也不代表理论碎片复用条件。

summary.json 分别提供：
- 全部假设的质量分位数。
- 全局排序后不同浮点质量之间的相邻差值；非常小的差值可能来自浮点表示，不代表仪器分辨率。
- 每张 scan 所有关联假设的最大质量减最小质量，再对这些跨度统计分位数。

质量分布不能单独证明理论碎片复用率；需要另行统计候选序列、修饰、电荷和配置的重复情况。
