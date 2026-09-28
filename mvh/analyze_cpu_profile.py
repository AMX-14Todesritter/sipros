"""Summarize sampled CPU stacks with nonoverlapping, deepest-first categories."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

# Subcalls must precede their callers so that each sample is counted once.
CATEGORIES = [
    ('理论离子生成（含子调用）', 'MVH::CalculateSequenceIons('),
    ('峰搜索', 'PeakList::findNear('),
    ('MVH 评分函数其他工作', 'MVH::ScoreSequenceVsSpectrum('),
    ('历史 Top 序列合并', 'MS2Scan::mergePeptide('),
    ('Top candidate 保留', 'MS2Scan::saveScore('),
    ('评分外层循环及其他子调用', 'MS2Scan::scorePeptidesMVH('),
    ('肽段预处理', 'Peptide::preprocessingMVH('),
    ('肽段与 scan 关联', 'MvhScanVector::assignPeptides2Scans('),
    ('数据库肽段生成', 'ProteinDatabase::getNextPeptide('),
    ('scan 预处理', 'MvhScanVector::preProcessAllMs2Mvh('),
    ('批处理其他（含清理或并行管理）', 'MvhScanVector::processPeptideArrayMvh('),
    ('搜索其他', 'MvhScanVector::searchDatabaseMvh('),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    output = parser.parse_args().output
    manifest = json.loads((output/'manifest.json').read_text())
    with (output/'results/run_summary.tsv').open() as stream:
        summary = {r['metric']: r['value'] for r in csv.DictReader(stream, delimiter='\t')}
    groups, copies = Counter(), Counter()
    for line in (output/'stacks.folded').read_text().splitlines():
        stack, count = line.rsplit(' ', 1)
        count = int(count)
        category = next((label for label, name in CATEGORIES if name in stack), '加载、导出及其他')
        groups[category] += count
        if '__memcpy' in stack:
            copies[category] += count
    total = sum(groups.values())
    rows = [dict(category=k, samples=v, percent=100*v/total if total else 0)
            for k, v in groups.most_common()]
    analysis = {'total_samples': total, 'exclusive_categories': rows,
                'memcpy_samples_by_category': dict(copies),
                'note': 'Percent of sampled CPU stacks, not percent of wall time.'}
    (output/'analysis.json').write_text(json.dumps(analysis, indent=2, ensure_ascii=False)+'\n')
    wall, cpu = manifest['wall_seconds'], manifest['observed_cpu_seconds']
    lines = ['# CPU baseline profiling', '',
             f"线程数：{summary['omp_max_threads']}；CPU 批次上限：{summary['peptide_batch_size']} 条已关联肽段。",
             'CPU 批次口径与 GPU 的生成肽段批次不同。', '',
             '| 指标 | 数值 |', '|---|---:|',
             f'| 进程墙钟时间（含最多约一秒轮询误差） | {wall:.3f} 秒 |']
    for label, key in [('输入加载','config_and_load_seconds'), ('scan 预处理','preprocess_seconds'),
                       ('数据库搜索','search_seconds'), ('PSM 导出','export_seconds')]:
        lines.append(f'| {label} | {float(summary[key]):.3f} 秒 |')
    lines += [f'| 观察到的累计进程 CPU 时间（含工作线程） | {cpu:.2f} CPU 秒 |',
              f'| 平均有效 CPU 使用量 | {cpu/wall:.2f} 核 |',
              f"| 观察到的 VmHWM | {manifest['observed_hwm_kib']/1048576:.3f} GiB |",
              f"| 输出 PSM 行数 | {summary['retained_psm_count']} |", '',
              f'共 {total:,} 个采样。下表按调用栈归类，每个样本只计一次。',
              '**占比是采样 CPU 占比，不是墙钟时间占比，不能乘搜索秒数当作阶段耗时。**', '',
              '| 类别 | 样本数 | 占比 |', '|---|---:|---:|']
    lines += [f"| {r['category']} | {r['samples']:,} | {r['percent']:.2f}% |" for r in rows]
    lines += ['', '## 解读与限制', '',
              '评分外层包括内联代码、字符串复制、分配/释放和遍历，不能全部解释为纯循环成本。',
              'flat.txt 的 self 列表示函数自身热点；cumulative.txt 包含子函数，不能逐行相加。',
              '工作线程栈不包含主线程调用者，因此主线程 searchDatabaseMvh 的累计占比不是全部搜索成本。',
              '采样有开销；优化/内联、符号解析和信号采样偏差会影响细粒度归因。',
              '请求频率是 100 Hz 时也不保证每个 CPU 秒恰有 100 个样本，不把样本数换算为精确函数秒数。',
              '本报告没有 CPU 温度、频率或功耗记录，不能据此确认降频。', '',
              'memcpy 调用路径（已包含在上表中）：', '']
    lines += [f'- {name}：{count:,} 个样本。' for name, count in copies.most_common()]
    lines += ['', '原始报告：flat.txt、cumulative.txt、stacks.folded、callgrind.out。',
              'process_samples.csv 保存 CPU 使用量与内存时间序列；manifest.json 保存输入及工具哈希。', '',
              f"PSM SHA-256：`{manifest.get('psm_sha256', 'unavailable')}`。", '']
    (output/'ANALYSIS.md').write_text('\n'.join(lines))
    print('Analysis:', output/'ANALYSIS.md')


if __name__ == '__main__':
    main()
