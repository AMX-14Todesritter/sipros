#!/usr/bin/env python3
"""Validate RT's MVH-score and final-candidate impact against CUDA bucket matching."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.output_paths import resolve_output
from score_impact_report import analyze

BACKENDS = ('rt-triangle', 'rt-instanced')


def digest(path):
    checksum = hashlib.sha256()
    opener = gzip.open(path, 'rb') if str(path).endswith('.gz') else Path(path).open('rb')
    with opener as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def psm_path(directory):
    plain = directory / 'mvh_psms.tsv'
    return plain if plain.exists() else directory / 'mvh_psms.tsv.gz'


def read_summary(directory):
    with (directory / 'run_summary.tsv').open() as stream:
        return dict(list(csv.reader(stream, delimiter='\t'))[1:])


def write_manifest(output, manifest):
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def run_search(binary, inputs, output, backend, batch_size, diagnostics, environment):
    command = [str(binary), '-f', str(inputs['scans']), '-c', str(inputs['config']),
               '-fasta', str(inputs['fasta']), '-o', str(output),
               '--match-backend', backend, '--peptide-batch-size', str(batch_size)]
    if diagnostics:
        command.append('--score-impact')
    print('RUN:', output.name, flush=True)
    with output.with_suffix('.log').open('w') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {'command': command, 'psm_sha256': digest(psm_path(output)), 'summary': read_summary(output)}


def compress_psms(output, manifest):
    """Validate compressed bytes before removing only this run's plain copies."""
    for name, run in manifest['runs'].items():
        source = output / name / 'mvh_psms.tsv'
        if not source.exists():
            continue
        target = source.with_suffix('.tsv.gz')
        with source.open('rb') as reader, gzip.open(target, 'wb') as writer:
            shutil.copyfileobj(reader, writer)
        if digest(target) != run['psm_sha256']:
            raise RuntimeError(f'Compressed PSM verification failed: {target}')
        source.unlink()


def percent(value, total):
    return f'{100 * value / total:.4f}%' if total else '—'


def write_reports(output, manifest, tolerance, repeat=False):
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    analysis_root = output / (f'analysis_{timestamp}' if repeat else 'analysis')
    analysis_root.mkdir(exist_ok=False)
    total_scans = int(manifest['runs']['cuda']['summary']['scan_count'])
    reports = {}
    for backend in manifest['backends']:
        if backend not in BACKENDS:
            raise ValueError(f'Unsupported stored backend: {backend}')
        diagnostic = backend + '_diagnostic'
        for name in ('cuda', backend, diagnostic):
            if digest(psm_path(output / name)) != manifest['runs'][name]['psm_sha256']:
                raise ValueError(f'Saved PSM changed: {name}')
            if int(manifest['runs'][name]['summary']['scan_count']) != total_scans:
                raise ValueError('Searches used inconsistent scan counts')
        if manifest['runs'][backend]['psm_sha256'] != manifest['runs'][diagnostic]['psm_sha256']:
            raise ValueError(f'Diagnostics changed final selection for {backend}; cannot use this comparison')
        report = analyze(psm_path(output / 'cuda'), psm_path(output / backend),
                         output / diagnostic, analysis_root / backend, total_scans, tolerance)
        reports[backend] = report
    lines = ['# RT 对 MVH 分数及最终候选筛选的影响', '',
             f'输入：{total_scans:,} 个 scan；FASTA：`{manifest["inputs"]["fasta"]}`。',
             f'同分判定容差：{tolerance:g} MVH 分数单位。原始分数差异另外按精确不等统计。', '',
             '参照为 CUDA 桶匹配，不是生物学真值。下表衡量计算结果变化，不是逐 peak 准确率。', '',
             '## 最终筛选', '',
             '| RT 后端 | top-1 肽段改变 / 全部 scan | top-1 完整假设改变 | top-N 成员改变 | 成员相同但重新排序 | top-1 仅同分顺序变化 |',
             '|---|---:|---:|---:|---:|---:|']
    for backend, report in reports.items():
        selection = report['final_selection']
        count = selection['top1_peptide_changed_scans']
        lines.append(f'| {backend} | {count:,} / {total_scans:,} ({percent(count,total_scans)}) | '
                     f'{selection["top1_hypothesis_changed_scans"]:,} | {selection["topn_membership_changed_scans"]:,} | '
                     f'{selection["same_membership_reordered_scans"]:,} | {selection["top1_tie_order_only_scans"]:,} |')
    lines += ['', 'top-N 指程序最终保留的完整列表（当前容量 50），按 scan_index 配对。完整假设包含肽段、前体电荷和前体质量。成员比较按肽段身份；同一肽段换了前体假设会另外计数，不混入相同候选的分数差。', '',
              '## 候选评分变化（merge/top 之前）', '',
              '| 后端 | 候选出现次数 | 双方均可评分 | 分数精确改变 | 超过容差 | CUDA 可评分而 RT 不可评分 | RT 新增可评分 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for backend, report in reports.items():
        counts, deltas = report['candidate_occurrences'], report['candidate_score_deltas']
        lines.append(f'| {backend} | {counts["candidates"]:,} | {counts["both_scored"]:,} | '
                     f'{counts["score_changed"]:,} | {deltas["changed_above_tolerance"]:,} | '
                     f'{counts["cuda_only_scored"]:,} | {counts["rt_only_scored"]:,} |')
    lines += ['', '| 后端 | 双方可评分候选的平均绝对差（含零差） | 改变候选的绝对差中位数 | 改变候选的绝对差 P95 | 最大绝对差 |',
              '|---|---:|---:|---:|---:|']
    for backend, report in reports.items():
        d = report['candidate_score_deltas']
        values = [d[key] for key in ('mean_absolute_delta_all','median_absolute_delta_changed','p95_absolute_delta_changed','max_absolute_delta')]
        lines.append('| '+backend+' | '+' | '.join('—' if value is None else f'{value:.9g}' for value in values)+' |')
    lines += ['', '候选出现次数包括跨蛋白重复候选及之后可能被合并的候选，不能直接解释成最终鉴定数。未达到匹配数门槛时没有有效 MVH 分数，明细留空而非将它当成 0 分。差值定义为 RT − CUDA。', '',
              '## 从分数差异到筛选影响', '']
    for backend, report in reports.items():
        c = report['final_selection']
        lines.append(f'- {backend}：{c["score_or_eligibility_affected_scans"]:,} 个 scan 存在候选分数/评分资格变化；其中 '
                     f'{c["affected_scans_with_unchanged_top1"]:,} 个 top-1 肽段不变，'
                     f'{c["affected_scans_with_unchanged_topn_membership"]:,} 个最终 top-N 成员不变。')
    unexplained = {backend: report['selection_changes_without_candidate_score_or_status_changes']
                   for backend, report in reports.items()}
    lines += ['', '## 复查文件', '',
              '- `../<backend>_diagnostic/candidate_score_changes.tsv`：每个发生评分或资格变化的候选出现实例。',
              '- `<backend>/scan_changes.tsv`：关联每个 scan 的候选分数变化、top-1 分差、成员进入/退出及排序变化。',
              '- `<backend>/retained_score_changes.tsv`：最终共同保留候选的分数变化、排名及前体假设变化。',
              '- `<backend>/report.json`：完整计数、分数差分布和分母。', '',
              '诊断模式额外计算 CUDA 分数并复制差异明细，不能用于性能比较。每个 RT 后端另跑不启用诊断的完整搜索，并要求两次最终 PSM 的 SHA-256 完全一致。分数变化与筛选变化在 scan 层关联；不声称某一重复候选事件必然导致该 scan 的最终变化。', '',
              f'缺少候选分数/资格变化支持的最终筛选变化：{sum(len(value) for value in unexplained.values())} 个 scan。']
    (analysis_root / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    (analysis_root / 'summary.json').write_text(json.dumps(reports, indent=2)+'\n')
    if any(unexplained.values()):
        raise ValueError('Unexplained final-selection changes; inspect report before drawing conclusions')
    print('REPORT:', analysis_root / 'REPORT.md', flush=True)
    return analysis_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ecoli', 'smoke'), default='ecoli')
    parser.add_argument('--scans', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--fasta', type=Path)
    parser.add_argument('--binary', type=Path, default=ROOT/'build/mvh_rt/gpu_integration/bin/sipros_mvh_cuda')
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument('--peptide-batch-size', type=int, default=1000000,
                        help='Generated peptides per batch; default 1000000 bounds diagnostic memory')
    parser.add_argument('--score-tolerance', type=float, default=1e-9)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--keep-psms', action='store_true', help='Keep uncompressed PSMs')
    parser.add_argument('--analyze-only', type=Path, help='Regenerate reports from a completed run, without executing searches')
    args = parser.parse_args()
    if not math.isfinite(args.score_tolerance) or args.score_tolerance < 0:
        parser.error('score tolerance must be finite and nonnegative')
    if args.analyze_only:
        manifest = json.loads((args.analyze_only/'manifest.json').read_text())
        if not manifest.get('searches_complete'):
            parser.error('Stored searches are incomplete')
        write_reports(args.analyze_only, manifest, args.score_tolerance, repeat=True)
        return
    if args.peptide_batch_size < 1 or len(set(args.backends)) != len(args.backends):
        parser.error('batch size must be positive and backends unique')
    data = ROOT/'mvh_cuda/tests/data'
    smoke = args.dataset == 'smoke'
    inputs = {'scans': args.scans or (data/'sample.ft2' if smoke else ROOT/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2'),
              'config': args.config or (data/'search.cfg' if smoke else ROOT/'experiments/Regular.cfg'),
              'fasta': args.fasta or (data/'proteins.fasta' if smoke else ROOT/'raw/Ecoli.fasta')}
    inputs = {name: path.resolve() for name, path in inputs.items()}
    binary = args.binary.resolve()
    for path in [binary, *inputs.values()]:
        if not path.is_file():
            parser.error(f'Missing input: {path}')
    output = resolve_output(args.output, 'validation', 'score_impact', args.dataset)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'inputs': {name: str(path) for name,path in inputs.items()},
                'input_sha256': {name: digest(path) for name,path in inputs.items()},
                'binary': str(binary), 'binary_sha256': digest(binary),
                'backends': args.backends, 'score_tolerance': args.score_tolerance,
                'peptide_batch_size': args.peptide_batch_size, 'runs': {}, 'searches_complete': False}
    environment = dict(os.environ, LD_LIBRARY_PATH=str(ROOT/'build/mvh_rt/optix_runtime')+':/usr/local/cuda/lib64')
    write_manifest(output, manifest)
    for name, backend, diagnostic in [('cuda','cuda',False)] + [
            item for backend in args.backends for item in ((backend,backend,False),(backend+'_diagnostic',backend,True))]:
        manifest['runs'][name] = run_search(binary, inputs, output/name, backend,
                                          args.peptide_batch_size, diagnostic, environment)
        write_manifest(output, manifest)
    manifest['searches_complete'] = True
    write_manifest(output, manifest)
    write_reports(output, manifest, args.score_tolerance)
    if not args.keep_psms:
        compress_psms(output, manifest)
    print('PASS: independent final searches, score diagnostics, final-selection comparison', flush=True)


if __name__ == '__main__':
    main()
