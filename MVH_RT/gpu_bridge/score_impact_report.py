"""Compare MVH score changes and final retention; no peak-level accuracy metric."""
import csv
import gzip
from collections import defaultdict
from itertools import groupby
import json
import math
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class ScoreDeltas:
    """Keep nonzero deltas only; the denominator also includes identical scores."""
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.compared = 0
        self.changed = []

    def add(self, delta):
        if not math.isfinite(delta):
            raise ValueError('Non-finite score delta')
        self.compared += 1
        if delta != 0:
            self.changed.append(delta)

    def report(self):
        absolute = [abs(value) for value in self.changed]
        return {
            'compared_valid_scores': self.compared,
            'changed_exact': len(absolute),
            'changed_above_tolerance': sum(value > self.tolerance for value in absolute),
            'mean_signed_delta_all': sum(self.changed) / self.compared if self.compared else None,
            'mean_absolute_delta_all': sum(absolute) / self.compared if self.compared else None,
            'rmse_all': math.sqrt(sum(value * value for value in absolute) / self.compared) if self.compared else None,
            'median_absolute_delta_changed': percentile(absolute, 0.5),
            'p95_absolute_delta_changed': percentile(absolute, 0.95),
            'max_absolute_delta': max(absolute, default=0),
            'min_signed_delta_changed': min(self.changed, default=None),
            'max_signed_delta_changed': max(self.changed, default=None),
            'absolute_delta_bins': {
                f'({lower}, {upper}]': sum(lower < value <= upper for value in absolute)
                for lower, upper in [(0, 1e-9), (1e-9, 1e-6), (1e-6, 1e-3),
                                     (1e-3, 0.01), (0.01, 0.1), (0.1, 1), (1, 10)]
            } | {'>10': sum(value > 10 for value in absolute)},
        }


def read_candidate_impact(directory, tolerance):
    totals = defaultdict(int)
    with (directory / 'score_impact_batches.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            for name, value in row.items():
                if name != 'batch':
                    totals[name] += int(value)
    if totals['invalid_results']:
        raise ValueError('Invalid candidate scoring results; comparison is incomplete')
    partition = sum(totals[name] for name in ('both_scored', 'cuda_only_scored', 'rt_only_scored', 'neither_scored'))
    if partition != totals['candidates']:
        raise ValueError('Candidate denominator does not match scoring categories')
    deltas = ScoreDeltas(tolerance)
    scans = defaultdict(lambda: {'score_changes': 0, 'eligibility_changes': 0, 'max_abs_delta': 0.0})
    lost = gained = 0
    with (directory / 'candidate_score_changes.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            left, right = int(row['cuda_status']) == 3, int(row['rt_status']) == 3
            scan = scans[int(row['scan_index'])]
            if left and right:
                delta = float(row['rt_score']) - float(row['cuda_score'])
                deltas.add(delta)
                scan['score_changes'] += 1
                scan['max_abs_delta'] = max(scan['max_abs_delta'], abs(delta))
            elif left != right:
                scan['eligibility_changes'] += 1
                lost += left
                gained += right
            else:
                raise ValueError('Unexpected unscored status change')
    if len(deltas.changed) != totals['score_changed'] or lost != totals['cuda_only_scored'] or gained != totals['rt_only_scored']:
        raise ValueError('Candidate detail rows do not reconcile with GPU counters')
    deltas.compared = totals['both_scored']
    return dict(totals), deltas.report(), scans


def psm_groups(path):
    """Stream by scan_index, which stays unique even when scan IDs repeat."""
    previous = -1
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        rows = csv.DictReader(stream, delimiter='\t')
        for scan_index, group in groupby(rows, key=lambda row: int(row['scan_index'])):
            if scan_index <= previous:
                raise ValueError('PSM file is not grouped in increasing scan_index order')
            previous = scan_index
            candidates = list(group)
            for rank, candidate in enumerate(candidates, 1):
                candidate['score'] = float(candidate['mvh_score'])
                candidate['rank'] = int(candidate['mvh_rank'])
                if candidate['rank'] != rank or not math.isfinite(candidate['score']):
                    raise ValueError('Invalid rank or score in PSM file')
            if len({row['peptide'] for row in candidates}) != len(candidates):
                raise ValueError('Duplicate final peptide identity within a scan')
            if any(a['score'] < b['score'] for a, b in zip(candidates, candidates[1:])):
                raise ValueError('Final candidate list is not sorted by score')
            yield scan_index, candidates


def merge_scans(left_path, right_path):
    left, right = iter(psm_groups(left_path)), iter(psm_groups(right_path))
    a, b = next(left, None), next(right, None)
    while a is not None or b is not None:
        index = min(value[0] for value in (a, b) if value is not None)
        left_rows = a[1] if a is not None and a[0] == index else []
        right_rows = b[1] if b is not None and b[0] == index else []
        yield index, left_rows, right_rows
        if left_rows:
            a = next(left, None)
        if right_rows:
            b = next(right, None)


def identity(row):
    if row is None:
        return None
    return row['peptide'], int(row['precursor_charge']), float(row['precursor_mass'])


def top_ties(rows, tolerance):
    return {row['peptide'] for row in rows if rows[0]['score'] - row['score'] <= tolerance} if rows else set()


def margin(rows):
    return rows[0]['score'] - rows[1]['score'] if len(rows) > 1 else None


def compare_final(reference, observed, candidate_scans, output, tolerance):
    counts = defaultdict(int)
    deltas = ScoreDeltas(tolerance)
    affected = set(candidate_scans)
    top1_changed_scans, membership_changed_scans = set(), set()
    unexplained = []
    scan_fields = ['scan_index', 'scan_id', 'cuda_count', 'rt_count', 'cuda_top1', 'rt_top1',
                   'cuda_top1_score', 'rt_top1_score', 'cuda_margin', 'rt_margin',
                   'top1_changed', 'top1_hypothesis_changed', 'top1_tie_order_only',
                   'membership_changed', 'order_changed', 'removed_peptides', 'added_peptides',
                   'candidate_score_changes', 'candidate_eligibility_changes', 'max_candidate_abs_delta']
    retained_fields = ['scan_index', 'scan_id', 'peptide', 'cuda_rank', 'rt_rank',
                       'cuda_score', 'rt_score', 'score_delta', 'hypothesis_changed']
    with (output / 'scan_changes.tsv').open('w') as scan_file, (output / 'retained_score_changes.tsv').open('w') as retained_file:
        scan_writer = csv.DictWriter(scan_file, fieldnames=scan_fields, delimiter='\t')
        retained_writer = csv.DictWriter(retained_file, fieldnames=retained_fields, delimiter='\t')
        scan_writer.writeheader()
        retained_writer.writeheader()
        for index, left, right in merge_scans(reference, observed):
            counts['scans_with_any_final_candidate'] += 1
            counts['scans_with_cuda_candidates'] += bool(left)
            counts['scans_with_rt_candidates'] += bool(right)
            counts['cuda_retained_candidates'] += len(left)
            counts['rt_retained_candidates'] += len(right)
            a, b = {row['peptide']: row for row in left}, {row['peptide']: row for row in right}
            left_order, right_order = list(a), list(b)
            left_top, right_top = left[0] if left else None, right[0] if right else None
            top1_changed = (left_top['peptide'] if left_top else None) != (right_top['peptide'] if right_top else None)
            hypothesis_changed = identity(left_top) != identity(right_top)
            membership_changed = set(a) != set(b)
            order_changed = left_order != right_order
            ties_left, ties_right = top_ties(left, tolerance), top_ties(right, tolerance)
            tie_order_only = top1_changed and ties_left == ties_right and len(ties_left) > 1
            for name, changed in [('top1_peptide_changed_scans', top1_changed),
                                  ('top1_hypothesis_changed_scans', hypothesis_changed),
                                  ('top1_tie_order_only_scans', tie_order_only),
                                  ('top1_tie_set_changed_scans', ties_left != ties_right),
                                  ('topn_membership_changed_scans', membership_changed),
                                  ('topn_order_changed_scans', order_changed),
                                  ('same_membership_reordered_scans', order_changed and not membership_changed)]:
                counts[name] += changed
            if top1_changed:
                top1_changed_scans.add(index)
            if membership_changed:
                membership_changed_scans.add(index)
            if (hypothesis_changed or order_changed) and index not in affected:
                unexplained.append(index)
            counts['removed_peptides'] += len(a.keys() - b.keys())
            counts['added_peptides'] += len(b.keys() - a.keys())
            changed_retained = False
            for peptide in sorted(a.keys() & b.keys()):
                old, new = a[peptide], b[peptide]
                same_hypothesis = identity(old) == identity(new)
                if same_hypothesis:
                    deltas.add(new['score'] - old['score'])
                else:
                    counts['common_peptide_hypothesis_changes'] += 1
                if new['score'] != old['score'] or not same_hypothesis:
                    changed_retained = True
                    retained_writer.writerow({'scan_index': index, 'scan_id': old['scan_id'],
                        'peptide': peptide, 'cuda_rank': old['rank'], 'rt_rank': new['rank'],
                        'cuda_score': old['score'], 'rt_score': new['score'],
                        'score_delta': new['score'] - old['score'] if same_hypothesis else '',
                        'hypothesis_changed': int(not same_hypothesis)})
            impact = candidate_scans.get(index, {'score_changes': 0, 'eligibility_changes': 0, 'max_abs_delta': 0})
            if index in affected or order_changed or hypothesis_changed or changed_retained:
                scan_writer.writerow({'scan_index': index, 'scan_id': (left_top or right_top)['scan_id'],
                    'cuda_count': len(left), 'rt_count': len(right),
                    'cuda_top1': left_top['peptide'] if left_top else '',
                    'rt_top1': right_top['peptide'] if right_top else '',
                    'cuda_top1_score': left_top['score'] if left_top else '',
                    'rt_top1_score': right_top['score'] if right_top else '',
                    'cuda_margin': margin(left), 'rt_margin': margin(right),
                    'top1_changed': int(top1_changed), 'top1_hypothesis_changed': int(hypothesis_changed),
                    'top1_tie_order_only': int(tie_order_only), 'membership_changed': int(membership_changed),
                    'order_changed': int(order_changed), 'removed_peptides': json.dumps(sorted(a.keys()-b.keys())),
                    'added_peptides': json.dumps(sorted(b.keys()-a.keys())),
                    'candidate_score_changes': impact['score_changes'],
                    'candidate_eligibility_changes': impact['eligibility_changes'],
                    'max_candidate_abs_delta': impact['max_abs_delta']})
    counts['score_or_eligibility_affected_scans'] = len(affected)
    counts['affected_scans_with_unchanged_top1'] = len(affected - top1_changed_scans)
    counts['affected_scans_with_unchanged_topn_membership'] = len(affected - membership_changed_scans)
    return dict(counts), deltas.report(), unexplained


def analyze(reference, observed, diagnostic, output, total_scans, tolerance=1e-9):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    totals, candidate_deltas, affected = read_candidate_impact(Path(diagnostic), tolerance)
    selection, retained_deltas, unexplained = compare_final(reference, observed, affected, output, tolerance)
    report = {'score_tolerance': tolerance, 'total_scans': total_scans,
              'candidate_occurrences': totals, 'candidate_score_deltas': candidate_deltas,
              'final_selection': selection, 'common_retained_hypothesis_score_deltas': retained_deltas,
              'selection_changes_without_candidate_score_or_status_changes': unexplained,
              'interpretation': 'CUDA bucket matching is the computational reference, not biological ground truth. Candidate statistics are pre-merge occurrences; final selection comes from separate uninstrumented complete searches. Blank scores mean ineligible, not zero. Delta is RT minus CUDA. A changed precursor hypothesis is not treated as a same-candidate score delta.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
