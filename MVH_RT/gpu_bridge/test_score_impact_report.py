"""Tests for score/selection semantics independent of CUDA availability."""
import csv
import gzip
from pathlib import Path
import tempfile
import unittest

from score_impact_report import ScoreDeltas, compare_final, psm_groups, read_candidate_impact


class ScoreImpactReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_psms(self, name, scans):
        path = self.root / name
        fields = ['scan_index','scan_id','peptide','precursor_charge','precursor_mass','mvh_score','mvh_rank']
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            for index, candidates in scans.items():
                for rank, (peptide, score, charge) in enumerate(candidates, 1):
                    writer.writerow(dict(scan_index=index, scan_id=42, peptide=peptide,
                                         precursor_charge=charge, precursor_mass=1000,
                                         mvh_score=score, mvh_rank=rank))
        return path

    def test_score_shift_membership_ties_and_precursor_hypotheses(self):
        before = self.write_psms('before.tsv', {
            0:[('A',10,2),('B',8,2)], 1:[('A',10,2),('B',8,2)],
            2:[('A',10,2),('B',10,2)], 3:[('A',10,2)], 4:[('A',10,2)]})
        after = self.write_psms('after.tsv', {
            0:[('A',9,2),('B',8,2)], 1:[('C',11,2),('A',10,2)],
            2:[('B',10,2),('A',10,2)], 3:[('A',11,3)], 5:[('D',5,2)]})
        impacts = {index: {'score_changes':1,'eligibility_changes':0,'max_abs_delta':1}
                   for index in range(6)}
        counts, deltas, missing = compare_final(before, after, impacts, self.root, 1e-9)
        self.assertEqual(counts['top1_peptide_changed_scans'], 4)
        self.assertEqual(counts['top1_hypothesis_changed_scans'], 5)
        self.assertEqual(counts['topn_membership_changed_scans'], 3)
        self.assertEqual(counts['same_membership_reordered_scans'], 1)
        self.assertEqual(counts['top1_tie_order_only_scans'], 1)
        self.assertEqual(counts['common_peptide_hypothesis_changes'], 1)
        self.assertEqual(deltas['changed_exact'], 1)
        self.assertEqual(deltas['max_absolute_delta'], 1)
        self.assertEqual(missing, [])

    def test_unexplained_final_change_is_not_silently_accepted(self):
        before = self.write_psms('before.tsv', {0:[('A',10,2)]})
        after = self.write_psms('after.tsv', {0:[('B',11,2)]})
        _, _, missing = compare_final(before, after, {}, self.root, 0)
        self.assertEqual(missing, [0])

    def test_ineligible_score_is_not_zero_and_denominator_includes_unchanged(self):
        (self.root/'score_impact_batches.tsv').write_text(
            'batch\tcandidates\tboth_scored\tcuda_only_scored\trt_only_scored\tneither_scored\tscore_changed\tinvalid_results\n'
            '1\t10\t6\t1\t1\t2\t1\t0\n')
        (self.root/'candidate_score_changes.tsv').write_text(
            'scan_index\tcuda_status\trt_status\tcuda_score\trt_score\n'
            '0\t3\t3\t10\t8\n0\t3\t2\t20\t\n1\t2\t3\t\t5\n')
        totals, deltas, scans = read_candidate_impact(self.root, 1e-9)
        self.assertEqual(deltas['compared_valid_scores'], 6)
        self.assertAlmostEqual(deltas['mean_absolute_delta_all'], 2/6)
        self.assertEqual(deltas['max_absolute_delta'], 2)
        self.assertEqual(scans[0]['eligibility_changes'], 1)
        self.assertEqual(totals['rt_only_scored'], 1)

    def test_tolerance_and_gzip(self):
        deltas = ScoreDeltas(1e-9)
        for value in (0, 1e-12, 2):
            deltas.add(value)
        self.assertEqual(deltas.report()['changed_exact'], 2)
        self.assertEqual(deltas.report()['changed_above_tolerance'], 1)
        path = self.write_psms('data.tsv', {0:[('A',10,2)]})
        archive = path.with_suffix('.tsv.gz')
        with gzip.open(archive, 'wb') as stream:
            stream.write(path.read_bytes())
        self.assertEqual(list(psm_groups(path)), list(psm_groups(archive)))


if __name__ == '__main__':
    unittest.main()
