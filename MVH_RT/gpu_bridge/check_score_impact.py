"""Run the reusable score-impact workflow on the checked-in small sample."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--binary', type=Path, required=True)
args = parser.parse_args()
script = Path(__file__).with_name('validate_score_impact.py')
with tempfile.TemporaryDirectory(prefix='score-impact-check-') as temporary:
    output = Path(temporary) / 'run'
    subprocess.run([sys.executable, '-B', str(script), '--dataset', 'smoke',
                    '--binary', str(args.binary), '--peptide-batch-size', '3',
                    '--output', str(output)], check=True)
    reports = json.loads((output / 'analysis/summary.json').read_text())
    for report in reports.values():
        assert report['candidate_occurrences']['candidates'] > 0
        assert report['candidate_occurrences']['score_changed'] == 0
        assert report['final_selection']['top1_peptide_changed_scans'] == 0
        assert report['final_selection']['topn_membership_changed_scans'] == 0
    # Confirm saved compressed PSMs suffice; this must not rerun searches.
    subprocess.run([sys.executable, '-B', str(script), '--analyze-only', str(output)], check=True)
print('PASS: normal/diagnostic PSM equality, score reports, compressed replay without searches')
