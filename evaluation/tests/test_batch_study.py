"""Coordinator integration tests using a fake child runner; no GPU required."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('batch_study', SCRIPTS / 'batch_study.py')
study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)
FAKE = r'''
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser()
for k in ['id','batch','backends']:p.add_argument('--'+k,required=True)
a,rest=p.parse_known_args()
d=Path(os.environ['TEST_OUT'])/'results/batch_size'/a.id;d.mkdir(parents=True,exist_ok=False)
if os.environ.get('TEST_PAUSE'):
 target=Path(os.environ['TEST_PAUSE']);target.touch()
fail=str(a.batch)==os.environ.get('FAIL_BATCH')
r={'name':a.backends+'_r1','backend':a.backends,'status':'completed','warmup':False,
'batches':[{'cached_ions':os.environ.get('BAD_CACHE','0'),'cache_charge_stride':'0'}],
'rt_setup_seconds':.1,'totals':{'fused_device_seconds':1.,'pack_seconds':2.,'gpu_service_seconds':3.,'replay_verify_seconds':4.,'candidates':10},
'summary':{'search_seconds':'12'},'wall_seconds':13.,'generated_peptides':20,
'device_peak_bytes':100,'host_observed_hwm_bytes':200,'psm_sha256':a.backends}
n={'status':'failed' if fail else 'completed','error':'Synthetic capacity failure' if fail else '', 'diagnostic':False,'runs':[r]}
(d/'manifest.json').write_text(json.dumps(n))
raise SystemExit(1 if fail else 0)
'''


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / 'output/evaluation'
        self.runner = self.root / 'fake.py'; self.runner.write_text(FAKE)
        for n in ['scans', 'fasta', 'cfg', 'binary']:(self.root / n).write_text('fixture')
        self.patches = [patch.object(study, 'ROOT', self.root), patch.object(study, 'OUT', self.out),
                        patch.object(study, 'RUNNER', self.runner), patch.dict(os.environ, {'TEST_OUT': str(self.out)})]
        for p in self.patches:p.start()
        self.args = ['--id','test','--scans',str(self.root/'scans'),'--fasta',str(self.root/'fasta'),
                     '--config',str(self.root/'cfg'),'--gpu-binary',str(self.root/'binary'),
                     '--batches','16','20','24','--backends','cuda','rt-custom']

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def manifest(self):return json.loads((self.out/'batch_studies/test/manifest.json').read_text())

    def test_order_statistics_and_completed_resume(self):
        self.assertEqual(study.main(self.args+['--repeats','2']),0)
        m=self.manifest();self.assertEqual(len(m['jobs']),12)
        self.assertEqual([j['backend'] for j in m['jobs'][:4]],['cuda','rt-custom','rt-custom','cuda'])
        self.assertEqual(m['jobs'][0]['metrics']['complete_scoring_seconds'],9)
        before=list((self.out/'results/batch_size').iterdir())
        self.assertEqual(study.main(['--id','test','--resume']),0)
        self.assertEqual(len(before),len(list((self.out/'results/batch_size').iterdir())))
        text=(self.out/'batch_studies/test/progress.md').read_text();self.assertIn('completed',text)
        self.assertIn('fused_seconds_mean',(self.out/'batch_studies/test/summary.csv').read_text())

    def test_stop_and_retry_preserves_attempts(self):
        with patch.dict(os.environ,{'FAIL_BATCH':'20'}):self.assertEqual(study.main(self.args),1)
        m=self.manifest();self.assertEqual([j['status'] for j in m['jobs']],['completed','completed','failed','pending','pending','pending'])
        failed_dir=Path(m['jobs'][2]['attempts'][0]['result_directory'])
        self.assertEqual(study.main(['--id','test','--resume','--retry-failed']),0)
        m=self.manifest();self.assertEqual(len(m['jobs'][2]['attempts']),2)
        self.assertTrue((failed_dir/'manifest.json').exists())

    def test_capacity_failure_skips_all_larger(self):
        with patch.dict(os.environ,{'FAIL_BATCH':'20'}):self.assertEqual(study.main(self.args+['--failure-policy','skip-larger']),1)
        self.assertEqual([j['status'] for j in self.manifest()['jobs']],['completed','completed','failed','skipped','skipped','skipped'])

    def test_cache_violation_always_stops(self):
        with patch.dict(os.environ,{'BAD_CACHE':'1'}):self.assertEqual(study.main(self.args+['--failure-policy','continue']),1)
        self.assertEqual(self.manifest()['status'],'failed')
        self.assertTrue(all(j['status']=='pending' for j in self.manifest()['jobs'][1:]))

    def test_pause_finishes_child_then_resume(self):
        pause=self.out/'batch_studies/test/PAUSE'
        with patch.dict(os.environ,{'TEST_PAUSE':str(pause)}):self.assertEqual(study.main(self.args),0)
        m=self.manifest();self.assertEqual(m['status'],'paused');self.assertEqual(m['jobs'][0]['status'],'completed')
        pause.unlink();self.assertEqual(study.main(['--id','test','--resume']),0)

    def test_legacy_pause_requires_scoped_authorization(self):
        p=self.out/'manifests/LARGE_BENCHMARKS_PAUSED.json'
        p.parent.mkdir(parents=True);p.write_text(json.dumps({'paused':True}))
        with self.assertRaisesRegex(RuntimeError,'Legacy pause guard'):study.main(self.args)
        self.assertFalse((self.out/'results').exists())
        self.assertEqual(study.main(['--id','test','--resume','--authorize-paused']),0)
        n=json.loads(p.read_text());self.assertTrue(n['paused']);self.assertEqual(len(n['authorized_runs']),6)

    def test_interrupted_attempt_is_preserved(self):
        self.assertEqual(study.main(self.args+['--dry-run']),0)
        m=self.manifest();j=m['jobs'][0];j['status']='running'
        j['attempts']=[{'id':'old_interrupted_attempt','runner_pid':99999999,'result_directory':str(self.root/'interrupted')}]
        (self.root/'interrupted').mkdir();(self.root/'interrupted/raw.log').write_text('partial evidence')
        (self.out/'batch_studies/test/manifest.json').write_text(json.dumps(m))
        self.assertEqual(study.main(['--id','test','--resume']),0)
        self.assertEqual(len(self.manifest()['jobs'][0]['attempts']),2)
        self.assertTrue((self.root/'interrupted/raw.log').exists())

    def test_resume_rejects_silently_changed_limits(self):
        self.assertEqual(study.main(self.args+['--dry-run']),0)
        with self.assertRaises(SystemExit):study.main(['--id','test','--resume','--min-device-free-gib','0'])
        self.assertFalse((self.out/'results').exists())

    def test_changed_binary_rejected_on_resume(self):
        self.assertEqual(study.main(self.args+['--dry-run']),0)
        (self.root/'binary').write_text('changed')
        with self.assertRaises(SystemExit):study.main(['--id','test','--resume'])
        self.assertFalse((self.out/'results').exists())


if __name__=='__main__':unittest.main()
