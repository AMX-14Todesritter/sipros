"""Small-fixture validation for portable nested scan preparation/reporting."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
import inventory
import scaling_inputs
import scan_study_5090
from prepare import records


class NestedScanTests(unittest.TestCase):
    def test_nested_sources_and_peak_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'output/evaluation';out.joinpath('inputs').mkdir(parents=True)
            root.joinpath('evaluation/configs').mkdir(parents=True)
            sources=[];cfg={'workloads':{}}
            originals={}
            for x,count in enumerate([2,3,4],1):
                path=root/f'x{x}.FT2';data='H\tfixture\n'
                for sid in range(1,count+1):
                    rec=f'S\t{sid}\t400.0\t12\nZ\t2\t800\n100.123\t456\t0\t0\t0\t1\n'
                    data+=rec;originals[(f'ecoli_x{x}',str(sid))]=rec.encode()
                path.write_text(data);sources.append(path);cfg['workloads'][f'ecoli_x{x}']={'expected_scans':count}
            (root/'evaluation/configs/workloads.json').write_text(json.dumps(cfg))
            with patch.object(scaling_inputs,'ROOT',root),patch.object(scaling_inputs,'OUT',out),patch.object(inventory,'ROOT',root):
                scaling_inputs.scans(7,[3,6,9],sources)
            dest=out/'inputs/scan_scaling_seed7';m=json.loads((dest/'manifest.json').read_text())
            self.assertEqual(m['status'],'completed')
            prior=set()
            import csv,re
            for n in [3,6,9]:
                with (dest/f'scans_{n}_map.csv').open() as f:rows=list(csv.DictReader(f))
                self.assertEqual(len(rows),n)
                selected={(r['source'],r['original_scan_id']) for r in rows}
                self.assertTrue(prior<=selected);prior=selected
                rs=list(records(dest/f'scans_{n}.FT2'))
                self.assertEqual(len(rs),n)
                self.assertEqual(len({r.splitlines()[0].split()[1] for r in rs}),n)
                for r,info in zip(rs,rows):
                    norm=lambda b:re.sub(rb'^(S[ \t]+)\S+',rb'\g<1>ID',b,count=1)
                    self.assertEqual(norm(r),norm(originals[(info['source'],info['original_scan_id'])]))

    def test_master_reports_preflight_error_without_native_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);d=out/'scan_studies/test';d.mkdir(parents=True)
            child=out/'batch_studies/case';child.mkdir(parents=True)
            result=out/'results/scan_scaling/missing'
            attempt={'id':'case_a1','exit_code':1,'result_directory':str(result)}
            (child/'case_a1.log').write_text('Traceback\nRuntimeError: NVML initialization failed\n')
            (child/'manifest.json').write_text(json.dumps({'jobs':[{'backend':'cuda','status':'failed','error':'missing manifest','attempts':[attempt]}]}))
            m={'status':'failed','batch':4000000,'repeats':3,'backends':['cuda'],'jobs':[
                {'id':'case','round':1,'scans':140297,'backends':['cuda'],'status':'failed'}]}
            with patch.object(scan_study_5090,'OUT',out):scan_study_5090.report(d,m)
            text=(d/'progress.md').read_text()
            self.assertIn('NVML initialization failed',text)
            self.assertIn('Every round and backend',text)
            self.assertIn('attempt 1, exit=1',text)
            self.assertIn('| 1 | 140297 | cuda | failed | 1 |',text)

    def test_master_reports_live_memory_and_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);d=out/'scan_studies/test';d.mkdir(parents=True)
            child=out/'batch_studies/case';child.mkdir(parents=True)
            result=out/'results/scan_scaling/active';result.mkdir(parents=True)
            (result/'manifest.json').write_text(json.dumps({'runs':[{'name':'cuda_r1','status':'running','resource_guard':'test guard'}]}))
            (result/'cuda_r1.log').write_text('[CUDA scoring] candidates=1\n')
            (result/'cuda_r1.memory.jsonl').write_text(json.dumps({'device_used_bytes':2**30,'device_peak_bytes':2**31,'elapsed_seconds':3})+'\n')
            (child/'manifest.json').write_text(json.dumps({'jobs':[{'backend':'cuda','status':'running','attempts':[
                {'id':'case_a1','result_directory':str(result)}]}]}))
            m={'status':'running','batch':4000000,'repeats':3,'backends':['cuda'],'jobs':[
                {'id':'case','round':1,'scans':20000,'backends':['cuda'],'status':'running'}]}
            with patch.object(scan_study_5090,'OUT',out):scan_study_5090.report(d,m)
            text=(d/'progress.md').read_text()
            self.assertIn('1.00 GiB; peak: 2.00 GiB',text)
            self.assertIn('Completed scoring batches: 1',text)
            self.assertIn('resource guard: test guard',text)

    def test_empty_master_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);m={'status':'preparing','batch':4000000,'repeats':3,'backends':['cuda','rt-custom'],'jobs':[]}
            rows=scan_study_5090.report(d,m)
            self.assertEqual(len(rows),14)
            self.assertTrue((d/'progress.md').exists())
            self.assertTrue((d/'summary.csv').exists())


if __name__=='__main__':unittest.main()
