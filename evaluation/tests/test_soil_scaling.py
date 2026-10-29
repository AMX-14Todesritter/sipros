"""GPU-free checks for unattended X1+X2 Soil matrix preparation/reporting."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import soil_scaling_5090 as module
from inventory import sha
from prepare import records

FAKE=r'''
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--id');p.add_argument('--backends',nargs='+');p.add_argument('--fasta')
a,rest=p.parse_known_args();root=Path(__file__).resolve().parents[2];d=root/'output/evaluation/batch_studies'/a.id;d.mkdir(parents=True)
rows=[]
for b in a.backends:rows.append(dict(batch=8000000,backend=b,repeat=1,fused_seconds=1,rt_setup_seconds=0,complete_scoring_seconds=2,search_seconds=3,device_peak_bytes=4,host_peak_bytes=5,generated=20,candidates=30,psm_sha256=b+Path(a.fasta).name))
with (d/'measurements.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(d/'manifest.json').write_text(json.dumps({'status':'completed','jobs':[{'backend':b,'status':'completed','attempts':[]} for b in a.backends]}))
(d/'progress.md').write_text('Completed fake child case')
'''


class SoilScalingTests(unittest.TestCase):
    def test_combined_scans_unique_and_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=[]
            for i,n in enumerate([2,3],1):
                p=root/f'x{i}.FT2';p.write_text('H\tfixture\n'+''.join(f'S\t{k}\t400\nZ\t2\t800\n100.1\t20\n' for k in range(1,n+1)));sources.append(p)
            with patch.object(module,'SOURCE_COUNTS',[2,3]):
                dest=module.prepare_x1_x2(sources,root/'combined')
                self.assertEqual(len(list(records(dest))),5)
                self.assertEqual([r.splitlines()[0].split()[1] for r in records(dest)],[b'1',b'2',b'3',b'4',b'5'])
                before=sha(dest);self.assertEqual(module.prepare_x1_x2(sources,root/'combined'),dest);self.assertEqual(sha(dest),before)
                sources[0].write_text(sources[0].read_text()+'101\t1\n')
                with self.assertRaisesRegex(RuntimeError,'changed'):module.prepare_x1_x2(sources,root/'combined')

    def test_complete_matrix_without_time_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'output/evaluation';sources=[]
            for i,n in enumerate([2,3],1):
                p=out/f'inputs/ecoli_x{i}/Pan_062822_X{i}iso5.FT2';p.parent.mkdir(parents=True)
                p.write_text('H\tfixture\n'+''.join(f'S\t{k}\t400\nZ\t2\t800\n100\t20\n' for k in range(1,n+1)));sources.append(p)
            (root/'raw').mkdir();soil=root/'raw/Soil_fw_3rev.fasta';soil.write_text('>a\nAAAAAAAK\n>Rev_1_a\nKAAAAAAA\n')
            (root/'experiments').mkdir();(root/'experiments/Regular.cfg').write_text('fixture')
            (root/'evaluation/scripts').mkdir(parents=True)
            (root/'evaluation/scripts/run.py').write_text('fixture runner')
            (root/'evaluation/scripts/batch_study.py').write_text(FAKE)
            build=out/'build/gpu_multi';(build/'bin').mkdir(parents=True);binary=build/'bin/sipros_mvh_cuda';binary.write_text('fixture binary')
            (build/'CMakeCache.txt').write_text('CMAKE_CUDA_ARCHITECTURES:STRING=120\n')
            (build/'build_manifest.json').write_text(json.dumps({'status':'completed','precursor_mode':'multi','ion_cache':'compiled_off','binary_sha256':sha(binary)}))
            inputs=out/'inputs/soil_nested_seed20261028';inputs.mkdir();subsets={}
            for pct in module.PERCENTAGES:
                f=soil if pct==100 else inputs/f'soil_{pct:03d}pct.fasta'
                if pct!=100:f.write_bytes(soil.read_bytes())
                subsets[str(pct)]={'sha256':sha(f)}
            (inputs/'manifest.json').write_text(json.dumps({'status':'completed','source_sha256':sha(soil),'subsets':subsets}))
            with patch.object(module,'ROOT',root),patch.object(module,'OUT',out),patch.object(module,'SOURCE_COUNTS',[2,3]),patch.object(sys,'argv',['soil_scaling_5090.py','--id','fixture','--repeats','3']):module.main()
            d=out/'soil_studies/fixture';m=json.loads((d/'manifest.json').read_text())
            self.assertEqual(m['status'],'completed');self.assertEqual(len(m['jobs']),18)
            self.assertTrue(all('--stop-at-utc' not in j['command'] for j in m['jobs']))
            self.assertTrue(all(j['status']=='completed' for j in m['jobs']))
            with (d/'measurements.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),36)
            text=(d/'progress.md').read_text();self.assertIn('36/36',text);self.assertIn('No time deadline',text)


if __name__=='__main__':unittest.main()
