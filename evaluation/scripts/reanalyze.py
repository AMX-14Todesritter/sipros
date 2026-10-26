"""Recover analysis-only failures from complete immutable run outputs."""
import argparse,csv,datetime,json,subprocess
from pathlib import Path
from inventory import ROOT,OUT,sha
from run import compare,save
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);a=p.parse_args();mp=a.manifest.resolve();directory=mp.parent
if OUT.resolve() not in mp.parents:raise ValueError('Manifest must be in output/evaluation')
m=json.loads(mp.read_text());allowed="Output spectrum coverage/order differs; requires explicit keyed alignment"
if m.get('status')!='failed' or allowed not in m.get('error',''):raise RuntimeError('Only the documented comparison-coverage analysis failure may be recovered')
if not m.get('statistics') or any(r['status']!='completed' or r['exit_code']!=0 for r in m['runs']):raise RuntimeError('Measured application runs incomplete')
for path,expected in m['inputs'].items():
 if sha(Path(path))!=expected:raise RuntimeError('Input changed')
for path,expected in m['binaries'].items():
 if sha(Path(path))!=expected:raise RuntimeError('Binary changed')
for path,expected in m['source_sha256'].items():
 if sha(ROOT/path)!=expected:raise RuntimeError('Scientific source changed')
measured=[r for r in m['runs'] if not r['warmup']]
for r in measured:
 if sha(directory/r['name']/'mvh_psms.tsv')!=r['psm_sha256']:raise RuntimeError('Saved result changed')
backup=directory/'manifest_before_analysis_recovery.json'
if backup.exists():raise FileExistsError('Analysis recovery backup already exists')
backup.write_bytes(mp.read_bytes())
ref=next(r for r in measured if r['backend']=='cuda');m['comparisons']={}
for backend in m['statistics']:
 if backend=='cuda':continue
 other=next(r for r in measured if r['backend']==backend)
 m['comparisons'][backend]=compare(directory/ref['name']/'mvh_psms.tsv',directory/other['name']/'mvh_psms.tsv',directory/(backend+'_differences.tsv'))
cols=['backend','repeat','wall_seconds','search_seconds','fused_event_seconds','fused_device_seconds','theory_seconds','gpu_service_seconds','candidates','generated_peptides','gpu_peak_bytes','host_hwm_bytes','psm_sha256']
with (directory/'metrics.csv').open('w') as f:
 w=csv.writer(f);w.writerow(cols)
 for r in measured:w.writerow([r['backend'],r['repeat'],r['wall_seconds'],r['summary']['search_seconds'],r['totals'].get('kernel_seconds'),r['totals'].get('fused_device_seconds'),r['totals'].get('theory_seconds'),r['totals'].get('gpu_service_seconds'),r['totals'].get('candidates'),r['generated_peptides'],r['device_peak_bytes'],r['host_observed_hwm_bytes'],r['psm_sha256']])
m['analysis_recovery']={'original_error':m.pop('error'),'backup':str(backup),'analysis_code_sha256':sha(ROOT/'evaluation/scripts/run.py'),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reran_searches':False,'validated':'All application runs completed, source/input/binary/output hashes rechecked'}
m['status']='completed';save(mp,m)
print('Analysis recovered; all original measured outputs preserved.')
