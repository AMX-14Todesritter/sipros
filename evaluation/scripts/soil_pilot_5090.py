"""Unattended 1%/5% Soil pilots: multi, sm120, cache off, fixed 140297 scans."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

from inventory import ROOT, OUT, sha
from batch_study import save, utc


def report(d,m):
    lines=['# 5090 Soil 1% / 5% pilot','','Status: '+m['status'],'Updated UTC: '+utc(),'',
           'Fixed 140297 X1/X2/X3 scans; 4M generated-peptide batch; multi-precursor; sm120; cache off.',
           'CUDA and RT-custom only. One measured process per cell; no warmups or profiler.',
           'GPU free-memory reserve 3 GiB; host process RSS limit 55 GiB; runner disk reserve 10 GiB.',
           'Any failure stops the queue. Larger Soil fractions are prepared, not benchmarked.','']
    for j in m['jobs']:
        lines+=['## Soil '+str(j['percent'])+'%: '+j['status'],'']
        p=OUT/'batch_studies'/j['id']/'progress.md'
        if p.exists():lines.append(p.read_text())
        else:lines.append('Measurement not started.')
    if m.get('error'):lines+=['','## Error','',m['error']]
    p=d/'progress.tmp';p.write_text('\n'.join(lines)+'\n');p.replace(d/'progress.md')
    save(d/'manifest.json',m)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--id',default='soil_5090_multi_4m_pilot_v1');a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,45}',a.id):p.error('Invalid pilot ID')
    scans=OUT/'inputs/scan_scaling_seed20261028/scans_140297.FT2'
    source=ROOT/'raw/Soil_fw_3rev.fasta';cfg=ROOT/'experiments/Regular.cfg'
    binary=OUT/'build/gpu_multi/bin/sipros_mvh_cuda';build=binary.parents[1]
    for path in [scans,source,cfg,binary,build/'build_manifest.json']:
        if not path.is_file():p.error('Missing input: '+str(path))
    meta=json.loads((build/'build_manifest.json').read_text())
    if (meta['status']!='completed' or meta['precursor_mode']!='multi' or meta['ion_cache']!='compiled_off'
            or meta['binary_sha256']!=sha(binary)
            or not re.search(r'^CMAKE_CUDA_ARCHITECTURES:[^=]+=120$',(build/'CMakeCache.txt').read_text(),re.M)):
        p.error('Verified multi/cache-off/sm120 binary required')
    sample=json.loads((scans.parent/'manifest.json').read_text())['outputs']['140297']
    if sample['scans']!=140297 or sample['sha256']!=sha(scans):p.error('Combined scan input changed')
    d=OUT/'soil_studies'/a.id;d.parent.mkdir(parents=True,exist_ok=True)
    with (d.parent/'.coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);d.mkdir(exist_ok=False)
        m={'status':'preparing','created_utc':utc(),'binary_sha256':sha(binary),'scans_sha256':sha(scans),
           'config_sha256':sha(cfg),'jobs':[{'id':a.id+f'_p{x:03d}','percent':x,'status':'pending'} for x in (1,5)]}
        report(d,m)
        try:
            print('Preparing/verifying nested Soil target-decoy families',flush=True)
            inp=OUT/'inputs/soil_nested_seed20261028'
            if not inp.exists():
                with (d/'prepare.log').open('w') as log:
                    subprocess.run([sys.executable,'-B',str(ROOT/'evaluation/scripts/scaling_inputs.py'),'soil','--seed','20261028'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            preparation=json.loads((inp/'manifest.json').read_text())
            if preparation['status']!='completed' or preparation['source_sha256']!=sha(source):raise RuntimeError('Soil input manifest invalid or source changed')
            m['database_preparation']=preparation;report(d,m)
            for j in m['jobs']:
                subset=inp/f"soil_{j['percent']:03d}pct.fasta"
                if sha(subset)!=preparation['subsets'][str(j['percent'])]['sha256']:raise RuntimeError('Subset hash mismatch')
                cmd=[sys.executable,'-B',str(ROOT/'evaluation/scripts/batch_study.py'),'--id',j['id'],
                     '--category','database_scaling','--precursor-mode','multi','--scans',str(scans),
                     '--fasta',str(subset),'--config',str(cfg),'--gpu-binary',str(binary),
                     '--batches','4000000','--backends','cuda','rt-custom','--repeats','1','--warmups','0',
                     '--threads','1','--min-device-free-gib','3','--max-host-rss-gib','55','--failure-policy','stop']
                j.update(status='running',command=cmd);m['status']='running';report(d,m);print('START '+j['id'],flush=True)
                with (d/(j['id']+'.log')).open('w') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:report(d,m)
                child_manifest=OUT/'batch_studies'/j['id']/'manifest.json'
                result=json.loads(child_manifest.read_text()) if child_manifest.exists() else {}
                j.update(status=result.get('status','failed'),exit_code=child.returncode);report(d,m)
                if child.returncode or j['status']!='completed':raise RuntimeError(result.get('error','Child runner failed; inspect '+str(d/(j['id']+'.log'))))
                print('DONE '+j['id'],flush=True)
            m.update(status='completed',finished_utc=utc())
        except BaseException as error:
            m.update(status='failed',error=repr(error));raise
        finally:report(d,m)


if __name__=='__main__':main()
