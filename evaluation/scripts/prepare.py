"""Deterministic pilot preparation and sequential RAW recovery."""
import argparse, csv, datetime, json, os, subprocess
from pathlib import Path
from inventory import ROOT, OUT, sha, inspect

def records(path):
    header=[];record=[]
    with path.open('rb') as f:
        for line in f:
            if line.startswith(b'S\t') or line.startswith(b'S '):
                if record:yield b''.join(record)
                record=[line]
            elif record:record.append(line)
            else:header.append(line)
        if record:yield b''.join(record)

def pilot(n):
    source=ROOT/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2'
    directory=OUT/'inputs'/('ecoli_x1_first'+str(n));directory.mkdir(parents=True,exist_ok=False)
    dest=directory/'spectra.FT2';header=[]
    with source.open('rb') as f:
        for line in f:
            if line.startswith(b'S\t') or line.startswith(b'S '):break
            header.append(line)
    with dest.open('wb') as f,(directory/'spectrum_map.csv').open('w') as m:
        f.writelines(header);w=csv.writer(m);w.writerow(['subset_index','source','original_scan_id','record_sha256'])
        import hashlib,itertools
        count=0
        for i,rec in enumerate(itertools.islice(records(source),n)):
            f.write(rec);w.writerow([i,str(source.relative_to(ROOT)),rec.splitlines()[0].split()[1].decode(),hashlib.sha256(rec).hexdigest()]);count+=1
    if count!=n:raise RuntimeError('Not enough source spectra')
    manifest={'status':'completed','selection':'deterministic prefix; pilot only, not the source-stratified scaling experiment','source_sha256':sha(source),'output':inspect(dest,'ft2'),'requested_scans':n}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2));print(directory)

def recover(names):
    cfg=json.loads((ROOT/'evaluation/configs/workloads.json').read_text())
    logroot=OUT/'results/preparation';logroot.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/'.benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in names:
            w=cfg['workloads'][name];raw=ROOT/w['raw'];dest=ROOT/w['ft2'];dest.parent.mkdir(parents=True,exist_ok=False)
            cmd=[str(ROOT/'tools/raxport'),'-f',str(raw),'-o',str(dest.parent),'-n','6']
            mp=dest.parent/'conversion_manifest.json';m={'status':'running','command':cmd,'raw_sha256':sha(raw),'converter_sha256':sha(ROOT/'tools/raxport'),'expected_scans':w['expected_scans']}
            mp.write_text(json.dumps(m,indent=2));print('CONVERT '+name,flush=True)
            env=os.environ.copy();env.update(DOTNET_GCHeapHardLimit='20000000',TMPDIR=str(OUT/'tmp'))
            with (logroot/(name+'.log')).open('w') as log:
                result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            m['exit_code']=result.returncode
            if result.returncode==0 and dest.is_file():
                m['output']=inspect(dest,'ft2');m['status']='completed' if m['output']['scans']==w['expected_scans'] and not m['output']['invalid_primary_charge_scans'] else 'invalidated'
            else:m['status']='failed'
            mp.write_text(json.dumps(m,indent=2))
            print(name+' '+m['status'],flush=True)
            if m['status']!='completed':raise RuntimeError('Conversion failed or scan count changed; inspect '+str(mp))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    t=s.add_parser('pilot');t.add_argument('--scans',type=int,default=256)
    t=s.add_parser('recover');t.add_argument('datasets',nargs='+',choices=['ecoli_x2','ecoli_x3','human'])
    a=p.parse_args()
    if a.action=='pilot':pilot(a.scans)
    else:recover(a.datasets)
