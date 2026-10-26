"""Plan and execute staged, serial main experiments. No automatic algorithm changes."""
import argparse,fcntl,json,os,re,subprocess,sys
from pathlib import Path
from inventory import ROOT,OUT
from run import save

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['real_world','batch_size','database_scaling','scan_scaling'],required=True)
    p.add_argument('--workloads',nargs='*');p.add_argument('--execute',action='store_true');p.add_argument('--id',required=True);p.add_argument('--batch',type=int,default=2000000)
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',a.id):p.error('invalid suite id')
    if a.batch<1:p.error('batch must be positive')
    cfg=json.loads((ROOT/'evaluation/configs/workloads.json').read_text());items=[]
    if a.stage=='real_world':
        for name in cfg.get('real_world_workloads',cfg['workloads']):
            w=cfg['workloads'][name]
            if a.workloads and name not in a.workloads:continue
            items.append((name,w['ft2'],w['fasta'],a.batch,(['cpu'] if w['cpu'] else [])+cfg['backends']))
    elif a.stage=='batch_size':
        w=cfg['workloads']['marine']
        for batch in cfg['batch_sizes']:items.append((f'marine_b{batch}',w['ft2'],w['fasta'],batch,cfg['backends']))
    elif a.stage=='database_scaling':
        m=json.loads((OUT/'inputs/soil_nested_seed20261026/manifest.json').read_text())
        if m['status']!='completed':raise RuntimeError('Soil inputs not validated')
        w=cfg['workloads']['soil']
        for frac in ('1','5','10','25','50','100'):items.append((f'soil_{frac}pct',w['ft2'],m['subsets'][frac]['path'],a.batch,cfg['backends']))
    else:
        m=json.loads((OUT/'inputs/scan_scaling_seed20261026/manifest.json').read_text())
        if m['status']!='completed':raise RuntimeError('Scan inputs not validated')
        for n,item in sorted(m['outputs'].items(),key=lambda t:int(t[0])):items.append((f'scans_{n}',item['path'],cfg['workloads']['ecoli_x1']['fasta'],a.batch,cfg['backends']))
    if not items:raise ValueError('No workloads selected')
    runs=[]
    for name,scans,fasta,batch,backends in items:
        scans=ROOT/scans;fasta=ROOT/fasta
        cmd=[sys.executable,'-B',str(ROOT/'evaluation/scripts/run.py'),'--id',a.id+'_'+name,'--category',a.stage,'--scans',str(scans),'--fasta',str(fasta),'--batch',str(batch),'--backends',*backends,'--repeats','5','--warmups','1']
        resource_required='soil' in name.lower()
        gate=OUT/'reports/soil_x1_fixed_v2_resource_gate.json'
        resource_ready=not resource_required or (gate.is_file() and json.loads(gate.read_text()).get('status')=='completed')
        runs.append({'workload':name,'status':'ready' if scans.is_file() and fasta.is_file() and resource_ready else 'blocked','resource_gate_required':resource_required,'command':cmd})
    dest=OUT/'manifests'/f'{a.id}_{a.stage}.json'
    if dest.exists():raise FileExistsError('Suite id exists; choose a fresh id')
    plan={'status':'ready' if all(r['status']=='ready' for r in runs) else 'blocked','stage':a.stage,'runs':runs,'fixed_batch_generated_peptides':a.batch,
          'resource_policy':'Stop on failed/OOM run, preserve failure. Never silently change batching or sample size. Full Soil requires an earlier resource pilot.'}
    save(dest,plan)
    if not a.execute:print(json.dumps(plan,indent=2));return
    invariants=json.loads((OUT/'reports/pilot_invariants.json').read_text())
    if invariants['status']!='completed':raise RuntimeError('Pilot gate failed')
    if plan['status']!='ready':raise RuntimeError('Missing input blocks suite')
    if a.stage=='database_scaling':
        gate=OUT/'reports/soil_x1_fixed_v2_resource_gate.json'
        if not gate.is_file() or json.loads(gate.read_text()).get('status')!='completed':raise RuntimeError('Soil resource gate not yet completed')
    with (OUT/'.suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan['status']='running';save(dest,plan)
        try:
            for r in runs:
                r['status']='running';save(dest,plan)
                result=subprocess.run(r['command'],cwd=ROOT);r['exit_code']=result.returncode;r['status']='completed' if result.returncode==0 else 'failed';save(dest,plan)
                if result.returncode:raise RuntimeError('Experiment failed; see preserved manifest: '+r['workload'])
                subprocess.run([sys.executable,'-B',str(ROOT/'evaluation/scripts/summarize.py')],cwd=ROOT,check=True)
            plan['status']='completed'
        except BaseException as e:plan.update(status='failed',error=repr(e));raise
        finally:save(dest,plan)
if __name__=='__main__':main()
