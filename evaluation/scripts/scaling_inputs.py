"""Prepare deterministic nested inputs, preserving decoy families and FT2 records."""
import argparse, csv, hashlib, json, re
from contextlib import ExitStack
from pathlib import Path
from inventory import ROOT, OUT, sha, inspect
from prepare import records

def fasta_records(path):
    record=[]
    with path.open('rb') as f:
        for line in f:
            if line.startswith(b'>'):
                if record:yield b''.join(record)
                record=[line]
            else:record.append(line)
        if record:yield b''.join(record)

def ident(record):return record.splitlines()[0][1:].split()[0].decode()
def family_id(name):return re.sub(r'^(?:Rev_[12]_|TestRev_)','',name)
def score(seed,name):return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8],'big')

def soil(seed):
    source=ROOT/'raw/Soil_fw_3rev.fasta';dest=OUT/'inputs'/f'soil_nested_seed{seed}';dest.mkdir(parents=True,exist_ok=False)
    percentages=[1,5,10,25,50];counts={x:0 for x in percentages};families=0;prefixes={};seen_targets=set()
    manifest={'status':'running','seed':seed,'method':'SHA256(seed:target_id) first 64 bits threshold, deterministic nested family selection; original record order','source':str(source),'source_sha256':sha(source),'subsets':{}}
    mp=dest/'manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    try:
        with ExitStack() as stack:
            streams={x:stack.enter_context((dest/f'soil_{x:03d}pct.fasta').open('wb')) for x in percentages}
            iterator=iter(fasta_records(source))
            for target in iterator:
                decoy=next(iterator,None)
                if decoy is None:raise ValueError('Odd FASTA records')
                a,b=ident(target),ident(decoy)
                if family_id(a)!=a or family_id(b)!=a or a==b:raise ValueError(f'Unexpected adjacent target/decoy family: {a}, {b}')
                if a in seen_targets:raise ValueError('Duplicate target ID')
                seen_targets.add(a);families+=1
                prefix=b[:-len(a)];prefixes[prefix]=prefixes.get(prefix,0)+1
                value=score(seed,a)
                for x,f in streams.items():
                    if value*100<x*2**64:f.write(target);f.write(decoy);counts[x]+=2
        for x in percentages:
            p=dest/f'soil_{x:03d}pct.fasta';manifest['subsets'][str(x)]={'path':str(p),'entries':counts[x],'sha256':sha(p)}
        manifest.update(status='completed',family_count=families,source_entries=2*families,decoy_prefix_counts=prefixes)
        manifest['subsets']['100']={'path':str(source),'entries':2*families,'sha256':manifest['source_sha256'],'reuse_original':True}
    except BaseException as e:manifest.update(status='failed',error=repr(e));raise
    finally:mp.write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

def scans(seed):
    cfg=json.loads((ROOT/'evaluation/configs/workloads.json').read_text());names=['ecoli_x1','ecoli_x2','ecoli_x3'];points=[5000,10000,20000,40000,60000,80000,100000,120000,140297]
    sources={k:ROOT/cfg['workloads'][k]['ft2'] for k in names}
    for p in sources.values():
        if not p.is_file():raise ValueError('Missing '+str(p))
    dest=OUT/'inputs'/f'scan_scaling_seed{seed}';dest.mkdir(parents=True,exist_ok=False)
    counts={};rankings={};headers={};source_hashes={}
    for name,path in sources.items():
        indices=[];seen=set();header=[]
        with path.open('rb') as f:
            for line in f:
                if line.startswith(b'S\t') or line.startswith(b'S '):break
                header.append(line)
        headers[name]=b''.join(header)
        for i,rec in enumerate(records(path)):
            sid=rec.splitlines()[0].split()[1].decode()
            if sid in seen:raise ValueError('Duplicate source scan ID')
            seen.add(sid);indices.append((score(seed,name+':'+sid),i))
        counts[name]=len(indices);rankings[name]=[i for _,i in sorted(indices)];source_hashes[name]=sha(path)
        if counts[name]!=cfg['workloads'][name]['expected_scans']:raise ValueError('Unexpected source scan count')
    total=sum(counts.values());assert total==points[-1]
    selected={};quotas={};last={k:0 for k in names}
    for n in points:
        q={k:n*counts[k]//total for k in names}
        for k in sorted(names,key=lambda k:(-(n*counts[k]%total),k))[:n-sum(q.values())]:q[k]+=1
        assert all(q[k]>=last[k] for k in names);last=q.copy();quotas[n]=q
        selected[n]={k:set(rankings[k][:q[k]]) for k in names}
    manifest={'status':'running','seed':seed,'source_sha256':source_hashes,'source_counts':counts,'source_quotas':quotas,'method':'nested per-source SHA256-ranked scan IDs; proportional largest-remainder quotas; stable global original ordinal IDs; preserve all non-S bytes','outputs':{}}
    mp=dest/'manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    try:
        with ExitStack() as stack:
            streams={n:stack.enter_context((dest/f'scans_{n}.FT2').open('wb')) for n in points}
            maps={n:csv.writer(stack.enter_context((dest/f'scans_{n}_map.csv').open('w'))) for n in points}
            written={n:0 for n in points}
            for n,f in streams.items():f.write(headers[names[0]]);maps[n].writerow(['internal_scan_id','source','original_scan_id','original_record_sha256','remapped_record_sha256'])
            offset=0
            for name,path in sources.items():
                for i,rec in enumerate(records(path)):
                    sid=rec.splitlines()[0].split()[1].decode();gid=offset+i+1
                    remapped=re.sub(rb'^(S[ \t]+)\S+',lambda m:m.group(1)+str(gid).encode(),rec,count=1)
                    normalized=lambda b:re.sub(rb'^(S[ \t]+)\S+',rb'\g<1>ID',b,count=1)
                    assert normalized(rec)==normalized(remapped)
                    original_hash=hashlib.sha256(rec).hexdigest();output_hash=hashlib.sha256(remapped).hexdigest()
                    for n in points:
                        if i in selected[n][name]:
                            streams[n].write(remapped);maps[n].writerow([gid,name,sid,original_hash,output_hash]);written[n]+=1
                offset+=counts[name]
        for n in points:
            assert written[n]==n
            if n!=points[0]:
                prev=points[points.index(n)-1]
                assert all(selected[prev][k]<=selected[n][k] for k in names)
            info=inspect(dest/f'scans_{n}.FT2','ft2');assert info['scans']==n and info['duplicate_scan_ids']==0 and info['invalid_primary_charge_scans']==0
            manifest['outputs'][str(n)]=info
        manifest['status']='completed'
    except BaseException as e:manifest.update(status='failed',error=repr(e));raise
    finally:mp.write_text(json.dumps(manifest,indent=2))
    print('Prepared '+str(dest),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['soil','scans']);p.add_argument('--seed',type=int,default=20261026);a=p.parse_args()
    if a.kind=='soil':soil(a.seed)
    else:scans(a.seed)
