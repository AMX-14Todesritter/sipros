"""Append native 4-thread CPU reference to the completed full primary-only GPU group."""
import os,sys,json,subprocess,hashlib,re,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';D=E/'diagnostics/cpu_primary_baseline';parent='full_marine_x2_primary_20261026T192358Z';mp=D/'manifest.json';m={'status':'running','parent_gpu_experiment':parent,'cpu_threads':4,'precursor_mode':'primary-only','cpu_native_batch':2000000,'cpu_batch_unit':'assigned peptide entries','gpu_batch':12000000,'gpu_batch_unit':'generated peptide entries','runs':[]};mp.write_text(json.dumps(m,indent=2))
env=os.environ.copy();env['OMP_DYNAMIC']='FALSE'
for k in ('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR','SIPROS_DIAGNOSTIC_HOT_REPLAY','SIPROS_CONTINUOUS_REPLAY'):env.pop(k,None)
binary=D/'build/bin/sipros_mvh';scans=E/'inputs/ecoli_x2/Pan_062822_X2iso5.FT2';fasta=R/'raw/Marine_fw_3rev.fasta'
def run(rid,binary,scans,fasta):
 cmd=[sys.executable,'-B',str(R/'evaluation/scripts/run.py'),'--id',rid,'--category','accuracy','--backends','cpu','--threads','4','--batch','12000000','--repeats','1','--warmups','0','--scans',str(scans),'--fasta',str(fasta),'--cpu-binary',str(binary)]
 item={'id':rid,'status':'running','command':cmd,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()};m['runs'].append(item);mp.write_text(json.dumps(m,indent=2));print(rid,flush=True)
 with (D/(rid+'.log')).open('w') as log:p=subprocess.run(cmd,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
 item.update(exit_code=p.returncode,status='completed' if p.returncode==0 else 'failed');mp.write_text(json.dumps(m,indent=2))
 if p.returncode:raise RuntimeError(rid)
 n=json.loads((E/'results/accuracy'/rid/'manifest.json').read_text());return n,n['runs'][0]
smallscans=E/'inputs/ecoli_x1_first256/spectra.FT2';smallfasta=E/'diagnostics/marine_control/marine_2048families.fasta'
_,a=run('cpu_primary_timer_check_original',E/'build/cpu/bin/sipros_mvh',smallscans,smallfasta)
_,b=run('cpu_primary_timer_check_observed',binary,smallscans,smallfasta)
m['timer_check_psm_unchanged']=a['psm_sha256']==b['psm_sha256'];mp.write_text(json.dumps(m,indent=2));assert m['timer_check_psm_unchanged']
n,r=run('full_marine_x2_primary_cpu',binary,scans,fasta)
assert int(r['summary']['scan_count'])==46862 and int(r['summary']['omp_max_threads'])==4 and int(r['summary']['peptide_batch_size'])==2000000
text=(E/'results/accuracy/full_marine_x2_primary_cpu/cpu_r1.log').read_text();batches=[dict(re.findall(r'(\w+)=([^\s]+)',l)) for l in text.splitlines() if l.startswith('[CPU batch]')];generated=int(re.search(r'\[CPU generated\] peptides=(\d+)',text)[1]);m.update(status='completed',generated_peptides=generated,cpu_scoring_batches=len(batches),assigned_peptides=sum(int(x['assigned_peptides']) for x in batches),associations=sum(int(x['associations']) for x in batches),cpu_search_seconds=float(r['summary']['search_seconds']),cpu_batch_service_seconds=sum(float(x['batch_service_seconds']) for x in batches),cpu_scoring_loop_seconds=sum(float(x['scoring_seconds']) for x in batches),cpu_preprocess_seconds=sum(float(x['preprocess_seconds']) for x in batches),cpu_cleanup_seconds=sum(float(x['cleanup_seconds']) for x in batches),cpu_psm_sha256=r['psm_sha256'],cpu_summary=r['summary']);mp.write_text(json.dumps(m,indent=2))
with (D/'batches.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(batches[0]));w.writeheader();w.writerows(batches)
sys.path.insert(0,str(R/'evaluation/scripts'));from run import compare
cpuPSM=E/'results/accuracy/full_marine_x2_primary_cpu/cpu_r1/mvh_psms.tsv';gpu=json.loads((E/'results/accuracy'/parent/'manifest.json').read_text());comparisons={}
for g in gpu['runs']:
 comparisons[g['backend']]=compare(cpuPSM,E/'results/accuracy'/parent/g['name']/'mvh_psms.tsv',D/(g['backend']+'_vs_cpu_differences.tsv'))
m['comparisons_cpu_reference']=comparisons;m['parent_manifest']=str(E/'diagnostics'/parent/'manifest.json');mp.write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
