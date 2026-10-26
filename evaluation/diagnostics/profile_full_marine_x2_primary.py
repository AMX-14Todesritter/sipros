"""Independent first-batch Nsight Compute profiles after full primary-only baseline."""
import os,sys,subprocess,json,hashlib,datetime,fcntl,csv,io
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'output/evaluation';rid=sys.argv[1];D=E/'diagnostics'/rid;P=D/'profiles';P.mkdir(exist_ok=False)
baseline=json.loads((D/'manifest.json').read_text());assert baseline['status']=='completed'
env=os.environ.copy();env.update(SIPROS_DIAGNOSTIC_NO_ION_CACHE='1',LD_LIBRARY_PATH=str(R/'build/mvh_rt/optix_runtime')+':/usr/local/cuda/lib64',TMPDIR=str(E/'tmp'),OPTIX_CACHE_PATH=str(E/'cache/optix'),CUDA_CACHE_PATH=str(E/'cache/cuda'))
for k in ('SIPROS_DIAGNOSTIC_MULTI_PRECURSOR','SIPROS_DIAGNOSTIC_HOT_REPLAY','SIPROS_CONTINUOUS_REPLAY'):env.pop(k,None)
binary=E/'diagnostics/build/bin/sipros_mvh_cuda';rows=[];manifest={'status':'running','precursor_mode':'primary-only','cache':'disabled','batch':12000000,'scope':'first actual fused scoring launch from full Marine/X2, intentionally partial profile, no full-search performance claims','profiles':[]};mp=P/'manifest.json'
fields={'kernel_time':'gpu__time_duration.sum','registers_requested':'launch__registers_per_thread','entry_stack_bytes':'launch__stack_size','achieved_occupancy_pct':'sm__warps_active.avg.pct_of_peak_sustained_active','sm_warp_instructions':'smsp__inst_executed.sum','dram_reads':'dram__bytes_op_read.sum','dram_writes':'dram__bytes_op_write.sum','l2_sectors':'lts__t_sectors.sum','fp64_pipe_elapsed_pct':'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed','local_load_requests':'l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum','local_store_requests':'l1tex__t_requests_pipe_lsu_mem_local_op_st.sum'}
with (E/'.benchmark.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for backend in ('cuda','rt-custom'):
  dest=P/backend;dest.mkdir();search=[str(binary),'-f',str(E/'inputs/ecoli_x2/Pan_062822_X2iso5.FT2'),'-fasta',str(R/'raw/Marine_fw_3rev.fasta'),'-c',str(R/'experiments/Regular.cfg'),'-o',str(dest/'partial_search'),'--match-backend',backend,'--peptide-batch-size','12000000','--spectrum-cache','host','--peptide-generation','cuda','--result-restoration','final']
  kernel='regex:.*'+('ScoreSequenceVsSpectrum' if backend=='cuda' else 'optixLaunch')+'.*'
  cmd=['ncu','--set','detailed','--kernel-name',kernel,'--launch-count','1','--kill','yes','--clock-control','none','--export',str(dest/'capture'),*search]
  item={'backend':backend,'status':'running','command':cmd};manifest['profiles'].append(item);mp.write_text(json.dumps(manifest,indent=2));print('PROFILE '+backend,flush=True)
  with (dest/'capture.log').open('w') as log:p=subprocess.run(cmd,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
  item['exit_code']=p.returncode
  if p.returncode==0 and (dest/'capture.ncu-rep').exists():
   export=['ncu','--import',str(dest/'capture.ncu-rep'),'--page','raw','--csv'];item['export_command']=export
   with (dest/'raw_metrics.csv').open('w') as log:q=subprocess.run(export,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
   if q.returncode==0:
    text=(dest/'raw_metrics.csv').read_text().splitlines();start=next(i for i,x in enumerate(text) if x.startswith('"ID",'));allrows=list(csv.DictReader(io.StringIO('\n'.join(text[start:]))));units=allrows[0];values=next(x for x in allrows if x.get('ID','').isdigit());row={'backend':backend,'kernel_name':values['Kernel Name']}
    for name,key in fields.items():row[name]=float(values[key]) if values.get(key) else None;row[name+'_unit']=units.get(key)
    rows.append(row);item['status']='completed'
   else:item['status']='failed'
  else:item['status']='failed'
  mp.write_text(json.dumps(manifest,indent=2))
manifest['status']='completed' if all(x['status']=='completed' for x in manifest['profiles']) else 'failed';mp.write_text(json.dumps(manifest,indent=2));(P/'selected_metrics.json').write_text(json.dumps(rows,indent=2))
if rows:
 with (P/'selected_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'status':manifest['status'],'metrics':rows},indent=2))
