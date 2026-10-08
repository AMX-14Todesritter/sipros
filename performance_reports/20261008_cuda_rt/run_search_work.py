import json,os,re,subprocess
from pathlib import Path
root=Path('/workspace/sipros');out=root/'output/benchmarks/profiling/20261008_characterization_final';out.mkdir(parents=True,exist_ok=True)
binary=root/'build/mvh_rt/search_work_diagnostic/bin/sipros_mvh_cuda'
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(root/'build/mvh_rt/optix_runtime')+':/usr/local/cuda/lib64'
report={}
for backend in ['cuda','rt-custom']:
 cmd=[str(binary),'-f',str(root/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2'),'-c',str(root/'experiments/Regular.cfg'),'-fasta',str(root/'raw/Marine_fw_3rev.fasta'),'--match-backend',backend,'--peptide-batch-size','8000000','-o',str(out/('diagnostic_'+backend))]
 p=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 counts=None
 with (out/(backend+'_search_work.log')).open('w') as log:
  for line in p.stdout:
   log.write(line);log.flush()
   if line.startswith('[SEARCH WORK batch]'):
    counts={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
    print(backend,line.strip(),flush=True);p.terminate();break
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:p.kill();p.wait()
 report[backend]={'command':cmd,'batch_index':0,'batch_size':8000000,'counts':counts,'stop':'external SIGTERM after completed count download; partial search, not timing','exit_code':p.returncode}
 (out/'search_work.json').write_text(json.dumps(report,indent=2))
 if counts is None:raise RuntimeError('No completed batch counters for '+backend)
