"""CPU reference with host stage timers only; native assigned-peptide batching preserved."""
import shutil,subprocess,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'output/evaluation/diagnostics/cpu_primary_baseline';D.mkdir(exist_ok=False);S=D/'source';B=D/'build';S.mkdir();shutil.copytree(R/'mvh',S/'mvh',ignore=shutil.ignore_patterns('__pycache__'))
for name in ('MSToolkit','shared'):(S/name).symlink_to(R/name,target_is_directory=True)
p=S/'mvh/src/database_search.cpp';s=p.read_text();s='#include <iomanip>\n'+s
anchor='\tiPeptideArraySize = (int)vpPeptideArray.size();';assert s.count(anchor)==1
s=s.replace(anchor,'''    unsigned long long diagnosticAssociations=0;
    for(const auto* scan:vpAllMS2Scans)diagnosticAssociations+=scan->vMassChargePeptidePtrTuples.size();
    const double diagnosticBatchStart=omp_get_wtime();
'''+anchor)
anchor='\t// every MS2 scans scores their matched peptides';assert s.count(anchor)==1;s=s.replace(anchor,'    const double diagnosticPrepared=omp_get_wtime();\n'+anchor)
anchor='\t// free memory of all peptide objects';assert s.count(anchor)==1;s=s.replace(anchor,'    const double diagnosticScored=omp_get_wtime();\n'+anchor)
anchor='\tvpPeptideArray.clear();';assert s.count(anchor)==1;s=s.replace(anchor,anchor+'''
    const double diagnosticReleased=omp_get_wtime();
    std::cout<<std::setprecision(12)<<"[CPU batch] assigned_peptides="<<iPeptideArraySize
      <<" associations="<<diagnosticAssociations
      <<" preprocess_seconds="<<diagnosticPrepared-diagnosticBatchStart
      <<" scoring_seconds="<<diagnosticScored-diagnosticPrepared
      <<" cleanup_seconds="<<diagnosticReleased-diagnosticScored
      <<" batch_service_seconds="<<diagnosticReleased-diagnosticBatchStart<<std::endl;''')
anchor='\tmyProteinDatabase.loadDatabase();';assert s.count(anchor)==1;s=s.replace(anchor,'    unsigned long long diagnosticGenerated=0;\n'+anchor)
anchor='\t\twhile (myProteinDatabase.getNextPeptide(currentPeptide))\n\t\t{';assert s.count(anchor)==1;s=s.replace(anchor,anchor+'\n            ++diagnosticGenerated;')
anchor='\tCLOCKSTOP;';assert s.count(anchor)==1;s=s.replace(anchor,'    std::cout<<"[CPU generated] peptides="<<diagnosticGenerated<<std::endl;\n'+anchor);p.write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected=['mvh/original/src/MVH.cpp','mvh/original/src/ms2scan.cpp','mvh/original/src/peptide.cpp','mvh/src/spectrum_input.cpp','mvh/include/mvh_scan_vector.h']
assert all(sha(R/f)==sha(S/f) for f in protected)
commands=[['cmake','-S',str(S/'mvh'),'-B',str(B),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release'],['cmake','--build',str(B),'--target','sipros_mvh','-j','4']]
m={'status':'running','commands':commands,'scope':'CPU reference, host coarse stage timers only; scientific matching and original native batching unchanged','protected_sha256':{f:sha(S/f) for f in protected},'cpu_threads':4,'cpu_native_batch_assigned_peptides':2000000};mp=D/'build.json'
try:
 for i,c in enumerate(commands):
  mp.write_text(json.dumps(m,indent=2))
  with (D/f'build_{i}.log').open('w') as log:subprocess.run(c,stdout=log,stderr=subprocess.STDOUT,check=True)
 m.update(status='completed',binary_sha256=sha(B/'bin/sipros_mvh'))
finally:mp.write_text(json.dumps(m,indent=2))
print(json.dumps({'status':m['status'],'directory':str(D)}))
