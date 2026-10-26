"""Small-control diagnostic figure; not a full-workload performance claim."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/evaluation/diagnostics';rows=list(csv.DictReader((OUT/'reports/same_context_hot_controls.csv').open()));dest=OUT/'figures';dest.mkdir(exist_ok=True)
labels=['Primary\ncache auto','Multi\ncache auto','Primary\ndirect ions','Multi\ndirect ions'];conditions=[('primary','auto'),('multi','auto'),('primary','disabled'),('multi','disabled')]
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for bi,(backend,color) in enumerate([('cuda','#4675a4'),('rt-instanced','#d99b45'),('rt-custom','#62966e')]):
 values=[next(float(r['median_ms']) for r in rows if r['backend']==backend and (r['precursor_mode'],r['cache_policy'])==c) for c in conditions]
 axes[0].bar([i+(bi-1)*.23 for i in range(4)],values,width=.23,color=color,label=backend)
 if backend!='cuda':
  ratios=[next(float(r['cuda_over_backend_ratio']) for r in rows if r['backend']==backend and (r['precursor_mode'],r['cache_policy'])==c) for c in conditions]
  axes[1].plot(range(4),ratios,'o-',color=color,label=backend)
for ax in axes:ax.set_xticks(range(4),labels,fontsize=8);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
axes[0].set_ylabel('Hot fused scoring event time (ms)');axes[1].set_ylabel('CUDA / RT time');axes[1].axhline(1,color='gray',linestyle='--',linewidth=.8)
fig.suptitle('Diagnostic only: 256 Pan X1 spectra, fixed Ecoli FASTA',fontsize=11);fig.text(.5,.01,'Same-context 3 warmups + 7 repeats per immutable batch; median of summed batch times.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.04,1,.95])
for ext in ('png','pdf','svg'):fig.savefig(dest/('precursor_cache_interaction.'+ext),dpi=180)
