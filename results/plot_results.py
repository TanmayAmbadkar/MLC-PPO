"""Plot recorded MATH-500 scores against generated training tokens."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
root=Path(__file__).parent
with (root/'benchmarks.csv').open(newline='') as f:bench=list(csv.DictReader(f))
with (root/'training_metrics.csv').open(newline='') as f:metrics=list(csv.DictReader(f))
tokens={(r['experiment'],r['method'],int(r['step'])):int(r['generated_tokens']) for r in metrics}
styles={'trivial_loo':('Trivial baseline','#6b7280'),'final_single':('Final head','#d97706'),'grpo':('GRPO','#7c3aed'),'multilayer_all':('All-layer MLC-PPO','#0284c7'),'full_critic':('Full critic','#dc2626')}
fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=False)
for ax,ex,title in zip(axes,['base_rlvr','viability_fast2'],['1.5B Base','1.5B Instruct']):
 for m,(label,color) in styles.items():
  rows=sorted([r for r in bench if r['experiment']==ex and r['method']==m and r['benchmark']=='math500' and (ex,m,int(r['step'])) in tokens],key=lambda r:int(r['step']))
  if rows:ax.plot([tokens[(ex,m,int(r['step']))]/1e6 for r in rows],[100*float(r['accuracy']) for r in rows],marker='o',markersize=3,label=label,color=color)
 initial=next(float(r['accuracy'])*100 for r in bench if r['experiment']==ex and r['method']=='base' and r['benchmark']=='math500')
 ax.axhline(initial,color='#9ca3af',linestyle='--',linewidth=1,label='Initial model');ax.set_title(title);ax.set_xlabel('Generated training tokens (millions)');ax.set_ylabel('MATH-500 accuracy (%)');ax.grid(alpha=.2)
axes[0].legend(fontsize=8,loc='lower right');fig.suptitle('Single-seed checkpoint evaluations',fontsize=12);fig.tight_layout();fig.savefig(root/'math500_learning_curves.png',dpi=160)
