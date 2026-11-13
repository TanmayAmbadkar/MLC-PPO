#!/usr/bin/env python3
"""Problem-level metrics for the saved-critic continuation study."""
import json
from pathlib import Path
import numpy as np
root=Path('runs/deepscaler/critic_validation');names=['final_ensemble','multilayer_mean'];data={}
priors={'final_ensemble':['runs/deepscaler/final_ensemble_seed42/rollouts.jsonl','runs/deepscaler/restart_final_from300_seed43/rollouts.jsonl'],'multilayer_mean':['runs/deepscaler/multilayer_mean_seed42/rollouts.jsonl','runs/deepscaler/restart_multilayer_from200_seed43/rollouts.jsonl']}
def prior(paths):
 values=[]
 for path in paths:
  for line in Path(path).read_text().splitlines():
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   if r.get('phase')=='ppo':values.append(r['reward'])
 if not values:raise ValueError('No training rollout rewards for constant baseline')
 return float(np.mean(values)),len(values)
def metrics(rows,c):
 p=np.array([x['value'] for x in rows]);y=np.array([np.mean([z['reward'] for z in x['continuations']]) for x in rows]);n=np.array([len(x['continuations']) for x in rows]);brier=np.array([np.mean([(x['value']-z['reward'])**2 for z in x['continuations']]) for x in rows]);cb=np.array([np.mean([(c-z['reward'])**2 for z in x['continuations']]) for x in rows]);return {'mse':float(np.mean((p-y)**2)),'mse_minus_mc_noise_estimate':float(np.mean((p-y)**2-y*(1-y)/(n-1))),'brier':float(brier.mean()),'constant_brier':float(cb.mean()),'bias':float(np.mean(p-y)),'mean_prediction':float(p.mean()),'mean_observed':float(y.mean()),'states':len(rows)}
rng=np.random.default_rng(42);report={'sampling':'temperature=1, top_p=1, top_k=0, float32, 2048 total response tokens','note':'Each critic is evaluated on prefixes and independent continuations from its own policy; cross-model differences combine critic and state/policy differences. Observed difficulty groups are 8 independent prompt rollouts, not true difficulty. MC-noise correction may be negative.','models':{}}
for name in names:
 files=sorted((root/name).glob('*.json'));rows=[json.loads(f.read_text()) for f in files]
 if len(rows)!=50 or len({r['id'] for r in rows})!=50:raise ValueError(f'{name}: expected 50 unique problems, got {len(rows)}')
 for r in rows:
  if len(r['states'])!=6 or len(r['difficulty_rollouts'])!=8 or any(len(s['continuations'])!=32 for s in r['states']):raise ValueError(f'incomplete {r["id"]}')
 data[name]=rows;c,count=prior(priors[name]);states=[s for r in rows for s in r['states']];overall=metrics(states,c);indices=rng.integers(0,len(rows),size=(2000,len(rows)));boot=[]
 for sample in indices:
  samples=[s for i in sample for s in rows[i]['states']];m=metrics(samples,c);boot.append((m['mse'],m['brier'],m['bias'],m['brier']-m['constant_brier']))
 cis=np.quantile(np.array(boot),[.025,.975],axis=0).T.tolist();bins=[]
 for low in np.arange(0,1,.1):
  subset=[s for s in states if s['value']>=low and (s['value']<low+.1 or low>.89 and s['value']<=1)]
  if subset:
   m=metrics(subset,c);bins.append({'lower':round(float(low),1),'states':len(subset),'mean_predicted':m['mean_prediction'],'observed_success':m['mean_observed']})
 groups={}
 for label,rule in [('0_of_8',lambda k:k==0),('1_to_7_of_8',lambda k:0<k<8),('8_of_8',lambda k:k==8)]:
  subset=[r for r in rows if rule(sum(x['reward'] for x in r['difficulty_rollouts']))];group_states=[s for r in subset for s in r['states']];groups[label]={'problems':len(subset),'metrics':metrics(group_states,c) if group_states else None}
 report['models'][name]={'problems':len(rows),'training_reward_constant':c,'training_reward_count':count,'overall':overall,'problem_bootstrap_95ci':dict(zip(['mse','brier','bias','brier_minus_constant'],cis)),'calibration_bins':bins,'by_progress':{str(t):metrics([s for s in states if s['progress']==t],c) for t in (.25,.5,.75)},'by_observed_difficulty':groups,'continuations_without_eos':sum(not x['ended_with_eos'] for s in states for x in s['continuations'])}
if {r['id'] for r in data[names[0]]}!={r['id'] for r in data[names[1]]}:raise ValueError('Held-out problem sets differ')
by_model={name:{r['id']:r for r in rows} for name,rows in data.items()}
ids=sorted(by_model[names[0]])
per_problem=[]
for pid in ids:
 pair=[]
 for name in names:
  c=report['models'][name]['training_reward_constant'];pair.append(metrics(by_model[name][pid]['states'],c))
 per_problem.append([pair[1]['mse']-pair[0]['mse'],pair[1]['brier']-pair[0]['brier']])
per_problem=np.array(per_problem);draws=rng.integers(0,len(ids),size=(2000,len(ids)))
report['paired_multilayer_minus_final']={key:{'difference':float(per_problem[:,i].mean()),'problem_bootstrap_95ci':np.quantile(per_problem[draws,i].mean(1),[.025,.975]).tolist()} for i,key in enumerate(('mse','brier'))}
(root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
