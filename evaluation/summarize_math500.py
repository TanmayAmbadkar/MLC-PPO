#!/usr/bin/env python3
"""Paired, problem-level comparison of controlled PPO checkpoints on MATH-500."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
root=Path('runs/ppo_controlled_math500');names=['base','final_single','final_ensemble','multilayer_mean'];data={}
for name in names:
 files=sorted((root/name).glob('[0-9][0-9][0-9][0-9].json'))
 if len(files)!=500:raise ValueError(f'{name}: expected 500 records, found {len(files)}')
 rows=[json.loads(p.read_text()) for p in files]
 data[name]={r['id']:r for r in rows}
 if len(data[name])!=500:raise ValueError(f'{name}: duplicate problem ID')
ids=set(data['base'])
if any(set(d)!=ids for d in data.values()):raise ValueError('MATH-500 problem sets differ')
for pid in ids:
 if len({data[n][pid]['status'] for n in names})!=1 or len({str(data[n][pid]['original_gold']) for n in names})!=1:
  raise ValueError(f'Inconsistent gold or exclusion for {pid}')
scored=sorted(pid for pid in ids if data['base'][pid]['status']=='scored')
if not scored:raise ValueError('No scored problems')
values={name:np.array([data[name][pid]['reward'] for pid in scored],dtype=float) for name in names}
rng=np.random.default_rng(42);draws=rng.integers(0,len(scored),size=(5000,len(scored)))
report={'dataset':'HuggingFaceH4/MATH-500','total':500,'scored':len(scored),'excluded_unparseable_gold':500-len(scored),
        'decoding':'greedy','max_new_tokens':2048,'models':{},'paired_vs_base':{},'note':'Single training seed; overlapping or paraphrased problems are not ruled out by exact-match decontamination.'}
for name in names:
 x=values[name];report['models'][name]={'correct':int(x.sum()),'accuracy':float(x.mean()),
   'problem_bootstrap_95ci':np.quantile(x[draws].mean(1),[.025,.975]).tolist(),
   'truncated':sum(not data[name][pid]['ended_with_eos'] for pid in scored)}
 if name!='base':
  d=x-values['base'];report['paired_vs_base'][name]={'accuracy_difference':float(d.mean()),
   'problem_bootstrap_95ci':np.quantile(d[draws].mean(1),[.025,.975]).tolist(),
   'improved_problems':int(((x==1)&(values['base']==0)).sum()),'regressed_problems':int(((x==0)&(values['base']==1)).sum())}
report['paired_between_checkpoints']={}
for first,second in combinations(names[1:],2):
 d=values[first]-values[second]
 report['paired_between_checkpoints'][f'{first}_minus_{second}']={
   'accuracy_difference':float(d.mean()),
   'problem_bootstrap_95ci':np.quantile(d[draws].mean(1),[.025,.975]).tolist(),
   'first_only_correct':int((d==1).sum()),'second_only_correct':int((d==-1).sum())}
(root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
