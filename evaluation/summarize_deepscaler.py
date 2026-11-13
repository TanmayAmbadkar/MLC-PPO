#!/usr/bin/env python3
"""Compare DeepScaleR warm-restart checkpoints with the original policy on MATH-500."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
root=Path('runs/deepscaler/math500');sources={'base':Path('runs/ppo_controlled_math500/base'),'final_ensemble':root/'final_ensemble','multilayer_mean':root/'multilayer_mean'};data={}
for name,path in sources.items():
 files=sorted(path.glob('[0-9][0-9][0-9][0-9].json'))
 if len(files)!=500:raise ValueError(f'{name}: expected 500 records, got {len(files)}')
 rows=[json.loads(f.read_text()) for f in files];data[name]={r['id']:r for r in rows}
 if len(data[name])!=500:raise ValueError(f'{name}: duplicate IDs')
ids=set(data['base'])
if any(set(v)!=ids for v in data.values()):raise ValueError('Problem IDs differ')
for pid in ids:
 if len({(d[pid]['status'],str(d[pid]['original_gold'])) for d in data.values()})!=1:raise ValueError(f'Inconsistent gold or exclusion: {pid}')
ids=sorted(i for i in ids if data['base'][i]['status']=='scored');n=len(ids)
if not n:raise ValueError('No scored problems')
vals={name:np.array([r[i]['reward'] for i in ids]) for name,r in data.items()};rng=np.random.default_rng(42);draws=rng.integers(0,n,size=(5000,n))
report={'dataset':'HuggingFaceH4/MATH-500','total':500,'scored':n,'decoding':'greedy','max_new_tokens':2048,'models':{},'paired':{},'note':'Single-seed warm restart with fresh optimizer and RNG; architectures restarted from different original steps. Exact/near duplicate filtering does not exclude paraphrases or pretraining exposure.'}
for name,x in vals.items():
 report['models'][name]={'correct':int(x.sum()),'accuracy':float(x.mean()),'problem_bootstrap_95ci':np.quantile(x[draws].mean(1),[.025,.975]).tolist(),'truncated':sum(not data[name][i]['ended_with_eos'] for i in ids)}
for first,second in combinations(vals,2):
 d=vals[first]-vals[second];report['paired'][f'{first}_minus_{second}']={'accuracy_difference':float(d.mean()),'problem_bootstrap_95ci':np.quantile(d[draws].mean(1),[.025,.975]).tolist(),'first_only_correct':int((d==1).sum()),'second_only_correct':int((d==-1).sum())}
(root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
