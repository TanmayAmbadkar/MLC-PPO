#!/usr/bin/env python3
"""Paired problem-bootstrap checkpoint comparisons for fixed hard-math subsets."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
root=Path('runs/hard_math_eval');models=['base','final_single','final_ensemble','multilayer_mean'];out={}
for bench in ['minerva','olympiad']:
 manifest=json.loads((root/f'{bench}_manifest.json').read_text());n=manifest['included'];data={}
 for name in models:
  files=sorted((root/bench/name).glob('[0-9][0-9][0-9][0-9].json'))
  if len(files)!=n:raise ValueError(f'{bench}/{name}: expected {n}, got {len(files)}')
  rows=[json.loads(f.read_text()) for f in files];data[name]={x['id']:x for x in rows}
  if len(data[name])!=n:raise ValueError('duplicate IDs')
 ids=set(data['base'])
 if any(set(data[x])!=ids for x in models):raise ValueError('problem sets differ')
 for i in ids:
  if len({(data[m][i]['status'],str(data[m][i]['original_gold'])) for m in models})!=1:raise ValueError(f'inconsistent gold/status {i}')
 ids=sorted(i for i in ids if data['base'][i]['status']=='scored');N=len(ids)
 if not N:raise ValueError('no scored examples')
 vals={m:np.array([data[m][i]['reward'] for i in ids]) for m in models};rng=np.random.default_rng(42);draw=rng.integers(0,N,size=(5000,N))
 result={'manifest':manifest,'scored':N,'accuracy':{},'paired':{}}
 for m in models:
  x=vals[m];result['accuracy'][m]={'correct':int(x.sum()),'total':N,'accuracy':float(x.mean()),'ci95':np.quantile(x[draw].mean(1),[.025,.975]).tolist(),'truncated':sum(not data[m][i]['ended_with_eos'] for i in ids)}
 for a,b in combinations(models,2):
  d=vals[a]-vals[b];result['paired'][f'{a}_minus_{b}']={'difference':float(d.mean()),'ci95':np.quantile(d[draw].mean(1),[.025,.975]).tolist()}
 out[bench]=result
(root/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
