#!/usr/bin/env python3
"""Compute paired held-out accuracy differences with problem-level bootstrap intervals."""
import json
from pathlib import Path
import numpy as np
root=Path('runs/dapo_eval');names=['base','mean','min','gate']
rows={name:{p.stem:json.loads(p.read_text()) for p in (root/name).glob('*.json') if p.name!='config.json'} for name in names}
expected={json.loads(s)['id'] for s in Path('runs/experiment2/prepared/test.jsonl').read_text().splitlines()}
if any(set(data)!=expected for data in rows.values()):raise ValueError('All four evaluations must contain exactly the held-out problems')
ids=sorted(expected);rng=np.random.default_rng(42);draws=rng.integers(0,len(ids),size=(5000,len(ids)))
values={n:np.array([rows[n][i]['reward'] for i in ids],dtype=float) for n in names}
report={'problems':len(ids),'decoding':'greedy','max_new_tokens':2048,'models':{},'paired_vs_base':{},'note':'Finite held-out sample; bootstrap resamples problems. Truncated responses count as verifier scores.'}
for n in names:
 x=values[n];r=[rows[n][i] for i in ids]
 report['models'][n]={'correct':int(x.sum()),'accuracy':float(x.mean()),'problem_bootstrap_95ci':np.quantile(x[draws].mean(1),[.025,.975]).tolist(),
                      'ended_with_eos':sum(a['ended_with_eos'] for a in r),'truncated':sum(not a['ended_with_eos'] for a in r),
                      'mean_completion_tokens':float(np.mean([a['completion_tokens'] for a in r]))}
 if n!='base':
  d=x-values['base'];report['paired_vs_base'][n]={'accuracy_difference':float(d.mean()),'problem_bootstrap_95ci':np.quantile(d[draws].mean(1),[.025,.975]).tolist(),
    'improved_problems':int(((x==1)&(values['base']==0)).sum()),'regressed_problems':int(((x==0)&(values['base']==1)).sum())}
(root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
