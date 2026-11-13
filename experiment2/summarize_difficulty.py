#!/usr/bin/env python3
"""Summarize independent prompt-level difficulty samples and truncation rates."""
import json
from pathlib import Path
from collections import Counter
root=Path('runs/experiment2/difficulty');rows=[]
if json.loads((root/'config.json').read_text())['samples'] != 8:raise ValueError('Expected eight rollouts per problem')
for split in ('train','validation','test'):
    rows.extend(json.loads(p.read_text()) for p in (root/split).glob('*.json'))
if not rows:raise ValueError('No completed difficulty samples')
ids=[r['problem_id'] for r in rows]
if len(ids)!=len(set(ids)):raise ValueError('Duplicate problem IDs across splits')
counts=Counter(str(int(round(r['success_rate']*8))) for r in rows)
rate={str(r['problem_id']):r['success_rate'] for r in rows}
(root/'difficulty_map.json').write_text(json.dumps(rate,indent=2))
report={'problems':len(rows),'by_split':dict(Counter(r['split'] for r in rows)),
        'successes_out_of_8':{str(i):counts.get(str(i),0) for i in range(9)},
        'responses':sum(len(r['responses']) for r in rows),
        'responses_without_eos':sum(not x['ended_with_eos'] for r in rows for x in r['responses']),
        'warning':'A missing EOS can indicate truncation. Zero reward on a truncated answer is not evidence of mathematical failure.'}
(root/'summary.json').write_text(json.dumps(report,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots();ax.bar(range(9),[counts.get(str(i),0) for i in range(9)]);ax.set(xlabel='Successful responses out of 8',ylabel='Problems');fig.tight_layout();fig.savefig(root/'difficulty.png',dpi=160)
print(json.dumps(report,indent=2))
