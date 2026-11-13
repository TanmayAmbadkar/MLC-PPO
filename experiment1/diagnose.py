#!/usr/bin/env python3
"""Inspect held-out calibration and difficult problems from Experiment 1."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from fit import Ensemble, load

root = Path('runs/experiment1')
data = load(root / 'data/test')
out = root / 'diagnostics'
out.mkdir(parents=True, exist_ok=True)
size = data['x'].shape[-1]
configs = {'final_head': ([2],128), 'final_ensemble': ([2,2,2],128),
           'multilayer_mean': ([0,1,2],128), 'final_large': ([2],384)}
preds = {}
with torch.no_grad():
    for name,(layers,width) in configs.items():
        model = Ensemble(size,layers,width)
        model.load_state_dict(torch.load(root / 'results' / f'{name}.pt', map_location='cpu', weights_only=True))
        model.eval()
        values = model(data['x'])
        preds[name] = values.mean(-1).numpy()
        if name == 'multilayer_mean':
            preds['multilayer_min'] = values.min(-1).values.numpy()
            gate = nn.Linear(3,3)
            gate.load_state_dict(torch.load(root / 'results/gate.pt', map_location='cpu', weights_only=True))
            gate.eval()
            preds['multilayer_gate'] = (torch.softmax(gate(values), -1)*values).sum(-1).numpy()
ids, progress, observed = data['problem'], data['progress'], data['y'].numpy()
means = {int(i): float(observed[ids == i].mean()) for i in np.unique(ids)}
low = sorted(i for i,v in means.items() if v < 1/3)
report = {'low_success_problem_ids': low, 'definition': 'problem mean of independent continuation success proportions below 1/3', 'methods': {}, 'problems': []}
mask = np.isin(ids, low)
for name,pred in preds.items():
    p = pred[mask]; y = observed[mask]
    bins = []
    for lower in np.arange(0,1,0.1):
        upper = lower+0.1
        b = (pred >= lower) & ((pred < upper) if upper < 0.999 else (pred <= 1))
        if b.any():
            bins.append({'range': [round(float(lower),1),round(float(upper),1)], 'states':int(b.sum()),
                         'mean_prediction':float(pred[b].mean()), 'observed_success':float(observed[b].mean())})
    report['methods'][name] = {'low_mean_prediction':float(p.mean()), 'low_mean_observed':float(y.mean()),
        'low_prediction_quantiles':np.quantile(p,[0,0.25,0.5,0.75,1]).tolist(),
        'low_by_progress':{str(f):{'mean_prediction':float(pred[mask & (progress==f)].mean()),
                                  'mean_observed':float(observed[mask & (progress==f)].mean()),
                                  'mse':float(((pred[mask & (progress==f)]-observed[mask & (progress==f)])**2).mean())}
                           for f in (0.25,0.5,0.75)}, 'calibration_bins_all_test_states': bins}
for problem in low:
    m = ids == problem
    report['problems'].append({'problem_id':problem, 'states':int(m.sum()), 'observed_success':means[problem],
                               'mean_predictions':{name:float(p[m].mean()) for name,p in preds.items()}})
(out/'diagnostic.json').write_text(json.dumps(report, indent=2))
with (out/'low_success_states.csv').open('w', newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['problem_id','progress','observed_success',*preds])
    for j in np.flatnonzero(mask):
        writer.writerow([int(ids[j]),float(progress[j]),float(observed[j]),*[float(p[j]) for p in preds.values()]])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,6))
ax.plot([0,1],[0,1], '--', color='gray', label='Ideal')
for name in ('final_head','final_ensemble','final_large','multilayer_mean','multilayer_min','multilayer_gate'):
    bins=report['methods'][name]['calibration_bins_all_test_states']
    ax.plot([b['mean_prediction'] for b in bins], [b['observed_success'] for b in bins], marker='o', label=name)
ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted success probability',ylabel='Observed continuation success',title='Held-out state calibration')
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'calibration.png',dpi=160)
print(json.dumps(report,indent=2))
