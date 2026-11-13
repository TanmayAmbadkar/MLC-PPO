"""Problem-level paired bootstrap, conditional on the trained checkpoints."""
import csv
from pathlib import Path
import numpy as np
root=Path(__file__).parent/'raw'
def load(name):
 with (root/f'{name}_math500.csv').open(newline='') as f:return {r['item_id']:float(r['reward']) for r in csv.DictReader(f)}
a=load('multilayer_all');b=load('trivial_loo');keys=sorted(a.keys()&b.keys());x=np.array([a[k]-b[k] for k in keys]);rng=np.random.default_rng(42);boot=rng.choice(x,size=(20000,len(x)),replace=True).mean(axis=1)
print(f'paired problems: {len(x)}; MLC minus trivial: {100*x.mean():+.2f} pp; 95% CI: [{100*np.quantile(boot,.025):+.2f}, {100*np.quantile(boot,.975):+.2f}] pp')
