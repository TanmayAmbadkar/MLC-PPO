#!/usr/bin/env python3
"""Small held-out diagnostic linear probe on frozen final-layer states."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
root=Path('runs/deepscaler/calibration_shared')
def load(split):
 rows=[json.loads(f.read_text()) for f in sorted((root/split).glob('*.json'))]
 if len(rows)!=50:raise ValueError(f'{split} incomplete')
 features=[];targets=[];groups=[]
 for i,r in enumerate(rows):
  for s in r['states']:
   features.append(s['final_feature']);targets.append(np.mean([c['reward'] for c in s['continuations']]));groups.append(i)
 return np.asarray(features),np.asarray(targets),np.asarray(groups)
x,y,g=load('validation');xt,yt,gt=load('test')
def fit(x,y,C):
 # Each binomial proportion is represented by weighted success/failure rows.
 labels=np.r_[np.ones(len(y)),np.zeros(len(y))];weights=np.r_[y,1-y]
 pipe=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=400,solver='lbfgs'))
 pipe.fit(np.concatenate([x,x]),labels,logisticregression__sample_weight=weights);return pipe
candidates=[.0001,.001,.01,.1];scores={}
for C in candidates:
 losses=[]
 for tr,va in GroupKFold(n_splits=5).split(x,y,g):
  model=fit(x[tr],y[tr],C);p=np.clip(model.predict_proba(x[va])[:,1],1e-8,1-1e-8);losses.append(float(-np.mean(y[va]*np.log(p)+(1-y[va])*np.log1p(-p))))
 scores[str(C)]=float(np.mean(losses))
C=min(candidates,key=lambda c:scores[str(c)]);model=fit(x,y,C);p=model.predict_proba(xt)[:,1];draw=np.random.default_rng(42).integers(0,50,size=(2000,50));ix=(draw[:,:,None]*6+np.arange(6)).reshape(2000,-1);mse=(p-yt)**2;brier=p*p-2*p*yt+yt
out={'note':'Diagnostic linear probe trained on the 50 validation problems only, with regularization selected by problem-grouped 5-fold cross-validation. PPO critics were trained on much more data; not a parameter- or training-budget-matched architecture comparison.','validation_cv_negative_log_likelihood':scores,'chosen_C':C,'test':{'mse':float(mse.mean()),'brier':float(brier.mean()),'bias':float((p-yt).mean()),'mse_problem_bootstrap_95ci':np.quantile(mse[ix].mean(1),[.025,.975]).tolist(),'brier_problem_bootstrap_95ci':np.quantile(brier[ix].mean(1),[.025,.975]).tolist()}}
(root/'linear_probe_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
