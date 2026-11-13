#!/usr/bin/env python3
"""Validation-only calibration and problem-clustered test evaluation."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from sklearn.isotonic import IsotonicRegression
root=Path('runs/deepscaler/calibration_shared');data={}
for split in ('validation','test'):
 rows=[json.loads(f.read_text()) for f in sorted((root/split).glob('*.json'))]
 if len(rows)!=50 or len({r['id'] for r in rows})!=50:raise ValueError(f'{split} incomplete: {len(rows)}')
 if any(len(r['states'])!=6 or len(r['difficulty_rollouts'])!=8 or any(len(s['continuations'])!=32 for s in r['states']) for r in rows):raise ValueError(f'{split} incomplete state')
 data[split]=rows
if {r['id'] for r in data['validation']}&{r['id'] for r in data['test']}:raise ValueError('Validation/test leakage')
models={'final_ensemble':'value','multilayer_mean':'peer_value'};parameters={};report={'policy':'final_ensemble PPO checkpoint','note':'Both trained heads evaluated on the same final-ensemble policy states; multi-layer head is transferred across policies, not a controlled training architecture comparison. Eight independent prompt outcomes form coarse difficulty groups. MSE includes finite-rollout noise; calibration and ECE are test-only.','models':{}}
training_rewards=[]
for path in ('runs/deepscaler/final_ensemble_seed42/rollouts.jsonl','runs/deepscaler/restart_final_from300_seed43/rollouts.jsonl'):
 for line in Path(path).read_text().splitlines():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  if r.get('phase')=='ppo':training_rewards.append(r['reward'])
if not training_rewards:raise ValueError('Missing training rewards for constant predictor')
constant=float(np.mean(training_rewards));report['training_reward_constant']=constant
def flat(rows,key):
 states=[s for r in rows for s in r['states']];return np.array([s[key] for s in states]),np.array([np.mean([c['reward'] for c in s['continuations']]) for s in states])
def logits(p):
 q=np.clip(p,1e-6,1-1e-6);return np.log(q/(1-q))
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))
def evaluate(p,y):
 bins=[];ece=0
 for i in range(10):
  select=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1));n=int(select.sum())
  if n:
   pred=float(p[select].mean());obs=float(y[select].mean());ece+=n/len(p)*abs(pred-obs);bins.append({'bin':i,'states':n,'predicted':pred,'observed':obs})
 return {'mse':float(np.mean((p-y)**2)),'brier':float(np.mean(p*p-2*p*y+y)),'bias':float(np.mean(p-y)),'ece_10bin':float(ece),'calibration_bins':bins}
for name,key in models.items():
 pv,yv=flat(data['validation'],key);pt,yt=flat(data['test'],key)
 def nll(logt):
  q=np.clip(sigmoid(logits(pv)/np.exp(logt)),1e-8,1-1e-8);return -np.mean(yv*np.log(q)+(1-yv)*np.log1p(-q))
 res=minimize_scalar(nll,bounds=(-3,3),method='bounded');temp=float(np.exp(res.x));iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(pv,yv)
 def affine_nll(ab):
  q=np.clip(sigmoid(ab[0]*logits(pv)+ab[1]),1e-8,1-1e-8);return -np.mean(yv*np.log(q)+(1-yv)*np.log1p(-q))
 affine=minimize(affine_nll,[1.,0.],method='L-BFGS-B',bounds=[(-20,20),(-20,20)])
 if not affine.success:raise RuntimeError(f'Affine calibration failed: {affine.message}')
 a,b=map(float,affine.x)
 parameters[name]={'temperature':temp,'affine_a':a,'affine_b':b,'isotonic_x':iso.X_thresholds_.tolist(),'isotonic_y':iso.y_thresholds_.tolist()}
 methods={'constant':lambda p:np.full_like(p,constant),'identity':lambda p:p,'temperature':lambda p:sigmoid(logits(p)/temp),'affine_logit':lambda p:sigmoid(a*logits(p)+b),'isotonic':lambda p:iso.predict(p)};result={}
 for method,fn in methods.items():
  v=fn(pt);entry=evaluate(v,yt);rng=np.random.default_rng(42);draw=rng.integers(0,50,size=(2000,50));pairs=[]
  for indices in draw:
   ix=(indices[:,None]*6+np.arange(6)).ravel();pairs.append([np.mean((v[ix]-yt[ix])**2),np.mean(v[ix]**2-2*v[ix]*yt[ix]+yt[ix])])
  ci=np.quantile(np.array(pairs),[.025,.975],axis=0).T.tolist();entry['problem_bootstrap_95ci']={'mse':ci[0],'brier':ci[1]};entry['by_difficulty']={}
  for group,check in [('0_of_8',lambda n:n==0),('1_to_7_of_8',lambda n:0<n<8),('8_of_8',lambda n:n==8)]:
   group_ids=[i for i,r in enumerate(data['test']) if check(sum(c['reward'] for c in r['difficulty_rollouts']))];ix=np.array([i*6+j for i in group_ids for j in range(6)],dtype=int);g={'problems':len(group_ids),'metrics':evaluate(v[ix],yt[ix]) if len(ix) else None}
   if group_ids:
    samples=rng.choice(group_ids,size=(1000,len(group_ids)),replace=True);gi=(samples[:,:,None]*6+np.arange(6)).reshape(1000,-1);g['bias_problem_bootstrap_95ci']=np.quantile((v[gi]-yt[gi]).mean(1),[.025,.975]).tolist()
   entry['by_difficulty'][group]=g
  result[method]=entry
 for method in ('temperature','affine_logit','isotonic'):
  v=methods[method](pt);un=methods['identity'](pt);draw=np.random.default_rng(47).integers(0,50,size=(2000,50));ix=(draw[:,:,None]*6+np.arange(6)).reshape(2000,-1);diff=(v*v-2*v*yt)-(un*un-2*un*yt)
  result[method]['brier_minus_identity']={'difference':float(diff.mean()),'problem_bootstrap_95ci':np.quantile(diff[ix].mean(1),[.025,.975]).tolist()}
 report['models'][name]={'temperature':temp,'validation_only_fit':True,'test_prediction_histogram':np.histogram(pt,bins=np.linspace(0,1,11))[0].tolist(),'test_mc_value_histogram':np.histogram(yt,bins=np.linspace(0,1,11))[0].tolist(),'methods':result}
(root/'calibration_parameters.json').write_text(json.dumps(parameters,indent=2));(root/'calibration_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
