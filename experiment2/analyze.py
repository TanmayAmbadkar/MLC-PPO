#!/usr/bin/env python3
"""Compare frozen-feature critics with paired problem-level uncertainty intervals."""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiment1'))
from fit import Ensemble, load, train

p=argparse.ArgumentParser();p.add_argument('--data-dir',default='runs/experiment2/states');p.add_argument('--output-dir',default='runs/experiment2/results')
p.add_argument('--epochs',type=int,default=150);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seeds',default='42,43,44');p.add_argument('--difficulty-file');p.add_argument('--require-counts',action='store_true')
a=p.parse_args();root=Path(a.data_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
tr,val,test=[load(root/s) for s in ('train','val','test')]
sets=[set(x['problem'].tolist()) for x in (tr,val,test)]
if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:raise ValueError('Problem leakage between splits')
if a.require_counts:
    for name,data,n in [('train',tr,1000),('validation',val,200),('test',test,200)]:
        if len(np.unique(data['problem']))!=n or len(data['y'])!=6*n:
            raise ValueError(f'Incomplete {name}: expected {n} problems and {6*n} states')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');size=tr['x'].shape[-1]
problem_ids=test['problem']; unique=np.unique(problem_ids); rng=np.random.default_rng(42)
draws=[rng.choice(unique,len(unique),replace=True) for _ in range(a.bootstrap)]
indices=[np.concatenate([np.flatnonzero(problem_ids==i) for i in draw]) for draw in draws]
y=test['y'].numpy(); constant=float(tr['y'].mean());difficulty={}
test_x=test['x'].to(device)
def measure_head_forward(fn):
    with torch.no_grad():
        fn()
        if device.type=='cuda':
            torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        for _ in range(10):fn()
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
    return {'states_per_second':10*len(test_x)/elapsed,
            'peak_extra_allocated_bytes':max(0,torch.cuda.max_memory_allocated()-before) if device.type=='cuda' else None,
            'scope':'Cached features and critic forward only; transformer excluded'}
if a.difficulty_file:
    difficulty=json.loads(Path(a.difficulty_file).read_text())
    missing=[str(i) for i in unique if str(i) not in difficulty]
    if missing:raise ValueError(f'Missing independent test difficulty for {len(missing)} problems')
# The gate is fit on validation problems withheld from head fitting and early stopping.
val_problems=np.unique(val['problem']); gate_rng=np.random.default_rng(31415);gate_rng.shuffle(val_problems)
gate_ids=val_problems[:max(1,len(val_problems)//2)];gate_mask=np.isin(val['problem'],gate_ids);select_mask=~gate_mask
if not select_mask.any():raise ValueError('Need at least two validation problems for independent gate fitting')
report={'split_counts':{name:{'problems':len(np.unique(data['problem'])),'states':len(data['y'])} for name,data in [('train',tr),('val',val),('test',test)]},'constant_success_rate':constant,'seeds':{},'gate_validation_problem_ids':gate_ids.tolist(),'difficulty_note':'Groups use independent prompt-level rollouts, never the state continuations being evaluated.'}
for seed in [int(x) for x in a.seeds.split(',')]:
    random.seed(seed);torch.manual_seed(seed);np.random.seed(seed)
    predictions={'constant':np.full(len(y),constant)};params={'constant':0};costs={};multi_model=None
    for name,layers in [('final_head',[2]),('final_ensemble',[2,2,2]),('multilayer_mean',[0,1,2])]:
        torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        model=train(Ensemble(size,layers),tr['x'],tr['y'],val['x'][select_mask],val['y'][select_mask],a.epochs,1e-4,device)
        if device.type=='cuda':torch.cuda.synchronize()
        costs[name]={'training_wall_seconds':time.perf_counter()-started,
                     'peak_allocated_bytes_heads_only':torch.cuda.max_memory_allocated() if device.type=='cuda' else None}
        torch.save(model.state_dict(),out/f'{name}_seed{seed}.pt')
        with torch.no_grad():
            per_head=model(test_x)
            predictions[name]=per_head.mean(-1).cpu().numpy()
            if name=='multilayer_mean':
                predictions['multilayer_min']=per_head.min(-1).values.cpu().numpy()
                multi_model=model
        params[name]=sum(x.numel() for x in model.parameters())
        costs[name]['cached_feature_forward']=measure_head_forward(lambda: model(test_x).mean(-1))
    params['multilayer_min']=params['multilayer_mean']
    # Fit the gate only on separate validation problems and choose its epoch
    # on the other half of validation; test outcomes are never used here.
    with torch.no_grad():
        gate_train=multi_model(val['x'][gate_mask].to(device)).detach()
        gate_select=multi_model(val['x'][select_mask].to(device)).detach()
        test_heads=multi_model(test_x).detach()
    torch.manual_seed(seed+10000);gate=nn.Linear(3,3).to(device);optimizer=torch.optim.AdamW(gate.parameters(),lr=1e-4)
    best=float('inf');gate_state=None;stale=0
    for _ in range(a.epochs):
        optimizer.zero_grad()
        weighted=(torch.softmax(gate(gate_train),-1)*gate_train).sum(-1)
        loss=(weighted-val['y'][gate_mask].to(device)).square().mean();loss.backward();optimizer.step()
        with torch.no_grad():
            prediction=(torch.softmax(gate(gate_select),-1)*gate_select).sum(-1)
            score=(prediction-val['y'][select_mask].to(device)).square().mean().item()
        if score<best-1e-6:
            best=score;gate_state={k:v.detach().cpu().clone() for k,v in gate.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=15:break
    gate.load_state_dict(gate_state);torch.save(gate.state_dict(),out/f'multilayer_gate_seed{seed}.pt')
    with torch.no_grad():predictions['multilayer_gate']=(torch.softmax(gate(test_heads),-1)*test_heads).sum(-1).cpu().numpy()
    params['multilayer_gate']=params['multilayer_mean']+sum(x.numel() for x in gate.parameters())
    costs['multilayer_min']={'cached_feature_forward':measure_head_forward(lambda: multi_model(test_x).min(-1).values)}
    costs['multilayer_gate']={'cached_feature_forward':measure_head_forward(lambda: (lambda values:(torch.softmax(gate(values),-1)*values).sum(-1))(multi_model(test_x)))}
    seed_report={'parameter_counts':params,'critic_training_costs':costs,'methods':{}}
    for name,pred in predictions.items():
        error=(pred-y)**2;brier=np.array([np.mean((p-r)**2) for p,r in zip(pred,test['rewards'])])
        bins=[];ece=0.
        for lower in np.arange(0,1,.1):
            upper=lower+.1;mask=(pred>=lower)&((pred<upper) if upper<.999 else (pred<=1))
            if mask.any():
                predicted=float(pred[mask].mean());observed=float(y[mask].mean());ece+=float(mask.mean())*abs(predicted-observed)
                bins.append({'lower':float(lower),'upper':float(upper),'states':int(mask.sum()),'predicted':predicted,'observed':observed})
        groups={}
        for name_group in ('zero_of_8','mixed_1_to_7','eight_of_8'):
            mask=np.array([str(i) in difficulty and ((difficulty[str(i)]==0) if name_group=='zero_of_8' else (difficulty[str(i)]==1 if name_group=='eight_of_8' else 0<difficulty[str(i)]<1)) for i in problem_ids])
            group_bins=[];group_ece=0.
            if mask.any():
                for lower in np.arange(0,1,.1):
                    upper=lower+.1;bm=mask&(pred>=lower)&((pred<upper) if upper<.999 else (pred<=1))
                    if bm.any():
                        predicted=float(pred[bm].mean());observed=float(y[bm].mean())
                        group_ece+=float(bm.sum()/mask.sum())*abs(predicted-observed)
                        group_bins.append({'lower':float(lower),'upper':float(upper),'states':int(bm.sum()),'predicted':predicted,'observed':observed})
            groups[name_group]={'states':int(mask.sum()),'problems':len(np.unique(problem_ids[mask])),
                'mse':float(error[mask].mean()) if mask.any() else None,
                'brier':float(brier[mask].mean()) if mask.any() else None,
                'mean_predicted':float(pred[mask].mean()) if mask.any() else None,
                'mean_observed':float(y[mask].mean()) if mask.any() else None,
                'prediction_bias':float((pred[mask]-y[mask]).mean()) if mask.any() else None,
                'calibration_ece_10bin':group_ece if mask.any() else None,'calibration_bins':group_bins}
        seed_report['methods'][name]={'mse':float(error.mean()),'mse_problem_bootstrap_95ci':np.quantile([error[ix].mean() for ix in indices],[.025,.975]).tolist(),
          'brier':float(brier.mean()),'prediction_bias':float((pred-y).mean()),'calibration_ece_10bin':ece,'calibration_bins':bins,
          'mse_by_progress':{str(f):float(error[test['progress']==f].mean()) for f in (.25,.5,.75)},'mse_by_independent_difficulty':groups}
    delta=(predictions['multilayer_mean']-y)**2-(predictions['final_ensemble']-y)**2
    seed_report['paired_multilayer_minus_final_ensemble']={'mse_difference':float(delta.mean()),'problem_bootstrap_95ci':np.quantile([delta[ix].mean() for ix in indices],[.025,.975]).tolist()}
    groups={}
    for label in ('zero_of_8','mixed_1_to_7','eight_of_8'):
        group_ids=[i for i in unique if str(i) in difficulty and ((difficulty[str(i)]==0) if label=='zero_of_8' else (difficulty[str(i)]==1 if label=='eight_of_8' else 0<difficulty[str(i)]<1))]
        if not group_ids:
            groups[label]={'problems':0,'mse_difference':None,'problem_bootstrap_95ci':None};continue
        mask=np.isin(problem_ids,group_ids);r=np.random.default_rng(42)
        samples=[np.concatenate([delta[problem_ids==i] for i in r.choice(group_ids,len(group_ids),replace=True)]).mean() for _ in range(a.bootstrap)]
        groups[label]={'problems':len(group_ids),'mse_difference':float(delta[mask].mean()),'problem_bootstrap_95ci':np.quantile(samples,[.025,.975]).tolist()}
    seed_report['paired_multilayer_minus_final_ensemble_by_difficulty']=groups
    for name in ('multilayer_min','multilayer_gate'):
        delta_other=(predictions[name]-y)**2-(predictions['final_ensemble']-y)**2
        seed_report['paired_'+name+'_minus_final_ensemble']={'mse_difference':float(delta_other.mean()),'problem_bootstrap_95ci':np.quantile([delta_other[ix].mean() for ix in indices],[.025,.975]).tolist()}
    np.savez_compressed(out/f'predictions_seed{seed}.npz',problem_ids=problem_ids,progress=test['progress'],mc_target=y,**predictions)
    report['seeds'][str(seed)]=seed_report
(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
