#!/usr/bin/env python3
"""Restricted-parameter REINFORCE gradient variance on matched independent continuations."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlcppo import MultiLayerCritic
p=argparse.ArgumentParser();p.add_argument('--test-dir',default='runs/deepscaler/calibration_shared/test');p.add_argument('--calibration',default='runs/deepscaler/calibration_shared/calibration_parameters.json');p.add_argument('--max-problems',type=int,default=50);p.add_argument('--samples-per-problem',type=int,default=4);p.add_argument('--output',default='runs/deepscaler/calibration_shared/gradient_variance.json');a=p.parse_args()
root=Path('runs/deepscaler/calibration_shared');parameters=json.loads(Path(a.calibration).read_text());files=sorted(Path(a.test_dir).glob('*.json'))[:a.max_problems]
if len(files)<a.max_problems or a.samples_per_problem<1:raise ValueError('Missing held-out problems')
checkpoint='runs/deepscaler/restart_final_from300_seed43/checkpoint-100';model=AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
for w in model.parameters():w.requires_grad_(False)
parameter=model.model.layers[-1].self_attn.q_proj.bias
if parameter is None:raise ValueError('No final-layer q_proj bias')
parameter.requires_grad_(True)
heads={}
for name,path in [('final_ensemble',checkpoint+'/critic.pt'),('multilayer_mean','runs/deepscaler/restart_multilayer_from200_seed43/checkpoint-200/critic.pt')]:
 saved=torch.load(path,map_location='cpu',weights_only=True);head=MultiLayerCritic(model.config.hidden_size,saved['layers'],head_size=saved['head_size'],aggregation=saved['aggregation']).cuda().eval();head.load_state_dict(saved['critic']);heads[name]=(head,saved['layers'])
prior=[]
for filename in ['runs/deepscaler/final_ensemble_seed42/rollouts.jsonl','runs/deepscaler/restart_final_from300_seed43/rollouts.jsonl']:
 for line in Path(filename).read_text().splitlines():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  if r.get('phase')=='ppo':prior.append(r['reward'])
if not prior:raise ValueError('No training rewards')
constant=float(np.mean(prior));names=['constant']+[f'{name}:{method}' for name in heads for method in ('identity','temperature','affine_logit','isotonic')];gradients={k:[] for k in names};evaluated=[]
for path in files:
 row=json.loads(path.read_text());selected=next(s for s in row['states'] if s['trajectory']==0 and s['progress']==.5);prefix=torch.tensor(selected['prefix_token_ids'],device='cuda');items=selected['continuations'][:a.samples_per_problem]
 if len(items)!=a.samples_per_problem:raise ValueError('Missing continuation samples')
 group={k:[] for k in names}
 for item in items:
  actions=torch.tensor(item['token_ids'],device='cuda');full=torch.cat([prefix,actions])[None,:]
  if not len(actions):raise ValueError('Empty continuation')
  outputs=model(input_ids=full,output_hidden_states=True,use_cache=False);start=len(prefix)-1;logits=outputs.logits[:,start:start+len(actions),:].float();logprobs=torch.log_softmax(logits,dim=-1).gather(-1,actions[None,:,None]).squeeze(-1)
  baselines={'constant':torch.full_like(logprobs,constant)}
  with torch.no_grad():
   for name,(head,layers) in heads.items():
    pred=head([outputs.hidden_states[k][:,start:start+len(actions),:].detach() for k in layers])[0].clamp(1e-6,1-1e-6);q=torch.logit(pred);config=parameters[name];baselines[name+':identity']=pred;baselines[name+':temperature']=torch.sigmoid(q/config['temperature']);baselines[name+':affine_logit']=torch.sigmoid(config['affine_a']*q+config['affine_b'])
    x=torch.tensor(config['isotonic_x'],device='cuda');y=torch.tensor(config['isotonic_y'],device='cuda');ix=torch.searchsorted(x,pred).clamp(1,len(x)-1);u=((pred-x[ix-1])/(x[ix]-x[ix-1]).clamp_min(1e-12)).clamp(0,1);baselines[name+':isotonic']=y[ix-1]+u*(y[ix]-y[ix-1])
  for j,name in enumerate(names):
   objective=(logprobs*(float(item['reward'])-baselines[name].detach())).sum();g=torch.autograd.grad(objective,parameter,retain_graph=j<len(names)-1)[0]
   if not torch.isfinite(g).all():raise RuntimeError('Nonfinite gradient')
   group[name].append(g.detach().cpu().numpy())
  del outputs,logits,logprobs,full
 for name in names:gradients[name].append(np.mean(group[name],axis=0))
 evaluated.append(row['id']);print('gradient problem',len(evaluated),'/',len(files),flush=True)
arr={name:np.stack(values) for name,values in gradients.items()};report={'policy':checkpoint,'parameter':'model.layers[-1].self_attn.q_proj.bias','objective':'on-policy Monte Carlo score gradient; sum tokens per trajectory, mean independent continuations per problem; no PPO clipping or advantage normalization','problem_count':len(files),'continuations_per_problem':a.samples_per_problem,'states':'first trajectory at 50% progress; same prefixes and continuations across all critics','constant_training_reward':constant,'problem_ids':evaluated,'methods':{},'paired_variance_differences':{},'limitations':'Variance across problem-conditional batch means combines between-prompt and within-prompt variation; parameter subset only. Saved critics have different training histories; one is transferred. Mean-gradient differences include Monte Carlo error.'}
def variance(x):return float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
for name,x in arr.items():report['methods'][name]={'gradient_variance':variance(x),'mean_gradient_norm':float(np.linalg.norm(x.mean(0))),'mean_gradient':x.mean(0).tolist()}
rng=np.random.default_rng(123)
for name in heads:
 raw=arr[name+':identity']
 for method in ('temperature','affine_logit','isotonic'):
  other=arr[name+':'+method];draws=rng.integers(0,len(files),size=(1000,len(files)));diff=[variance(other[ix])-variance(raw[ix]) for ix in draws];report['paired_variance_differences'][name+':'+method+'_minus_identity']={'difference':variance(other)-variance(raw),'problem_bootstrap_95ci':np.quantile(diff,[.025,.975]).tolist(),'mean_gradient_difference_norm':float(np.linalg.norm(other.mean(0)-raw.mean(0)))}
for method in ('identity','temperature','affine_logit','isotonic'):
 first=arr['final_ensemble:'+method];second=arr['multilayer_mean:'+method];draws=rng.integers(0,len(files),size=(1000,len(files)));diff=[variance(second[ix])-variance(first[ix]) for ix in draws]
 report['paired_variance_differences']['multilayer_minus_final:'+method]={'difference':variance(second)-variance(first),'problem_bootstrap_95ci':np.quantile(diff,[.025,.975]).tolist(),'mean_gradient_difference_norm':float(np.linalg.norm(second.mean(0)-first.mean(0)))}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
