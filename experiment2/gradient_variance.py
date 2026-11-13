#!/usr/bin/env python3
"""Compare detached-baseline score-gradient variance on independent held-out prompts."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiment1'))
from fit import Ensemble

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--difficulty-dir',default='runs/experiment2/gradient_data',help='Independent single-trajectory generation records with stored sampling log probabilities');p.add_argument('--results-dir',default='runs/experiment2/results')
p.add_argument('--model',default='Qwen/Qwen2.5-Math-1.5B-Instruct');p.add_argument('--seed',type=int,default=42)
p.add_argument('--max-problems',type=int,default=64);p.add_argument('--batch-size',type=int,default=2)
a=p.parse_args();root=Path(a.difficulty_dir);out=Path(a.results_dir);out.mkdir(parents=True,exist_ok=True)
config=json.loads((root/'config.json').read_text())
if config['model']!=a.model:raise ValueError('Model differs from difficulty generation')
if a.batch_size<1:raise ValueError('Batch size must be positive')
records=[json.loads(path.read_text()) for path in sorted((root/'test').glob('*.json'))]
np.random.default_rng(42).shuffle(records);records=records[:a.max_problems]
if len(records)<2*a.batch_size:raise ValueError('Need at least two independent evaluation batches')
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
for par in model.parameters():par.requires_grad_(False)
parameter=model.model.layers[-1].self_attn.q_proj.bias
if parameter is None:raise ValueError('Selected final-layer query bias is unavailable')
parameter.requires_grad_(True)
size=model.config.hidden_size;layers=[round(model.config.num_hidden_layers*f) for f in (.5,.75,1.)]
heads={}
for name,locations in [('final_head',[2]),('final_ensemble',[2,2,2]),('multilayer_mean',[0,1,2])]:
 head=Ensemble(size,locations).cuda().eval()
 head.load_state_dict(torch.load(out/f'{name}_seed{a.seed}.pt',map_location='cuda',weights_only=True))
 for par in head.parameters():par.requires_grad_(False)
 heads[name]=head
metrics=json.loads((out/'metrics.json').read_text());constant=metrics['constant_success_rate']
warpers=[TopKLogitsWarper(top_k=50),TopPLogitsWarper(top_p=config['top_p'])]
gradients={name:[] for name in ['constant',*heads]};skipped=[];score_differences=[]
for row in records:
 response=row['responses'][0]
 if 'token_ids' not in response or 'prompt_token_ids' not in row:raise ValueError('Difficulty samples must contain exact token IDs')
 prompt=torch.tensor(row['prompt_token_ids'],device='cuda',dtype=torch.long)
 actions=torch.tensor(response['token_ids'],device='cuda',dtype=torch.long)
 sampled=response.get('sampled_log_probs')
 if sampled is None or len(sampled)!=len(actions):raise ValueError('Missing original generation log probabilities')
 if actions.numel()==0:skipped.append({'id':row['problem_id'],'reason':'empty response'});continue
 cache=None;context=prompt[None,:]
 accumulators={name:torch.zeros_like(parameter,dtype=torch.float32) for name in gradients}
 valid=True
 differences=[]
 for token_index,action in enumerate(actions):
  outputs=model(context,past_key_values=cache,output_hidden_states=True,use_cache=True)
  features=torch.stack([outputs.hidden_states[k][0,-1].detach().float() for k in layers],dim=0)[None,:,:]
  with torch.no_grad():
   baselines={'constant':constant}
   baselines.update({name:float(head(features).mean().item()) for name,head in heads.items()})
  scores=outputs.logits[:,-1].float()/config['temperature']
  for warper in warpers:scores=warper(None,scores)
  log_prob=torch.log_softmax(scores,dim=-1)[0,action]
  if not torch.isfinite(log_prob):
   skipped.append({'id':row['problem_id'],'reason':'sampled token outside cached top-k/top-p support'})
   valid=False
   break
  differences.append(abs(float(log_prob.detach())-float(sampled[token_index])))
  score_gradient=torch.autograd.grad(log_prob,parameter)[0].detach().float()
  for name,baseline in baselines.items():
   accumulators[name].add_(score_gradient,alpha=float(response['reward'])-baseline)
  cache=outputs.past_key_values
  # q_proj bias does not affect cached keys or values, but explicitly detach
  # to prevent retaining the prior token's computation graph.
  for layer in cache.layers:
   if layer.keys is not None:layer.keys=layer.keys.detach()
   if layer.values is not None:layer.values=layer.values.detach()
  context=action.reshape(1,1)
 if not valid:continue
 max_difference=max(differences,default=0.)
 if max_difference>.01:
  skipped.append({'id':row['problem_id'],'reason':'cached log probabilities do not match generation','max_absolute_difference':max_difference})
  continue
 score_differences.append(max_difference)
 for name,gradient in accumulators.items():gradients[name].append(gradient.cpu().numpy())
 print('evaluated',len(gradients['constant']),'skipped',len(skipped),flush=True)
 del outputs,features,cache
count=len(gradients['constant']);nbatches=count//a.batch_size
if nbatches<2:raise RuntimeError('Not enough valid independent gradient batches')
report={'model':a.model,'critic_seed':a.seed,'selected_parameter':'model.layers[-1].self_attn.q_proj.bias',
        'sampling':{'temperature':config['temperature'],'top_p':config['top_p'],'top_k':50},
        'normalization':'sum token contributions per trajectory; average independent problems per batch',
        'evaluation_problems':count,'batch_size':a.batch_size,'batches':nbatches,'skipped':skipped,'methods':{},
        'max_absolute_generation_logprob_difference':max(score_differences,default=None),
        'score_reconstruction':'incremental KV-cache forwards, using the same sampler and precision as generation',
        'limitation':'Restricted to the final-layer query-projection bias; this is not full-policy gradient variance.'}
batched={}
for name,items in gradients.items():
 x=np.stack(items[:nbatches*a.batch_size]).reshape(nbatches,a.batch_size,-1).mean(1)
 batched[name]=x
 center=x.mean(0); report['methods'][name]={'gradient_variance':float(np.mean(np.sum((x-center)**2,axis=1))),
                                               'mean_gradient_norm':float(np.linalg.norm(center))}
rng=np.random.default_rng(42)
paired=[]
for _ in range(1000):
 ix=rng.integers(0,nbatches,size=nbatches)
 def variance(x):
  sample=x[ix];return np.mean(np.sum((sample-sample.mean(0))**2,axis=1))
 paired.append(variance(batched['multilayer_mean'])-variance(batched['final_ensemble']))
report['paired_multilayer_minus_final_ensemble_variance']={'difference':report['methods']['multilayer_mean']['gradient_variance']-report['methods']['final_ensemble']['gradient_variance'],
 'batch_bootstrap_95ci':np.quantile(paired,[.025,.975]).tolist()}
(out/f'gradient_variance_seed{a.seed}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
