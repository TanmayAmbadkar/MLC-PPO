#!/usr/bin/env python3
"""Measure frozen-actor critic-step resource costs in isolated processes."""
import argparse,json,sys,time
from pathlib import Path
import torch
from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiment1'))
from fit import Ensemble
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['shared','full'],required=True);p.add_argument('--length',type=int,default=128);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--steps',type=int,default=2);p.add_argument('--output-dir',default='runs/experiment2/resource_smoke');a=p.parse_args()
if a.length<2 or a.steps<2:raise ValueError('Need length and steps >=2 to include optimizer state')
model_id='Qwen/Qwen2.5-Math-1.5B-Instruct';tok=AutoTokenizer.from_pretrained(model_id)
row=json.loads(Path('runs/experiment2/smoke_v2/responses.jsonl').read_text().splitlines()[0])
tokens=tok(row['formatted_prompt'],add_special_tokens=False,return_tensors='pt').input_ids[0]
ids=tokens.repeat((a.length+len(tokens)-1)//len(tokens))[:a.length].repeat(a.batch_size,1).cuda()
policy=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
for parameter in policy.parameters():parameter.requires_grad_(False)
layers=[round(policy.config.num_hidden_layers*f) for f in (.5,.75,1.)];size=policy.config.hidden_size
# Compare inference with and without retaining intermediate layer outputs.
for capture in (False,True):
 with torch.no_grad():policy(ids,output_hidden_states=capture,use_cache=False)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 with torch.no_grad():sample=policy(ids,output_hidden_states=capture,use_cache=False)
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 feature_bytes=sum(sample.hidden_states[k].numel()*sample.hidden_states[k].element_size() for k in layers) if capture else 0
 if capture:with_hidden={'seconds':elapsed,'peak_bytes':torch.cuda.max_memory_allocated(),'selected_feature_bytes':feature_bytes}
 else:without_hidden={'seconds':elapsed,'peak_bytes':torch.cuda.max_memory_allocated()}
 del sample
if a.mode=='shared':
 critic=Ensemble(size,[0,1,2]).cuda();params=list(critic.parameters());optimizer=torch.optim.AdamW(params,lr=1e-4)
else:
 critic=AutoModel.from_pretrained(model_id,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().train()
 head=torch.nn.Sequential(torch.nn.LayerNorm(size),torch.nn.Linear(size,1),torch.nn.Sigmoid()).cuda()
 params=list(critic.parameters())+list(head.parameters());optimizer=torch.optim.AdamW(params,lr=1e-4)
torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
for step in range(a.steps):
 optimizer.zero_grad(set_to_none=True)
 with torch.no_grad():
  output=policy(ids,output_hidden_states=a.mode=='shared',use_cache=False)
  if a.mode=='shared':features=torch.stack([output.hidden_states[k].detach().float() for k in layers],dim=2)
  del output
 if a.mode=='shared':values=critic(features.reshape(-1,3,size)).mean(-1)
 else:values=head(critic(input_ids=ids,use_cache=False).last_hidden_state.float()).squeeze(-1)
 loss=(values-.5).square().mean();loss.backward();optimizer.step()
 del values,loss
 if a.mode=='shared':del features
torch.cuda.synchronize();elapsed=time.perf_counter()-start
report={'mode':a.mode,'model':model_id,'batch_size':a.batch_size,'sequence_length':a.length,'steps':a.steps,
        'critic_trainable_parameters':sum(x.numel() for x in params),'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'critic_step_seconds_mean':elapsed/a.steps,
        'policy_tokens_per_second_including_critic_update':a.batch_size*a.length*a.steps/elapsed,
        'policy_forward_without_hidden':without_hidden,'policy_forward_with_hidden':with_hidden,
        'scope':'Frozen policy forward plus critic optimizer steps; this is not peak memory of a complete PPO actor update.'}
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);(out/(a.mode+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
