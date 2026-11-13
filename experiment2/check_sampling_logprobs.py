#!/usr/bin/env python3
"""Compare sampled-distribution log probabilities with recomputed scores."""
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.generation.logits_process import TopKLogitsWarper,TopPLogitsWarper
p=argparse.ArgumentParser();p.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16');p.add_argument('--mode',choices=['full','incremental'],default='full');a=p.parse_args()
root=Path('runs/experiment2');row=json.loads((root/'smoke_v2/responses.jsonl').read_text().splitlines()[0])
model_id='Qwen/Qwen2.5-Math-1.5B-Instruct';tok=AutoTokenizer.from_pretrained(model_id)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=getattr(torch,a.dtype),attn_implementation='sdpa').cuda().eval()
for p in model.parameters():p.requires_grad_(False)
ids=tok(row['formatted_prompt'],return_tensors='pt',add_special_tokens=False).input_ids.cuda()
torch.manual_seed(214);torch.cuda.manual_seed_all(214)
with torch.inference_mode():
 result=model.generate(ids,do_sample=True,temperature=.8,top_k=50,top_p=.95,max_new_tokens=32,
     pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,return_dict_in_generate=True,output_scores=True)
 actions=result.sequences[0,ids.shape[1]:]
 if a.mode=='full':
  raw=model(result.sequences,use_cache=False).logits[0,ids.shape[1]-1:ids.shape[1]-1+len(actions)].float()/.8
 else:
  current=model(ids,use_cache=True)
  logits=[]
  for action in actions:
   logits.append(current.logits[0,-1].float())
   current=model(action.reshape(1,1),past_key_values=current.past_key_values,use_cache=True)
  raw=torch.stack(logits)/.8
 processed=TopPLogitsWarper(top_p=.95)(None,TopKLogitsWarper(top_k=50)(None,raw))
 recalculated=torch.log_softmax(processed,dim=-1).gather(-1,actions[:,None]).squeeze(-1)
 sampled=torch.stack([torch.log_softmax(s[0].float(),dim=-1)[actions[i]] for i,s in enumerate(result.scores)])
 delta=recalculated-sampled
 if not torch.isfinite(delta).all(): raise ValueError('Recomputed sampler has nonfinite log-probability for a generated token')
 max_delta=float(delta.abs().max()); ratios=delta.exp()
 report={'model':model_id,'dtype':a.dtype,'mode':a.mode,'tokens':len(actions),'temperature':.8,'top_k':50,'top_p':.95,
         'max_absolute_logprob_difference':max_delta,'min_recomputed_to_generation_ratio':float(ratios.min()),
         'max_recomputed_to_generation_ratio':float(ratios.max()),
         'note':f'Single rollout; compares generation scores with a {a.mode} forward. Does not validate the older PPO training implementation.'}
 (root/f'sampler_logprob_check_{a.dtype}_{a.mode}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
 if max_delta>(.01 if a.mode=='incremental' else (.1 if a.dtype=='bfloat16' else .005)):raise ValueError('Sampler log-probability mismatch exceeds tolerance')
