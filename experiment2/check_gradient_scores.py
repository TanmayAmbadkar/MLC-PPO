#!/usr/bin/env python3
"""Check cached score gradients against actual sampler scores with autograd enabled."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.generation.logits_process import TopKLogitsWarper,TopPLogitsWarper
root=Path('runs/experiment2');row=json.loads((root/'smoke_v2/responses.jsonl').read_text().splitlines()[0])
model_id='Qwen/Qwen2.5-Math-1.5B-Instruct';tok=AutoTokenizer.from_pretrained(model_id)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
for p in model.parameters():p.requires_grad_(False)
parameter=model.model.layers[-1].self_attn.q_proj.bias;parameter.requires_grad_(True)
ids=tok(row['formatted_prompt'],return_tensors='pt',add_special_tokens=False).input_ids.cuda()
torch.manual_seed(214);torch.cuda.manual_seed_all(214)
with torch.no_grad():
 result=model.generate(ids,do_sample=True,temperature=.8,top_k=50,top_p=.95,max_new_tokens=32,
     pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,return_dict_in_generate=True,output_scores=True)
 actions=result.sequences[0,ids.shape[1]:].clone()
 generated=[float(torch.log_softmax(s[0].float(),dim=-1)[actions[i]]) for i,s in enumerate(result.scores)]
cache=None;context=ids;recomputed=[];nonzero=[];support_sizes=[]
for token in actions:
 output=model(context,past_key_values=cache,use_cache=True,output_hidden_states=True)
 scores=output.logits[:,-1].float()/.8
 scores=TopPLogitsWarper(top_p=.95)(None,TopKLogitsWarper(top_k=50)(None,scores))
 support_sizes.append(int(torch.isfinite(scores).sum().item()))
 logprob=torch.log_softmax(scores,dim=-1)[0,token]
 if not torch.isfinite(logprob):raise ValueError('Sampled token absent from recomputed distribution')
 g=torch.autograd.grad(logprob,parameter)[0]
 recomputed.append(float(logprob.detach()));nonzero.append(bool(torch.count_nonzero(g).item()))
 cache=output.past_key_values
 for layer in cache.layers:
  if layer.keys is not None:layer.keys=layer.keys.detach()
  if layer.values is not None:layer.values=layer.values.detach()
 context=token.reshape(1,1)
deltas=torch.tensor(recomputed)-torch.tensor(generated)
report={'tokens':len(actions),'max_absolute_logprob_difference':float(deltas.abs().max()),
        'ratio_min':float(deltas.exp().min()),'ratio_max':float(deltas.exp().max()),
        'nonzero_token_gradients':sum(nonzero),'support_sizes':support_sizes,
        'nondeterministic_tokens_with_zero_gradient':sum(size>1 and not nz for size,nz in zip(support_sizes,nonzero)),
        'mode':'bfloat16 incremental with q_proj.bias gradients enabled'}
(root/'gradient_score_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if report['max_absolute_logprob_difference']>.01 or not any(nonzero):raise ValueError('Gradient-path score mismatch or all gradients zero')
