#!/usr/bin/env python3
"""Exercise real cached sampling and PPO scoring on unequal left-padded prompts."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlcppo import MultiLayerCritic
from train_mlc_ppo import build_rollout,extract_features,position_ids_from_mask,response_mask,sequence_log_probs

torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.manual_seed(782)
model_id='Qwen/Qwen2.5-Math-1.5B-Instruct'
tok=AutoTokenizer.from_pretrained(model_id);tok.padding_side='left'
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
for p in model.parameters():p.requires_grad_(False)
parameter=model.model.layers[-1].self_attn.q_proj.bias
if parameter is None: raise RuntimeError('No query projection bias')
parameter.requires_grad_(True)
layers=[round(model.config.num_hidden_layers*f) for f in (.5,.75,1.)]
critic=MultiLayerCritic(model.config.hidden_size,layers,head_size=32).cuda()
batch=[{'prompt':'What is 1+1?','answer':'2'}, {'prompt':'Explain carefully what remains after subtracting four from fifteen, and give a numeric final answer.','answer':'11'}]
results=[]
for temperature in (1.0,0.7):
 args=SimpleNamespace(model_id=model_id,max_prompt_length=512,max_completion_length=24,
                      generation_temperature=temperature,generation_top_p=1.0,reward_type='exact_match')
 rollout=build_rollout(model,tok,critic,batch,torch.device('cuda'),args,layers)
 output=model(rollout['input_ids'],attention_mask=rollout['attention_mask'],
              position_ids=position_ids_from_mask(rollout['attention_mask']),output_hidden_states=True,use_cache=False)
 start=rollout['prompt_width']-1
 logits=output.logits[:,start:start+rollout['completions'].shape[1]]
 recomputed=sequence_log_probs(logits,rollout['completions'],temperature)
 ratio=torch.exp(recomputed-rollout['old_log_probs'])
 error=(ratio[rollout['mask']]-1).abs().max().item()
 assert error<0.01, f'unchanged-policy ratio error {error}'
 assert rollout['old_log_probs'].grad_fn is None and rollout['advantages'].grad_fn is None
 # Actor loss reaches the selected policy parameter while the detached critic stays untouched.
 model.zero_grad(set_to_none=True);critic.zero_grad(set_to_none=True)
 loss=-(recomputed*rollout['mask']).sum()/rollout['mask'].sum()
 loss.backward()
 assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
 assert all(p.grad is None for p in critic.parameters())
 # Critic loss cannot update the backbone, even with features extracted from an actor forward.
 model.zero_grad(set_to_none=True);critic.zero_grad(set_to_none=True)
 features=extract_features(output,layers,start,rollout['completions'].shape[1])
 baseline,_,_=critic(features)
 baseline.square().mean().backward()
 assert parameter.grad is None and any(p.grad is not None for p in critic.parameters())
 results.append({'temperature':temperature,'max_unchanged_ratio_error':error,
                 'valid_actions':int(rollout['mask'].sum())})
# A sampled pad token before EOS is an action, while post-EOS padding is masked.
mask=response_mask(torch.tensor([[7,0,8,2,0],[8,9,2,0,0]]),eos_id=2,pad_id=0)
assert mask.tolist()==[[True,True,True,True,False],[True,True,True,False,False]]
report={'model':model_id,'dtype':'float32','sampling':'top_k=0, top_p=1','checks':results,
        'critic_gradients_detached':True,'pad_before_eos_retained':True}
out=Path('runs/ppo_ratio_check');out.mkdir(parents=True,exist_ok=True)
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
