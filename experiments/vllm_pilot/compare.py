import json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from experiment2.reward import reward
p=Path('runs/vllm_pilot'); rows=json.loads((p/'rollouts.json').read_text())
model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-1.5B-Instruct',torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
all_dev=[]; reward_checks=[]
with torch.no_grad():
 for row in rows:
  prompt=row['prompt_token_ids']; completion=row['token_ids']; tokens=torch.tensor([prompt+completion],device='cuda')
  output=model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False)
  logits=output.logits[:,len(prompt)-1:len(prompt)+len(completion)-1].float()
  new=logits.log_softmax(-1).gather(-1,torch.tensor(completion,device='cuda')[None,:,None]).squeeze()
  sample=torch.tensor(row['sampled_log_probs'],device='cuda')
  all_dev.append((new-sample).exp().sub(1).abs().cpu())
  r=reward(row['text'],row['gold']); reward_checks.append({'id':row['id'],'reward_first':r,'reward_second':reward(row['text'],row['gold'])})
  del output,logits
v=torch.cat(all_dev).float(); q=torch.quantile(v,torch.tensor([.5,.95,.99,1.]))
d={'n_tokens':v.numel(),'median_abs_ratio_deviation':q[0].item(),'p95':q[1].item(),'p99':q[2].item(),'max':q[3].item(),'fraction_over_0.2':(v>.2).float().mean().item(),'reward_deterministic':all(x['reward_first']==x['reward_second'] for x in reward_checks),'reward_checks':reward_checks}
(p/'comparison.json').write_text(json.dumps(d,indent=2)); print(json.dumps(d,indent=2))
