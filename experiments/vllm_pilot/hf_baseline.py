import json, os, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
p=Path('runs/vllm_pilot'); rows=json.loads((p/'rollouts.json').read_text()); model_id='Qwen/Qwen2.5-Math-1.5B-Instruct'
dtype=os.environ.get('PILOT_DTYPE','float32'); model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=getattr(torch,dtype),device_map='cuda:0').eval(); tok=AutoTokenizer.from_pretrained(model_id); tok.padding_side='left'; tok.pad_token=tok.eos_token
config=GenerationConfig(do_sample=True,temperature=1,top_p=1,top_k=0,max_new_tokens=512,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,return_dict_in_generate=True,output_scores=True)
def run(batch):
 enc=tok.pad([{'input_ids':x['prompt_token_ids']} for x in batch],return_tensors='pt').to('cuda')
 with torch.no_grad():
  out=model.generate(**enc,generation_config=config)
  comp=out.sequences[:,enc.input_ids.shape[1]:]
  logp=torch.stack([torch.log_softmax(s.float(),-1).gather(-1,comp[:,t,None]).squeeze(-1) for t,s in enumerate(out.scores)],dim=1)
  mask=(comp==tok.eos_token_id).int().cumsum(-1).le(1) & ((comp != tok.eos_token_id) | (comp==tok.eos_token_id).int().cumsum(-1).eq(1))
  return int(mask.sum().item())
with torch.no_grad():
 enc=tok.pad([{'input_ids':rows[0]['prompt_token_ids']}],return_tensors='pt').to('cuda'); model.generate(**enc,max_new_tokens=8,do_sample=True,pad_token_id=tok.pad_token_id)
torch.cuda.synchronize(); t=time.monotonic(); count=sum(run(rows[i:i+4]) for i in range(0,len(rows),4)); torch.cuda.synchronize(); elapsed=time.monotonic()-t
d={'dtype':dtype,'micro_batch':4,'max_tokens':512,'n_responses':8,'tokens':count,'elapsed_seconds':elapsed,'tokens_per_second':count/elapsed}
(p/f'hf_generation_{dtype}.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
