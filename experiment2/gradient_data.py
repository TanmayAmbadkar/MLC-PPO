#!/usr/bin/env python3
"""Collect independent single-trajectory policy scores on held-out problems."""
import argparse,hashlib,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from reward import reward
p=argparse.ArgumentParser();p.add_argument('--manifest-dir',default='runs/experiment2/prepared');p.add_argument('--output-dir',default='runs/experiment2/gradient_data')
p.add_argument('--model',default='Qwen/Qwen2.5-Math-1.5B-Instruct');p.add_argument('--max-problems',type=int,default=64)
p.add_argument('--max-new-tokens',type=int,default=4096);p.add_argument('--temperature',type=float,default=.8);p.add_argument('--top-p',type=float,default=.95);p.add_argument('--seed',type=int,default=9384)
a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);path=out/'config.json';config=vars(a)
if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Gradient collection config differs; use a new output directory')
path.write_text(json.dumps(config,indent=2));dest=out/'test';dest.mkdir(exist_ok=True)
rows=[json.loads(x) for x in (Path(a.manifest_dir)/'test.jsonl').read_text().splitlines()][:a.max_problems]
tok=AutoTokenizer.from_pretrained(a.model)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
for param in model.parameters():param.requires_grad_(False)
for index,row in enumerate(rows):
 target=dest/(row['id']+'.json')
 if target.exists():continue
 seed=int(hashlib.sha256((row['id']+':gradient:'+str(a.seed)).encode()).hexdigest()[:8],16)
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 prompt=row['problem']+'\n\nSolve step by step and put your final answer in \\boxed{...}.'
 formatted=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
 ids=tok(formatted,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
 with torch.inference_mode():
  generated=model.generate(ids,do_sample=True,temperature=a.temperature,top_p=a.top_p,top_k=50,max_new_tokens=a.max_new_tokens,
     pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,return_dict_in_generate=True,output_scores=True)
  actions=generated.sequences[0,ids.shape[1]:]
  sampled=[float(torch.log_softmax(s[0].float(),dim=-1)[actions[i]]) for i,s in enumerate(generated.scores)]
 text=tok.decode(actions,skip_special_tokens=True)
 record={'problem_id':row['id'],'split':'test','question':row['problem'],'gold':row['gold'],
         'formatted_prompt':formatted,'prompt_token_ids':ids[0].tolist(),'seed':seed,
         'responses':[{'text':text,'token_ids':actions.tolist(),'reward':reward(text,row['gold']),
                       'sampled_log_probs':sampled,'ended_with_eos':bool(len(actions) and actions[-1].item()==tok.eos_token_id)}]}
 tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(record,ensure_ascii=False));tmp.replace(target)
 print('gradient example',index+1,'/',len(rows),flush=True)
 del generated
