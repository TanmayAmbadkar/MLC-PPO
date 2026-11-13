#!/usr/bin/env python3
"""Independently sample prompt-level difficulty, separate from state continuations."""
import argparse, hashlib, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reward import reward
p=argparse.ArgumentParser();p.add_argument('--manifest-dir',default='runs/experiment2/prepared');p.add_argument('--output-dir',default='runs/experiment2/difficulty')
p.add_argument('--model',default='Qwen/Qwen2.5-Math-1.5B-Instruct');p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
p.add_argument('--train-limit',type=int,default=500);p.add_argument('--validation-limit',type=int,default=200);p.add_argument('--test-limit',type=int,default=200)
p.add_argument('--samples',type=int,default=8);p.add_argument('--max-new-tokens',type=int,default=1024);p.add_argument('--temperature',type=float,default=.8);p.add_argument('--top-p',type=float,default=.95);p.add_argument('--seed',type=int,default=12345)
a=p.parse_args()
if not 0<=a.rank<a.world_size:raise ValueError('Invalid rank')
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);config={k:v for k,v in vars(a).items() if k not in ('rank','world_size')}
path=out/'config.json'
if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Difficulty config differs; use a new directory')
if not path.exists():
 tmp=out/f'config-{a.rank}.tmp';tmp.write_text(json.dumps(config,indent=2));tmp.replace(path)
tok=AutoTokenizer.from_pretrained(a.model)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
for param in model.parameters():param.requires_grad_(False)
for split,limit in [('train',a.train_limit),('validation',a.validation_limit),('test',a.test_limit)]:
 rows=[json.loads(x) for x in (Path(a.manifest_dir)/(split+'.jsonl')).read_text().splitlines()][:limit]
 dest=out/split;dest.mkdir(exist_ok=True)
 for i,row in enumerate(rows):
  if i%a.world_size!=a.rank:continue
  target=dest/(row['id']+'.json')
  if target.exists():continue
  seed=int(hashlib.sha256((row['id']+':difficulty:'+str(a.seed)).encode()).hexdigest()[:8],16)
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  prompt=row['problem']+'\n\nSolve step by step and put your final answer in \\boxed{...}.'
  formatted=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
  ids=tok(formatted,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
  with torch.inference_mode():
   generated=model.generate(ids.repeat(a.samples,1),do_sample=True,temperature=a.temperature,top_p=a.top_p,top_k=50,
       max_new_tokens=a.max_new_tokens,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
  responses=[]
  for tokens in generated[:,ids.shape[1]:]:
   eos=(tokens==tok.eos_token_id).nonzero(as_tuple=True)[0];ended=bool(len(eos))
   action_tokens=tokens[:int(eos[0])+1] if ended else tokens
   text=tok.decode(action_tokens,skip_special_tokens=True)
   responses.append({'text':text,'token_ids':action_tokens.tolist(),'reward':reward(text,row['gold']),'ended_with_eos':ended})
  metadata={'problem_id':row['id'],'split':split,'question':row['problem'],'gold':row['gold'],
            'formatted_prompt':formatted,'prompt_token_ids':ids[0].tolist(),'seed':seed,'success_rate':sum(x['reward'] for x in responses)/len(responses),
            'responses':responses}
  tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(metadata,ensure_ascii=False));tmp.replace(target)
  print(split,'rank',a.rank,'problem',i+1,'/',len(rows),'rate',metadata['success_rate'],flush=True)
