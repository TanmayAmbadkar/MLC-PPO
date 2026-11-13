#!/usr/bin/env python3
"""Measure EOS and verifier rates at a longer fixed response budget."""
import json,random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from reward import reward
root=Path('runs/experiment2');out=root/'length_pilot';out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(s) for s in (root/'smoke_v2/responses.jsonl').read_text().splitlines()][:25]
if len(rows)!=25:raise ValueError('Need 25 smoke problems')
model_id='Qwen/Qwen2.5-Math-1.5B-Instruct';tok=AutoTokenizer.from_pretrained(model_id);tok.padding_side='left'
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
for p in model.parameters():p.requires_grad_(False)
random.seed(53);torch.manual_seed(53);torch.cuda.manual_seed_all(53)
records=[]
with (out/'responses.jsonl').open('w') as f,torch.inference_mode():
 for i in range(0,len(rows),2):
  batch=rows[i:i+2];encoded=tok([r['formatted_prompt'] for r in batch],return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
  generated=model.generate(**encoded,do_sample=True,temperature=.8,top_p=.95,top_k=50,max_new_tokens=4096,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
  actions=generated[:,encoded.input_ids.shape[1]:];texts=tok.batch_decode(actions,skip_special_tokens=True)
  for r,tokens,text in zip(batch,actions,texts):
   eos=(tokens==tok.eos_token_id).nonzero(as_tuple=True)[0];ended=bool(len(eos))
   record={'id':r['problem_id'],'question':r['problem'],'gold':r['gold'],'formatted_prompt':r['formatted_prompt'],
           'completion':text,'reward':reward(text,r['gold']),'ended_with_eos':ended,
           'generated_tokens':int(eos[0])+1 if ended else len(tokens)}
   records.append(record);f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
  print('generated',len(records),'of',len(rows),flush=True)
report={'problems':len(records),'max_new_tokens':4096,'temperature':.8,'top_p':.95,'top_k':50,'seed':53,
        'ended_with_eos':sum(r['ended_with_eos'] for r in records),'boxed_answers':sum('\\boxed{' in r['completion'] for r in records),
        'successes':sum(r['reward'] for r in records),'generated_token_counts':[r['generated_tokens'] for r in records],
        'sampling_note':'25 fresh responses on the first 25 smoke problems, not selected by earlier success or truncation; small sample.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
