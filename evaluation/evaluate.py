#!/usr/bin/env python3
"""Greedy held-out DAPO evaluation with per-problem records and restart support."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from math_verify import parse
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment2.reward import PRED_CONFIG,reward,valid_gold
from train_mlc_ppo import format_prompt

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--name',required=True)
p.add_argument('--dataset-dir',default='runs/experiment2/prepared');p.add_argument('--output-dir',required=True)
p.add_argument('--max-problems',type=int,default=200);p.add_argument('--max-prompt-length',type=int,default=2048)
p.add_argument('--max-new-tokens',type=int,default=2048);a=p.parse_args()
root=Path(a.dataset_dir)
splits={n:[json.loads(s) for s in (root/(n+'.jsonl')).read_text().splitlines()] for n in ('train','validation','test')}
ids={n:{r['id'] for r in rows} for n,rows in splits.items()}
if len(ids['test'])!=len(splits['test']) or ids['test'] & (ids['train'] | ids['validation']):raise ValueError('Problem IDs overlap or repeat')
if a.max_problems<1 or a.max_problems>len(splits['test']):raise ValueError('Invalid evaluation size')
rows=splits['test'][:a.max_problems]
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
config=vars(a)|{'decoding':'greedy','policy_dtype':'float32','dataset_sha256':hashlib.sha256((root/'test.jsonl').read_bytes()).hexdigest()}
config_path=out/'config.json'
if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Evaluation configuration differs; choose a new output directory')
config_path.write_text(json.dumps(config,indent=2))
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model_id='Qwen/Qwen2.5-Math-1.5B-Instruct'
tok=AutoTokenizer.from_pretrained(model_id)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
for param in model.parameters():param.requires_grad_(False)
gen=GenerationConfig(do_sample=False,num_beams=1,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,max_new_tokens=a.max_new_tokens)
for i,row in enumerate(rows):
 path=out/(row['id']+'.json')
 if path.exists():continue
 if not valid_gold(row['gold']):raise ValueError(f"Unparseable gold for {row['id']}")
 prompt=row['problem']+'\n\nSolve step by step and put your final answer in \\boxed{...}.'
 formatted=format_prompt(tok,prompt,model_id)
 inputs=tok(formatted,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
 if inputs.shape[1]>a.max_prompt_length:raise ValueError(f"Prompt too long for {row['id']}: {inputs.shape[1]}")
 started=time.monotonic()
 with torch.inference_mode():generated=model.generate(input_ids=inputs,generation_config=gen)
 tokens=generated[0,inputs.shape[1]:];ended=bool(len(tokens) and tokens[-1].item()==tok.eos_token_id)
 text=tok.decode(tokens,skip_special_tokens=True)
 record={'id':row['id'],'problem':row['problem'],'gold':row['gold'],'formatted_prompt':formatted,
         'completion':text,'parsed_answer':[str(x) for x in parse(text,extraction_config=PRED_CONFIG,extraction_mode='first_match')],
         'reward':reward(text,row['gold']),'ended_with_eos':ended,'completion_tokens':int(len(tokens)),
         'elapsed_seconds':time.monotonic()-started}
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,ensure_ascii=False));tmp.replace(path)
 print(a.name,'problem',i+1,'/',len(rows),'reward',record['reward'],'eos',ended,'tokens',len(tokens),flush=True)
print(a.name,'evaluation complete',flush=True)
