#!/usr/bin/env python3
"""Held-out greedy benchmark evaluation of viability checkpoints."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment2.reward import reward,valid_gold,require_validated_versions
from train_mlc_ppo import format_prompt
from evaluation.math500 import format_gold
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True);versions=require_validated_versions()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B-Instruct');tok.pad_token=tok.pad_token or tok.eos_token
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
for parameter in model.parameters():parameter.requires_grad_(False)
gen=GenerationConfig(do_sample=False,num_beams=1,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,max_new_tokens=2048)
specs=[('math500','HuggingFaceH4/MATH-500','test','6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'),('aime2024','HuggingFaceH4/aime_2024','train','2fe88a2f1091d5048c0f36abc874fb997b3dd99a')];summaries={}
for name,repo,split,revision in specs:
 rows=load_dataset(repo,split=split,revision=revision);rows=rows.select(range(min(a.limit,len(rows)))) if a.limit else rows;out=root/name;out.mkdir(exist_ok=True);records=[]
 for i,row in enumerate(rows):
  target=out/f'{i:04d}.json'
  if target.exists():records.append(json.loads(target.read_text()));continue
  gold=format_gold(row['answer']);problem=row['problem'];pid=str(row.get('unique_id',row.get('id',i)));record={'id':pid,'problem':problem,'original_gold':row['answer'],'verifier_gold':gold}
  if not valid_gold(gold):record.update(status='excluded_unparseable_gold',reward=None)
  else:
   prompt=problem+'\n\nSolve step by step and put your final answer in \\boxed{...}.';formatted=format_prompt(tok,prompt,a.model);ids=tok(formatted,return_tensors='pt',add_special_tokens=False).input_ids.cuda();started=time.monotonic()
   with torch.inference_mode():result=model.generate(input_ids=ids,generation_config=gen)
   completion=result[0,ids.shape[1]:];text=tok.decode(completion,skip_special_tokens=True);record.update(status='scored',completion=text,reward=reward(text,gold),ended_with_eos=bool(len(completion) and completion[-1].item()==tok.eos_token_id),completion_tokens=len(completion),seconds=time.monotonic()-started)
  tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(record,ensure_ascii=False));tmp.replace(target);records.append(record);print(name,i+1,'/',len(rows),'reward',record['reward'],flush=True)
 scored=[r for r in records if r['status']=='scored'];summaries[name]={'dataset':repo,'revision':revision,'total':len(records),'scored':len(scored),'correct':sum(r['reward'] for r in scored),'accuracy':sum(r['reward'] for r in scored)/len(scored) if scored else None,'truncated':sum(not r['ended_with_eos'] for r in scored)}
(root/'summary.json').write_text(json.dumps({'model':a.model,'decoding':'greedy','max_new_tokens':2048,'reward_versions':versions,'benchmarks':summaries,'note':'Exact and normalized decontamination only; benchmark contamination and gold correctness are not guaranteed.'},indent=2));print(json.dumps(summaries,indent=2))
