#!/usr/bin/env python3
"""Greedy MATH-500 evaluation using the pinned Math-Verify reward."""
import argparse,json,sys,time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment2.reward import reward,valid_gold,require_validated_versions
from train_mlc_ppo import format_prompt

def format_gold(answer):
 text=str(answer).strip()
 if not text:return ''
 if text.startswith(('$',r'\[',r'\(')) or r'\boxed' in text:return text
 return '$'+text+'$'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output-dir',required=True)
 p.add_argument('--max-problems',type=int,default=500);p.add_argument('--max-new-tokens',type=int,default=2048)
 args=p.parse_args();root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
 config=vars(args)|{'dataset':'HuggingFaceH4/MATH-500','decoding':'greedy','dtype':'float32','reward_versions':require_validated_versions()}
 path=root/'config.json'
 if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Output directory belongs to a different configuration')
 path.write_text(json.dumps(config,indent=2));rows=load_dataset('HuggingFaceH4/MATH-500',split='test')
 if not 1<=args.max_problems<=len(rows):raise ValueError('Invalid evaluation size')
 tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B-Instruct')
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
 for param in model.parameters():param.requires_grad_(False)
 gen=GenerationConfig(do_sample=False,num_beams=1,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,max_new_tokens=args.max_new_tokens)
 records=[]
 for i,row in enumerate(rows.select(range(args.max_problems))):
  filename=root/f'{i:04d}.json'
  if filename.exists():records.append(json.loads(filename.read_text()));continue
  gold=format_gold(row['answer']);record={'id':row['unique_id'],'problem':row['problem'],'original_gold':row['answer'],'verifier_gold':gold}
  if not valid_gold(gold):
   record.update(status='excluded_unparseable_gold',reward=None)
  else:
   prompt=row['problem']+'\n\nSolve step by step and put your final answer in \\boxed{...}.'
   formatted=format_prompt(tok,prompt,'Qwen/Qwen2.5-Math-1.5B-Instruct')
   ids=tok(formatted,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
   start=time.monotonic()
   with torch.inference_mode():output=model.generate(input_ids=ids,generation_config=gen)
   tokens=output[0,ids.shape[1]:];text=tok.decode(tokens,skip_special_tokens=True)
   record.update(status='scored',formatted_prompt=formatted,completion=text,reward=reward(text,gold),
                 ended_with_eos=bool(len(tokens) and tokens[-1].item()==tok.eos_token_id),
                 completion_tokens=len(tokens),elapsed_seconds=time.monotonic()-start)
  tmp=filename.with_suffix('.tmp');tmp.write_text(json.dumps(record,ensure_ascii=False));tmp.replace(filename)
  records.append(record);print(f"MATH-500 {i+1}/{args.max_problems} {record['status']} reward={record['reward']}",flush=True)
 scored=[r for r in records if r['status']=='scored']
 summary={'total':len(records),'scored':len(scored),'excluded_unparseable_gold':len(records)-len(scored),
          'correct':int(sum(r['reward'] for r in scored)),'accuracy':sum(r['reward'] for r in scored)/len(scored) if scored else None,
          'truncated':sum(not r['ended_with_eos'] for r in scored),'decoding':'greedy','max_new_tokens':args.max_new_tokens,
          'note':'No claim of paraphrase-level decontamination or perfect gold/parser correctness.'}
 (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
