#!/usr/bin/env python3
"""Regenerate and record auditable responses for the low-success test problems."""
import json
import sys
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_mlc_ppo import normalize_answer, answer_reward

root=Path('runs/experiment1')
config=json.loads((root/'data/config.json').read_text())
problems=json.loads((root/'diagnostics/diagnostic.json').read_text())['low_success_problem_ids']
out=root/'verifier_audit'; out.mkdir(parents=True,exist_ok=True)
tokenizer=AutoTokenizer.from_pretrained(config['model'])
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(config['model'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
dataset=load_dataset('openai/gsm8k','main',split='train')
metadata={'source_experiment_config':config,'audit_seed_base':100042,'rollouts_per_problem':16,
          'generation':'fresh full responses; original Experiment 1 prefix texts were not saved'}
(out/'config.json').write_text(json.dumps(metadata,indent=2))
with (out/'responses.jsonl').open('w',encoding='utf-8') as f, torch.inference_mode():
    for problem_id in problems:
        row=dataset[problem_id]
        prompt='Solve the following math problem. Show your reasoning, then put the final answer in \\boxed{...}.\n\n'+row['question']
        text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        inputs=tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
        for i in range(16):
            seed=100042+problem_id*100+i
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            generated=model.generate(inputs,do_sample=True,temperature=config['temperature'],top_p=config['top_p'],
                max_new_tokens=config['max_new_tokens'],pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,use_cache=True)
            tokens=generated[0,inputs.shape[1]:]
            response=tokenizer.decode(tokens,skip_special_tokens=True)
            record={'problem_id':problem_id,'question':row['question'],'gold_answer':row['answer'],
                    'response':response,'extracted_answer':normalize_answer(response),
                    'extracted_gold':normalize_answer(row['answer']),'reward':answer_reward(response,row['answer']),
                    'seed':seed,'generated_tokens':len(tokens),
                    'ended_with_eos':bool(len(tokens) and tokens[-1].item()==tokenizer.eos_token_id)}
            f.write(json.dumps(record,ensure_ascii=False)+'\n'); f.flush()
        print(f'completed problem {problem_id}',flush=True)
