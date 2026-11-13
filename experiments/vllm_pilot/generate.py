import json, random, time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_id='Qwen/Qwen2.5-Math-1.5B-Instruct'
outdir=Path('runs/vllm_pilot'); outdir.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in Path('runs/viability_prepared/train.jsonl').read_text().splitlines()]
rows=random.Random(99).sample(rows,8)
tokenizer=AutoTokenizer.from_pretrained(model_id)
prompts=[tokenizer.apply_chat_template([{'role':'user','content':r['problem']+'\n\nSolve step by step and put your final answer in \\boxed{...}.'}],tokenize=False,add_generation_prompt=True) for r in rows]
t0=time.monotonic()
llm=LLM(model=model_id,dtype='bfloat16',max_model_len=1024,gpu_memory_utilization=0.8,seed=99,trust_remote_code=True)
load=time.monotonic()-t0
llm.generate(prompts[:1],SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=8,seed=91),use_tqdm=False)
t0=time.monotonic()
outputs=llm.generate(prompts,SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=512,logprobs=1,seed=99),use_tqdm=False)
elapsed=time.monotonic()-t0
records=[]
for row,out in zip(rows,outputs):
 completion=out.outputs[0]; ids=list(completion.token_ids)
 logps=[float(d[t].logprob) for t,d in zip(ids,completion.logprobs)]
 assert len(ids)==len(logps)
 records.append({'id':row['id'],'gold':row['gold'],'prompt_token_ids':list(out.prompt_token_ids),'token_ids':ids,'sampled_log_probs':logps,'text':completion.text,'finish_reason':completion.finish_reason})
(outdir/'rollouts.json').write_text(json.dumps(records))
count=sum(len(x['token_ids']) for x in records)
(outdir/'generation.json').write_text(json.dumps({'model':model_id,'dtype':'bfloat16','sampled_logprobs':True,'n_responses':len(records),'tokens':count,'load_seconds':load,'elapsed_seconds':elapsed,'tokens_per_second':count/elapsed,'max_tokens':512,'temperature':1.0,'top_p':1.0},indent=2))
print((outdir/'generation.json').read_text())
