import json, os, random, time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
model='Qwen/Qwen2.5-Math-1.5B-Instruct'; n=int(os.getenv('PILOT_BATCH_SIZE','64')); max_tokens=int(os.getenv('PILOT_MAX_TOKENS','2048')); out=Path(os.environ['PILOT_OUT']); out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(s) for s in Path('runs/viability_prepared/train.jsonl').read_text().splitlines()]; rows=random.Random(42).sample(rows,n)
tok=AutoTokenizer.from_pretrained(model); prompts=[]
for row in rows:
 s=tok.apply_chat_template([{'role':'user','content':row['problem']+'\n\nSolve step by step and put your final answer in \\boxed{...}.'}],tokenize=False,add_generation_prompt=True)
 prompts.append({'prompt_token_ids':tok(s,add_special_tokens=False)['input_ids']})
max_context=max(len(x['prompt_token_ids']) for x in prompts)+max_tokens
started=time.monotonic();llm=LLM(model=model,dtype='bfloat16',max_model_len=max_context,gpu_memory_utilization=0.8,seed=42); init=time.monotonic()-started
llm.generate(prompts[:1],SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=8,seed=91),use_tqdm=False)
started=time.monotonic();outputs=llm.generate(prompts,SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=max_tokens,logprobs=1,seed=42),use_tqdm=False);elapsed=time.monotonic()-started
records=[]
for row,o,prompt in zip(rows,outputs,prompts):
 c=o.outputs[0]; ids=list(c.token_ids); assert list(o.prompt_token_ids)==prompt['prompt_token_ids']; assert len(ids)==len(c.logprobs)
 if c.finish_reason=='stop': assert ids[-1]==tok.eos_token_id, 'Stopping token missing; cannot reproduce terminal action'
 records.append({'prompt':row['problem'],'answer':str(row['gold']),'prompt_token_ids':list(o.prompt_token_ids),'token_ids':ids,'behavior_log_probs':[float(d[t].logprob) for t,d in zip(ids,c.logprobs)],'finish_reason':c.finish_reason})
(out/'rollouts.json').write_text(json.dumps(records)); count=sum(len(r['token_ids']) for r in records)
(out/'generation.json').write_text(json.dumps({'n':n,'tokens':count,'max_tokens':max_tokens,'engine_start_seconds':init,'generation_seconds':elapsed,'generation_tps':count/elapsed},indent=2)); print((out/'generation.json').read_text())
