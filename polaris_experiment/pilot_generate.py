import json,random,time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
from base_experiment.prompt import base_prompt
model='Qwen/Qwen3-4B-Base';rows=[json.loads(x) for x in Path('runs/polaris_4b/prepared/train.jsonl').read_text().splitlines()];rows=random.Random(2026).sample(rows,100);tok=AutoTokenizer.from_pretrained(model,revision='906bfd4b4dc7f14ee4320094d8b41684abff8539')
prompts=[tok(base_prompt(r['problem']),add_special_tokens=False).input_ids for r in rows];requests=[{'prompt_token_ids':p} for p in prompts for _ in range(4)]
params=[SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=8192,seed=3100+i,detokenize=False,ignore_eos=True,stop_token_ids=[tok.eos_token_id],include_stop_str_in_output=True) for i in range(len(requests))]
t=time.monotonic();engine=LLM(model=model,revision='906bfd4b4dc7f14ee4320094d8b41684abff8539',dtype='bfloat16',max_model_len=max(map(len,prompts))+8192,gpu_memory_utilization=.8,skip_tokenizer_init=True,max_num_seqs=8,enable_prefix_caching=False);init=time.monotonic()-t
t=time.monotonic();outputs=engine.generate(requests,params,use_tqdm=True);seconds=time.monotonic()-t
out=Path('runs/polaris_4b/pilot');out.mkdir(parents=True,exist_ok=True)
with (out/'raw.jsonl').open('w') as f:
 for i,o in enumerate(outputs):
  c=o.outputs[0];ids=list(c.token_ids)
  if c.finish_reason=='stop':assert ids and ids[-1]==tok.eos_token_id
  assert tok.eos_token_id not in ids[:-1]
  f.write(json.dumps({'problem_id':rows[i//4]['id'],'problem':rows[i//4]['problem'],'gold':rows[i//4]['gold'],'sample':i%4,'token_ids':ids,'finish_reason':c.finish_reason})+'\n')
(out/'generation.json').write_text(json.dumps({'problems':100,'samples_per_problem':4,'model':model,'model_eos_token_id':tok.eos_token_id,'init_seconds':init,'generation_seconds':seconds,'generated_tokens':sum(len(x.outputs[0].token_ids) for x in outputs),'prompt_template':'plain math problem plus explicit boxed answer instruction','sampling':{'temperature':1,'top_p':1,'max_tokens':8192}},indent=2))
