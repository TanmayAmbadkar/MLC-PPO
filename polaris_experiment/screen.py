"""Independent difficulty screening for Qwen3 4B; preserves every completion."""
import json,random,time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
from base_experiment.prompt import base_prompt
MODEL='Qwen/Qwen3-4B-Base'; REV='906bfd4b4dc7f14ee4320094d8b41684abff8539'; ROOT=Path('runs/polaris_4b/screen');ROOT.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in Path('runs/polaris_4b/prepared/train_context_valid.jsonl').read_text().splitlines()]
pilot_file=Path('runs/polaris_4b/pilot/scored.jsonl')
exclude={json.loads(x)['problem_id'] for x in pilot_file.read_text().splitlines()} if pilot_file.exists() else set()
rows=random.Random(491).sample([r for r in rows if r['id'] not in exclude],1000)
tok=AutoTokenizer.from_pretrained(MODEL,revision=REV);prompts=[tok(base_prompt(r['problem']),add_special_tokens=False).input_ids for r in rows]
engine=LLM(model=MODEL,revision=REV,dtype='bfloat16',max_model_len=max(map(len,prompts))+8192,gpu_memory_utilization=0.8,skip_tokenizer_init=True,max_num_seqs=16,enable_prefix_caching=False)
path=ROOT/'raw.jsonl'; done=set()
if path.exists():
 for line in path.read_text().splitlines():
  x=json.loads(line);done.add((x['problem_id'],x['sample']))
start=time.monotonic()
with path.open('a') as f:
 for first in range(0,len(rows),4):
  jobs=[(i,j) for i in range(first,min(first+4,len(rows))) for j in range(8) if (rows[i]['id'],j) not in done]
  if not jobs:continue
  params=[SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=8192,seed=200000+i*8+j,detokenize=False,ignore_eos=True,stop_token_ids=[tok.eos_token_id],include_stop_str_in_output=True) for i,j in jobs]
  outs=engine.generate([{'prompt_token_ids':prompts[i]} for i,j in jobs],params,use_tqdm=False)
  for (i,j),o in zip(jobs,outs):
   c=o.outputs[0];ids=list(c.token_ids)
   if not ids or (c.finish_reason=='stop' and ids[-1]!=tok.eos_token_id) or tok.eos_token_id in ids[:-1]:raise RuntimeError('Invalid generated tokens')
   f.write(json.dumps({'problem_id':rows[i]['id'],'problem':rows[i]['problem'],'gold':rows[i]['gold'],'sample':j,'seed':200000+i*8+j,'token_ids':ids,'finish_reason':c.finish_reason})+'\n')
  f.flush();print(f'screened {min(first+4,len(rows))}/{len(rows)} problems, elapsed {time.monotonic()-start:.1f}s',flush=True)
(ROOT/'generation_config.json').write_text(json.dumps({'model':MODEL,'revision':REV,'candidate_count':len(rows),'sample_count':8,'temperature':1,'top_p':1,'max_tokens':8192,'selection_seed':491,'excluded_pilot_problems':len(exclude)},indent=2))
