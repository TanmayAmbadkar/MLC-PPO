"""Generate evaluation responses in batches in the isolated vLLM environment."""
import json,sys,time
from pathlib import Path
from vllm import LLM,SamplingParams
req=json.loads(Path(sys.argv[1]).read_text());t=time.monotonic()
engine=LLM(model=req['model'],dtype='float32',max_model_len=req['max_model_len'],gpu_memory_utilization=0.8,max_num_seqs=req['batch_size'],skip_tokenizer_init=True,enable_prefix_caching=False)
init=time.monotonic()-t;count=0;generation=0.0
with Path(req['results']).open('w') as f:
 for start in range(0,len(req['items']),req['batch_size']):
  batch=req['items'][start:start+req['batch_size']];params=SamplingParams(temperature=0,max_tokens=2048,ignore_eos=True,stop_token_ids=[req['eos']],include_stop_str_in_output=True,detokenize=False)
  t=time.monotonic();outs=engine.generate([{'prompt_token_ids':r['ids']} for r in batch],params,use_tqdm=False);secs=time.monotonic()-t;generation+=secs
  for r,o in zip(batch,outs):
   c=o.outputs[0];tokens=list(c.token_ids)
   if c.finish_reason=='stop' and (not tokens or tokens[-1]!=req['eos']):raise RuntimeError('Missing EOS after stop')
   if req['eos'] in tokens[:-1]:raise RuntimeError('Tokens after EOS')
   f.write(json.dumps({'name':r['name'],'i':r['i'],'token_ids':tokens,'finish_reason':c.finish_reason,'batch_seconds':secs,'batch_count':len(batch)})+'\n');count+=len(tokens)
  f.flush();print(f'finished {min(start+len(batch),len(req["items"]))}/{len(req["items"])}',flush=True)
Path(req['timing']).write_text(json.dumps({'engine_init_seconds':init,'generation_seconds':generation,'generated_tokens':count,'tokens_per_second':count/generation if generation else None},indent=2))
