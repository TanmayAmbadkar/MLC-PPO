"""Greedy vLLM evaluation of a PEFT checkpoint, saved incrementally."""
import json,sys,time
from pathlib import Path
from vllm import LLM,SamplingParams
from vllm.lora.request import LoRARequest
req=json.loads(Path(sys.argv[1]).read_text());start=time.monotonic()
llm=LLM(model=req['model'],revision=req['revision'],dtype='bfloat16',enable_lora=True,max_lora_rank=32,max_model_len=req['max_model_len'],max_num_seqs=16,gpu_memory_utilization=.8,skip_tokenizer_init=True,enable_prefix_caching=False)
params=SamplingParams(temperature=0,max_tokens=8192,ignore_eos=True,stop_token_ids=[req['eos']],include_stop_str_in_output=True,detokenize=False)
with Path(req['output']).open('w') as f:
 for i in range(0,len(req['items']),16):
  batch=req['items'][i:i+16];out=llm.generate([{'prompt_token_ids':x['ids']} for x in batch],params,lora_request=LoRARequest('eval',1,req['adapter']) if req['adapter'] else None,use_tqdm=False)
  for item,got in zip(batch,out):
   o=got.outputs[0];f.write(json.dumps({'name':item['name'],'index':item['index'],'tokens':list(o.token_ids),'finish_reason':o.finish_reason})+'\n')
  f.flush();print(f'evaluated {min(i+len(batch),len(req["items"]))}/{len(req["items"])}',flush=True)
print('seconds',time.monotonic()-start,flush=True)
