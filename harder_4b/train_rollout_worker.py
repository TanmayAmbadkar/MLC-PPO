"""Isolated rollout worker; input and snapshot are owned by the calling training job."""
import json, os, sys, time
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
request=Path(sys.argv[1]); data=json.loads(request.read_text()); t=time.monotonic()
llm=LLM(model=data['model'],revision=data['revision'],enable_lora=True,max_lora_rank=32,dtype='bfloat16',max_model_len=data['max_model_len'],gpu_memory_utilization=float(os.environ.get('FAST_VLLM_GPU_FRACTION','0.35')),seed=data['seed'],enable_prefix_caching=False,skip_tokenizer_init=True)
init=time.monotonic()-t; t=time.monotonic()
outputs=llm.generate([{'prompt_token_ids':x} for x in data['prompt_token_ids']],[SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=data['max_tokens'],logprobs=1,seed=data['seed']+i,detokenize=False,ignore_eos=True,stop_token_ids=[data['eos_token_id']],include_stop_str_in_output=True) for i in range(len(data['prompt_token_ids']))],lora_request=LoRARequest('current',1,data['snapshot']),use_tqdm=False)
records=[]
for p,o in zip(data['prompt_token_ids'],outputs):
 c=o.outputs[0]; ids=list(c.token_ids); assert list(o.prompt_token_ids)==p and len(ids)==len(c.logprobs)
 records.append({'tokens':ids,'log_probs':[float(d[t].logprob) for t,d in zip(ids,c.logprobs)],'finish_reason':c.finish_reason})
Path(data['output']).write_text(json.dumps({'records':records,'init_seconds':init,'generation_seconds':time.monotonic()-t}))
