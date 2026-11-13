"""Single-process vLLM rollout with an explicitly loaded LoRA adapter."""
import json, sys, time
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
req=json.loads(Path(sys.argv[1]).read_text())
t=time.monotonic()
engine=LLM(model=req['model'], revision=req['revision'], dtype='bfloat16', enable_lora=True,
           max_lora_rank=32, max_model_len=req['max_model_len'], max_num_seqs=1,
           gpu_memory_utilization=0.80, skip_tokenizer_init=True, enable_prefix_caching=False)
init=time.monotonic()-t
params=SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=req['max_tokens'],seed=1234,
                      detokenize=False,logprobs=1,ignore_eos=True,
                      stop_token_ids=[req['eos']],include_stop_str_in_output=True)
t=time.monotonic()
out=engine.generate([{'prompt_token_ids':req['prompt_ids']}],params,
                    lora_request=LoRARequest('pilot',1,req['adapter']),use_tqdm=False)[0].outputs[0]
elapsed=time.monotonic()-t
ids=list(out.token_ids)
if not ids: raise RuntimeError('Empty rollout')
result={'tokens':ids,'log_probs':[float(d[token].logprob) for d,token in zip(out.logprobs,ids)],
        'finish_reason':out.finish_reason,'engine_init_seconds':init,'generation_seconds':elapsed}
Path(req['output']).write_text(json.dumps(result))
