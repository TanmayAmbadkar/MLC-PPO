"""Measure persistent vLLM sleep/wake and changed-weight synchronization."""
import json, tempfile, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from vllm import LLM, SamplingParams
def main():
    model_id='Qwen/Qwen2.5-Math-1.5B-Instruct'; out=Path('runs/persistent_vllm_pilot');out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(model_id); actor=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
    # Match resident FP32 Adam moments after an optimizer update; these are simulated,
    # not used for training.
    states=[torch.empty_like(p) for p in actor.parameters() for _ in range(2)]
    torch.cuda.synchronize();free0,total=torch.cuda.mem_get_info()
    t0=time.monotonic();engine=LLM(model=model_id,dtype='bfloat16',max_model_len=512,gpu_memory_utilization=0.35,skip_tokenizer_init=True,enable_sleep_mode=True,enable_prefix_caching=False);init=time.monotonic()-t0
    prompts=[]
    for i in range(8):
     text=tok.apply_chat_template([{'role':'user','content':f'Solve step by step and put your final answer in \\boxed{{...}}: What is {i+13} times {i+17}?'}],tokenize=False,add_generation_prompt=True)
     prompts.append({'prompt_token_ids':tok(text,add_special_tokens=False).input_ids})
    def probe(m):return m.model.layers[0].self_attn.qkv_proj.weight[:2,:2].float().cpu().tolist()
    def generate():
     params=[SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=128,seed=42+i,detokenize=False,ignore_eos=True,stop_token_ids=[tok.eos_token_id],include_stop_str_in_output=True) for i in range(8)]
     t=time.monotonic();o=engine.generate(prompts,params,use_tqdm=False);return time.monotonic()-t,sum(len(x.outputs[0].token_ids) for x in o)
    before=engine.apply_model(probe)[0]; gen1,n1=generate();t=time.monotonic();engine.sleep(level=1);sleep=time.monotonic()-t;free_sleep,_=torch.cuda.mem_get_info()
    # Exercise memory while vLLM is asleep to check co-residency headroom.
    activation=torch.empty(12*1024**3,dtype=torch.uint8,device='cuda');activation.fill_(1);torch.cuda.synchronize();free_during,_=torch.cuda.mem_get_info();del activation;torch.cuda.empty_cache()
    with torch.no_grad():actor.model.layers[0].self_attn.q_proj.weight.add_(0.25)
    expected=actor.model.layers[0].self_attn.q_proj.weight[:2,:2].to(torch.bfloat16).float().cpu().tolist()
    with tempfile.TemporaryDirectory(prefix='persistent_sync_',dir=out) as tmp:
     t=time.monotonic();actor.save_pretrained(tmp);snapshot=time.monotonic()-t
     t=time.monotonic();engine.wake_up();wake=time.monotonic()-t
     t=time.monotonic();engine.collective_rpc('reload_weights',kwargs={'weights_path':tmp});reload=time.monotonic()-t
     after=engine.apply_model(probe)[0];gen2,n2=generate()
    result={'engine_init_seconds':init,'first_generation_seconds':gen1,'first_tokens':n1,'sleep_seconds':sleep,'snapshot_seconds':snapshot,'wake_seconds':wake,'reload_seconds':reload,'second_generation_seconds':gen2,'second_tokens':n2,'gpu_free_before_engine_gib':free0/2**30,'gpu_free_asleep_gib':free_sleep/2**30,'gpu_free_with_12gib_activation_gib':free_during/2**30,'before_probe':before,'after_probe':after,'expected_after_bf16':expected,'weights_changed':before!=after,'weights_match':after==expected,'note':'Two short generation batches and a deliberate FP32 weight perturbation; no PPO update. Adam moments and activation allocation are simulated.'}
    (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['weights_changed'] and result['weights_match']

if __name__ == '__main__':
    main()
