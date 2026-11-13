"""Memory pilot: one GPU LoRA actor, optional second GPU full critic.

CPU FP32 AdamW master parameters and optimizer states permit a full BF16 critic
on one GPU. Repeated sampled tokens are used only as an 8192-token stress test.
"""
import json,os,subprocess,tempfile,time
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig,AutoModel,AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,get_peft_model
from base_experiment.prompt import base_prompt
from fast_vllm_rollout import _worker_environment
MODEL='Qwen/Qwen3-4B-Base';REV='906bfd4b4dc7f14ee4320094d8b41684abff8539';CAP=int(os.environ.get('PILOT_TOKENS','8192'));MODE=os.environ['PILOT_MODE']
assert MODE in ('actor','full_critic')
assert torch.cuda.device_count()>=(2 if MODE=='full_critic' else 1)
outdir=Path('runs/harder_4b/single_gpu_pilot')/MODE;outdir.mkdir(parents=True,exist_ok=True)
for i in range(2 if MODE=='full_critic' else 1):
 torch.cuda.set_device(i);torch.cuda.init();torch.cuda.reset_peak_memory_stats(i)
torch.cuda.set_device(0)
tok=AutoTokenizer.from_pretrained(MODEL,revision=REV);cfg=AutoConfig.from_pretrained(MODEL,revision=REV)
row=json.loads(Path('runs/base_rlvr/prepared/train.jsonl').read_text().splitlines()[0]);prompt=tok(base_prompt(row['problem']),add_special_tokens=False).input_ids
assert len(prompt)+CAP<=cfg.max_position_embeddings
actor=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
actor=get_peft_model(actor,LoraConfig(r=32,lora_alpha=64,lora_dropout=0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});actor.train()
with tempfile.TemporaryDirectory(prefix='single-lora-',dir='/tmp') as tmp:
 tmp=Path(tmp);t=time.monotonic();actor.save_pretrained(tmp/'adapter');save_seconds=time.monotonic()-t
 req={'model':MODEL,'revision':REV,'adapter':str(tmp/'adapter'),'max_model_len':len(prompt)+CAP,'prompt_ids':prompt,'max_tokens':CAP,'eos':tok.eos_token_id,'output':str(tmp/'rollout.json')}
 (tmp/'request.json').write_text(json.dumps(req)); env=_worker_environment();env['VLLM_HOST_IP']='127.0.0.1'
 t=time.monotonic();subprocess.run(['/scratch/tsa5252/.venvs/vllm-pilot/bin/python','harder_4b/lora_rollout_worker.py',str(tmp/'request.json')],env=env,check=True);worker_seconds=time.monotonic()-t
 roll=json.loads((tmp/'rollout.json').read_text());sampled=roll['tokens'];stress=(sampled*((CAP+len(sampled)-1)//len(sampled)))[:CAP]
 ids=torch.tensor([prompt+stress],device='cuda:0');start=len(prompt)-1
 actor=actor.to('cuda:0');aopt=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=5e-6)
 def logp():
  logits=actor(input_ids=ids,use_cache=False).logits[:,start:start+CAP];target=ids[:,start+1:start+1+CAP];return torch.cat([torch.log_softmax(logits[:,i:i+64].float(),-1).gather(-1,target[:,i:i+64,None]).squeeze(-1) for i in range(0,CAP,64)],dim=1)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):old=logp().detach()
 critic=None;value_head=None;copt=None;masters=None;critic_seconds=None
 if MODE=='full_critic':
  critic=AutoModel.from_pretrained(MODEL,revision=REV,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:1')
  critic.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});critic.train();value_head=nn.Linear(cfg.hidden_size,1,dtype=torch.float32,device='cuda:1')
  params=list(critic.parameters())+list(value_head.parameters())
  masters=[nn.Parameter(p.detach().float().cpu()) for p in params];copt=torch.optim.AdamW(masters,lr=1e-5)
  torch.cuda.synchronize(1);t=time.monotonic()
  with torch.autocast('cuda',dtype=torch.bfloat16):h=critic(input_ids=ids.to('cuda:1'),use_cache=False).last_hidden_state[:,start:start+CAP];values=value_head(h.float()).squeeze(-1)
  advantage=(0.5-values.detach().to('cuda:0')).float()
 else:advantage=torch.full_like(old,0.5)
 torch.cuda.synchronize(0);ta=time.monotonic()
 with torch.autocast('cuda',dtype=torch.bfloat16):p=logp();ratio=(p-old).exp();loss=-torch.minimum(ratio*advantage,ratio.clamp(0.8,1.2)*advantage).sum()/CAP
 if not torch.isfinite(loss):raise RuntimeError('Nonfinite actor loss')
 loss.backward();aopt.step();aopt.zero_grad(set_to_none=True);torch.cuda.synchronize(0);actor_seconds=time.monotonic()-ta
 ta=time.monotonic();actor.save_pretrained(tmp/'updated');updated_save=time.monotonic()-ta
 if MODE=='full_critic':
  c_loss=(values.float()-0.5).square().mean()
  if not torch.isfinite(c_loss):raise RuntimeError('Nonfinite critic loss')
  c_loss.backward()
  for p,m in zip(params,masters):m.grad=p.grad.detach().float().cpu() if p.grad is not None else None
  copt.step()
  with torch.no_grad():
   for p,m in zip(params,masters):p.copy_(m.to(device=p.device,dtype=p.dtype))
  copt.zero_grad(set_to_none=True);critic.zero_grad(set_to_none=True);value_head.zero_grad(set_to_none=True);torch.cuda.synchronize(1);critic_seconds=time.monotonic()-t
 result={'model':MODEL,'revision':REV,'mode':MODE,'max_completion':CAP,'sampled_tokens':len(sampled),'repeated_to_stress_cap':len(sampled)<CAP,'actor':'LoRA r32 alpha64 attention+MLP','critic':'separate full BF16 model; FP32 CPU AdamW masters, gradients and optimizer states' if critic is not None else None,'initial_adapter_save_seconds':save_seconds,'vllm_engine_init_seconds':roll['engine_init_seconds'],'vllm_generation_seconds':roll['generation_seconds'],'rollout_worker_seconds':worker_seconds,'actor_update_seconds':actor_seconds,'updated_adapter_save_seconds':updated_save,'critic_forward_backward_cpu_optimizer_seconds':critic_seconds,'peak_reserved_bytes':[torch.cuda.max_memory_reserved(i) for i in range(2 if MODE=='full_critic' else 1)],'peak_allocated_bytes':[torch.cuda.max_memory_allocated(i) for i in range(2 if MODE=='full_critic' else 1)],'note':'Synthetic stress update, not a learning result; worker GPU peak and live adapter reload overhead are not measured by PyTorch peak counters.'}
 (outdir/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
